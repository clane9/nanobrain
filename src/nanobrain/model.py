import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainmarks_smri.models import EmbeddingOutput, ImageInput, register_model
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import (
    crop_to_mask,
    fit_to_shape,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)
from .losses import PrototypeCELoss, PrototypeHead, entropy_regularizer
from .modules import (
    Block,
    LayerNorm,
    Rope3D,
    SeparablePosEmbed3D,
    SinCosPosEmbed3D,
)


class ViTEncoder3D(nn.Module):
    """ViT over a subset of patch tokens. Also implements the brainmarks model contract."""

    name = "vit_encoder_3d"

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 0,
        drop_path_rate: float = 0.0,
        pos_embed: str = "learned_sep",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.target_spacing = target_spacing
        # tuple for the brainmarks-smri model contract
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.class_tokens = class_tokens
        self.pos_embed_type = pos_embed
        patch_grid_size = tuple(size // patch_size for size in grid_size)

        # patch center coords in mm, relative to the volume center
        coord_grid = make_coord_grid(grid_size, patch_size, target_spacing)
        self.register_buffer("coord_grid", coord_grid, persistent=False)
        unit = patch_size * target_spacing

        self.patch_embed = nn.Linear(patch_size**3, embed_dim)
        if pos_embed == "learned_sep":
            self.pos_embed = SeparablePosEmbed3D(patch_grid_size, embed_dim, unit=unit)
        elif pos_embed == "sincos":
            self.pos_embed = SinCosPosEmbed3D(patch_grid_size, embed_dim, unit=unit)
        elif pos_embed == "rope_axial":
            self.pos_embed = None
        else:
            raise ValueError(f"unknown pos_embed {pos_embed}")
        pos_embed_layer = Rope3D if pos_embed == "rope_axial" else None
        if class_tokens:
            self.cls_token = nn.Parameter(torch.empty(1, class_tokens, embed_dim))
        else:
            self.cls_token = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                    pos_embed_layer=pos_embed_layer,
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        self.apply(_init_weights)
        if class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)

    def extra_repr(self):
        return (
            f"{self.grid_size}, {self.patch_size}, class_tokens={self.class_tokens}, "
            f"pos_embed={self.pos_embed_type}"
        )

    def forward(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        B = patches.shape[0]
        x = self.patch_embed(patches)
        if self.pos_embed is not None:
            x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            # class tokens at the center, i.e. not rotated by rope
            coord = F.pad(coord, (0, 0, self.class_tokens, 0))

        for block in self.blocks:
            x = block(x, coord=coord)
        x = self.norm(x)
        return x

    # the methods below implement the brainmarks model contract.
    #   - forward_embeddings: computes a global and dense embedding per input sample
    #   - transform: cpu data transform, runs in loader workers (optional)
    #   - transform_cuda: cuda data transform. you call it inside forward_embeddings (optional)

    def transform(self, sample: ImageInput) -> dict[str, Tensor]:
        image = torch.from_numpy(sample["image"].get_fdata(dtype=np.float32))
        mask = torch.from_numpy(np.asarray(sample["brain_mask"].dataobj) > 0)
        affine = sample["image"].affine
        image, mask, affine = crop_to_mask(image, mask, affine)
        return {"values": image[mask], "mask": mask, "affine": affine}

    def transform_cuda(self, sample: dict[str, Tensor]) -> dict:
        values = sample["values"]
        mask = sample["mask"]
        affine = sample["affine"]
        values = normalize_values(values)
        image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
        image, mask, affine = reorient_to_ras(image, mask, affine)
        image, mask, affine = resample_to_spacing(image, mask, affine, self.target_spacing)
        image, mask, affine = fit_to_shape(image, mask, affine, self.grid_size)
        return {"image": image, "mask": mask, "affine": affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.coord_grid.device
        patch_grid_size = [size // self.patch_size[0] for size in self.grid_size]

        # one forward per image since the number of patches in the mask varies
        outputs = []
        for sample in samples:
            sample = {
                "values": sample["values"].to(device),
                "mask": sample["mask"].to(device),
                "affine": sample["affine"],
            }
            sample = self.transform_cuda(sample)
            patches = patchify3d(sample["image"][None], self.patch_size[0])
            in_mask = (patchify3d(sample["mask"][None], self.patch_size[0]) > 0).any(dim=2)[0]
            # cudnn attention builds a new graph (~45 ms) for each new sequence length
            with (
                sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                tokens = self.forward(patches[:, in_mask], self.coord_grid[:, in_mask])
            tokens = tokens[0, self.class_tokens :].float()

            output = EmbeddingOutput(
                global_embedding=tokens.mean(dim=0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = tokens.new_zeros(len(in_mask), self.embed_dim)
                dense[in_mask] = tokens
                output["dense_embedding"] = dense.reshape(*patch_grid_size, self.embed_dim)
                output["dense_affine"] = sample["affine"]
                output["dense_mask"] = in_mask.reshape(patch_grid_size)
            outputs.append(output)
        return outputs


class Predictor3D(nn.Module):
    """Predicts tokens at query positions from encoder tokens.

    self_attn: queries and projected encoder tokens attend jointly (MAE decoder, I-JEPA).
    cross_attn: queries attend only to the encoder tokens, so each prediction is independent
        (CAPI). Keys and values are projected from the encoder width directly.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        patch_size: int,
        target_spacing: float,
        embed_dim: int,
        out_dim: int,
        depth: int = 4,
        predictor_dim: int = 512,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        attn_type: str = "self_attn",
        pos_embed: str = "learned_sep",
    ):
        super().__init__()
        assert attn_type in ("self_attn", "cross_attn"), f"unknown attn_type {attn_type}"
        self.attn_type = attn_type
        patch_grid_size = tuple(size // patch_size for size in grid_size)
        unit = patch_size * target_spacing

        self.proj = nn.Linear(embed_dim, predictor_dim) if attn_type == "self_attn" else None
        self.mask_token = nn.Parameter(torch.empty(1, 1, predictor_dim))
        # added to the queries only, the encoder tokens already carry position
        if pos_embed == "learned_sep":
            self.pos_embed = SeparablePosEmbed3D(patch_grid_size, predictor_dim, unit=unit)
        elif pos_embed == "sincos":
            self.pos_embed = SinCosPosEmbed3D(patch_grid_size, predictor_dim, unit=unit)
        else:
            self.pos_embed = None
        pos_embed_layer = Rope3D if pos_embed == "rope_axial" else None

        context_dim = embed_dim if attn_type == "cross_attn" else None
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    context_dim=context_dim,
                    mlp_ratio=mlp_ratio,
                    pos_embed_layer=pos_embed_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = LayerNorm(predictor_dim)
        self.head = nn.Linear(predictor_dim, out_dim)
        self.apply(_init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def extra_repr(self):
        return f"attn_type={self.attn_type}"

    def forward(
        self,
        coord: Float[Tensor, "B M 3"],
        context: Float[Tensor, "B N D"],
        context_coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B M O"]:
        B, M, _ = coord.shape
        mask_tokens = self.mask_token.expand(B, M, -1)
        if self.pos_embed is not None:
            mask_tokens = self.pos_embed(mask_tokens, coord)

        if self.attn_type == "self_attn":
            x = torch.cat([self.proj(context), mask_tokens], dim=1)
            coord = torch.cat([context_coord, coord], dim=1)
            context = context_coord = None
        else:
            x = mask_tokens

        for block in self.blocks:
            x = block(x, coord=coord, context=context, context_coord=context_coord)

        if self.attn_type == "self_attn":
            x = x[:, -M:]

        x = self.norm(x)
        x = self.head(x)
        return x


# Latent prediction models. Each has a student (encoder, predictor, optional head) and an EMA
# teacher (encoder, optional head). forward runs forward_teacher for the targets, then the
# student, and returns (loss, state). Token weights let the loss weight student-visible and unseen
# positions differently. The encoder outputs class tokens first; the predictor sees them as
# context, the targets are patch tokens only.


class JEPA3D(nn.Module):
    """I-JEPA: smooth l1 between predictions and layer-normed teacher tokens."""

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        encoder_kwargs: dict | None = None,
        predictor_kwargs: dict | None = None,
    ):
        super().__init__()
        self.encoder = ViTEncoder3D(
            grid_size=grid_size,
            target_spacing=target_spacing,
            patch_size=patch_size,
            **(encoder_kwargs or {}),
        )
        embed_dim = self.encoder.embed_dim
        self.predictor = Predictor3D(
            grid_size=grid_size,
            patch_size=patch_size,
            target_spacing=target_spacing,
            embed_dim=embed_dim,
            out_dim=embed_dim,
            pos_embed=self.encoder.pos_embed_type,
            **(predictor_kwargs or {}),
        )
        self.teacher_encoder = copy.deepcopy(self.encoder).requires_grad_(False)

    def train(self, mode: bool = True):
        # the teacher always runs in eval mode, e.g. no drop path (as capi)
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        student_params = list(self.encoder.parameters())
        teacher_params = list(self.teacher_encoder.parameters())
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)

    @torch.no_grad()
    def forward_teacher(
        self, patches: Float[Tensor, "B M P"], coord: Float[Tensor, "B M 3"]
    ) -> tuple[Float[Tensor, "B M D"], dict[str, Tensor]]:
        tokens = self.teacher_encoder(patches, coord)
        tokens = tokens[:, self.teacher_encoder.class_tokens :]
        targets = F.layer_norm(tokens, (tokens.shape[-1],))
        # collapse monitor: feature std across tokens, before the layer norm
        state = {"teacher_std": tokens.float().std(dim=(0, 1)).mean()}
        return targets, state

    def forward(
        self,
        student_patches: Float[Tensor, "B N P"],
        student_coord: Float[Tensor, "B N 3"],
        teacher_patches: Float[Tensor, "B M P"],
        teacher_coord: Float[Tensor, "B M 3"],
        token_weights: Float[Tensor, "B M"],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        targets, state = self.forward_teacher(teacher_patches, teacher_coord)

        tokens = self.encoder(student_patches, student_coord)
        # class tokens sit at the center, as in the encoder
        context_coord = F.pad(student_coord, (0, 0, self.encoder.class_tokens, 0))
        pred = self.predictor(teacher_coord, tokens, context_coord)
        loss = F.smooth_l1_loss(pred.float(), targets.float(), reduction="none").mean(dim=-1)
        loss = (loss * token_weights).sum() / token_weights.sum()

        patch_tokens = tokens[:, self.encoder.class_tokens :].detach().float()
        state["student_std"] = patch_tokens.std(dim=(0, 1)).mean()
        return loss, state


class PrototypeCE3D(nn.Module):
    """iBOT / DINOv2 / JEM: cross entropy from teacher to student prototype distributions.

    teacher_balance: softmax (JEM, no balancing), center (DINO / iBOT), sinkhorn (DINOv2).
    entropy_weight: JEM's R_global on the student distributions, 0 disables.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        encoder_kwargs: dict | None = None,
        predictor_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        student_temp: float = 0.1,
        teacher_temp: float = 0.1,
        teacher_balance: str = "softmax",
        center_momentum: float = 0.9,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        assert teacher_balance in ("softmax", "center", "sinkhorn")
        self.teacher_temp = teacher_temp
        self.teacher_balance = teacher_balance
        self.entropy_weight = entropy_weight

        self.encoder = ViTEncoder3D(
            grid_size=grid_size,
            target_spacing=target_spacing,
            patch_size=patch_size,
            **(encoder_kwargs or {}),
        )
        embed_dim = self.encoder.embed_dim
        # the predictor outputs encoder sized tokens, then the shared head design (iBOT applies
        # its head to encoder outputs)
        self.predictor = Predictor3D(
            grid_size=grid_size,
            patch_size=patch_size,
            target_spacing=target_spacing,
            embed_dim=embed_dim,
            out_dim=embed_dim,
            pos_embed=self.encoder.pos_embed_type,
            **(predictor_kwargs or {}),
        )
        self.head = PrototypeHead(embed_dim, **(head_kwargs or {}))
        self.loss = PrototypeCELoss(self.head.num_prototypes, student_temp, center_momentum)
        self.teacher_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.teacher_head = copy.deepcopy(self.head).requires_grad_(False)

    def train(self, mode: bool = True):
        # the teacher always runs in eval mode, e.g. no drop path (as capi)
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        student_params = list(self.encoder.parameters()) + list(self.head.parameters())
        teacher_params = list(self.teacher_encoder.parameters()) + list(
            self.teacher_head.parameters()
        )
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)

    @torch.no_grad()
    def forward_teacher(
        self, patches: Float[Tensor, "B M P"], coord: Float[Tensor, "B M 3"]
    ) -> tuple[Float[Tensor, "B M K"], dict[str, Tensor]]:
        tokens = self.teacher_encoder(patches, coord)
        tokens = tokens[:, self.teacher_encoder.class_tokens :]
        logits = self.teacher_head(tokens).float()
        if self.teacher_balance == "softmax":
            probs = F.softmax(logits / self.teacher_temp, dim=-1)
        elif self.teacher_balance == "center":
            probs = self.loss.softmax_center_teacher(logits, self.teacher_temp)
            self.loss.update_center(logits)
        else:
            probs = self.loss.sinkhorn_knopp_teacher(logits.flatten(0, 1), self.teacher_temp)
            probs = probs.unflatten(0, logits.shape[:2])

        # collapse monitors: entropy of the mean teacher distribution, and of single tokens
        mean_probs = probs.flatten(0, 1).mean(dim=0)
        state = {
            "teacher_std": tokens.float().std(dim=(0, 1)).mean(),
            "teacher_marginal_entropy": -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(),
            "teacher_token_entropy": -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean(),
        }
        return probs, state

    def forward(
        self,
        student_patches: Float[Tensor, "B N P"],
        student_coord: Float[Tensor, "B N 3"],
        teacher_patches: Float[Tensor, "B M P"],
        teacher_coord: Float[Tensor, "B M 3"],
        token_weights: Float[Tensor, "B M"],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        targets, state = self.forward_teacher(teacher_patches, teacher_coord)

        tokens = self.encoder(student_patches, student_coord)
        # class tokens sit at the center, as in the encoder
        context_coord = F.pad(student_coord, (0, 0, self.encoder.class_tokens, 0))
        pred = self.predictor(teacher_coord, tokens, context_coord)
        logits = self.head(pred)
        loss = self.loss(logits, targets)
        loss = (loss * token_weights).sum() / token_weights.sum()

        patch_tokens = tokens[:, self.encoder.class_tokens :].detach().float()
        state["student_std"] = patch_tokens.std(dim=(0, 1)).mean()
        if self.entropy_weight:
            entropy_loss = entropy_regularizer(logits, self.loss.student_temp)
            loss = loss + self.entropy_weight * entropy_loss
            state["entropy_loss"] = entropy_loss.detach()
        return loss, state


MODELS = {"jepa": JEPA3D, "prototype_ce": PrototypeCE3D}


def build_model(args: dict) -> nn.Module:
    model_class = MODELS[args["model"]]
    return model_class(
        grid_size=tuple(args["grid_size"]),
        target_spacing=args["target_spacing"],
        patch_size=args["patch_size"],
        **args["model_kwargs"],
    )


@register_model
def vit_latent_3d(
    checkpoint: str, encoder: str = "teacher", random_weights: bool = False
) -> ViTEncoder3D:
    """Eval loader: the teacher (EMA) or student encoder of a latent prediction checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    args = ckpt["args"]
    model = ViTEncoder3D(
        grid_size=tuple(args["grid_size"]),
        target_spacing=args["target_spacing"],
        patch_size=args["patch_size"],
        **args["model_kwargs"].get("encoder_kwargs", {}),
    )
    if not random_weights:
        prefix = {"teacher": "teacher_encoder.", "student": "encoder."}[encoder]
        state = {
            k.removeprefix(prefix): v for k, v in ckpt["model"].items() if k.startswith(prefix)
        }
        model.load_state_dict(state)
    return model


def select_tokens(
    images: Float[Tensor, "B X Y Z"],
    patch_ids: Int[Tensor, "B S N"],
    coord_grid: Float[Tensor, "1 L 3"],
    patch_size: int,
) -> tuple[Float[Tensor, "BS N P"], Float[Tensor, "BS N 3"]]:
    """Patches and coords at patch_ids, with the S samples per image flattened into the batch."""
    B = images.shape[0]
    patches = patchify3d(images, patch_size)
    batch_ids = torch.arange(B, device=images.device)[:, None, None]
    patches = patches[batch_ids, patch_ids].flatten(0, 1)
    coord = coord_grid.expand(B, -1, -1)[batch_ids, patch_ids].flatten(0, 1)
    return patches, coord


def patchify3d(x: Tensor, patch_size: int = 8) -> Tensor:
    p = patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=p, py=p, pz=p)
    return x


def make_coord_grid(
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
    spacing: float = 1.0,
    device: torch.device | None = None,
) -> Float[Tensor, "1 N 3"]:
    patch_grid_size = tuple(size // patch_size for size in grid_size)
    grid_ids = [torch.arange(size, device=device) for size in patch_grid_size]
    coords = torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 3)
    center = (torch.tensor(patch_grid_size, device=device) - 1) / 2
    coords = (coords - center) * patch_size * spacing
    return coords


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
