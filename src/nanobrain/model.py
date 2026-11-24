import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainmarks_smri.models import EmbeddingOutput, ImageInput, register_model
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from jaxtyping import Bool, Float, Int

from .modules import (
    Block,
    LayerNorm,
    Rope3D,
    SeparablePosEmbed3D,
    SinCosPosEmbed3D,
)
from .data import (
    crop_to_mask,
    fit_to_shape,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)


class ViTMAE3D(nn.Module):
    name = "vit_mae_3d"

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 1,
        drop_path_rate: float = 0.0,
        clip_quantiles: tuple[float, float] = (0.005, 0.995),
        min_mask_frac: float = 0.25,
        pos_embed: str = "learned_sep",
        num_partner_visible: int = 0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.target_spacing = target_spacing
        # tuple for the brainmarks-smri model contract
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.class_tokens = class_tokens
        patch_grid_size = tuple(size // patch_size for size in grid_size)
        self.clip_quantiles = clip_quantiles
        self.min_mask_frac = min_mask_frac
        self.pos_embed_type = pos_embed
        self.num_partner_visible = num_partner_visible

        patch_dim = patch_size**3
        # patch center coords in mm, relative to the volume center
        coord_grid = make_coord_grid(grid_size, patch_size, target_spacing)
        self.register_buffer("coord_grid", coord_grid, persistent=False)
        unit = patch_size * target_spacing

        # encoder
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
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

        # stochastic depth decay rule
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

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        # visible partner patches enter the decoder through their own patch embedding
        if num_partner_visible:
            self.partner_patch_embed = nn.Linear(patch_dim, decoder_embed_dim)
        else:
            self.partner_patch_embed = None
        if pos_embed == "learned_sep":
            self.decoder_pos_embed = SeparablePosEmbed3D(
                patch_grid_size, decoder_embed_dim, unit=unit
            )
        elif pos_embed == "sincos":
            self.decoder_pos_embed = SinCosPosEmbed3D(patch_grid_size, decoder_embed_dim, unit=unit)
        else:
            self.decoder_pos_embed = None
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    pos_embed_layer=pos_embed_layer,
                )
                for ii in range(decoder_depth)
            ]
        )

        self.decoder_norm = LayerNorm(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_dim)
        self.init_weights()

    @classmethod
    def from_config(cls, args: dict) -> "ViTMAE3D":
        return cls(
            grid_size=tuple(args["grid_size"]),
            target_spacing=args["target_spacing"],
            clip_quantiles=tuple(args["clip_quantiles"]),
            patch_size=args["patch_size"],
            **args["model_kwargs"],
        )

    def extra_repr(self):
        return (
            f"{self.grid_size}, {self.patch_size}, class_tokens={self.class_tokens}, "
            f"pos_embed={self.pos_embed_type}, num_partner_visible={self.num_partner_visible}"
        )

    def init_weights(self):
        self.apply(_init_weights)
        if self.class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # initialize the decoder head to zero to prevent hockey stick
        self.decoder_head.weight.data.mul_(0.0)

    def forward_encoder(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        B, N, P = patches.shape
        assert coord.shape == (B, N, 3)

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

    def forward_decoder(
        self,
        embeds: Float[Tensor, "B R+N D"],
        coord: Float[Tensor, "B N 3"],
        target_coord: Float[Tensor, "B M 3"],
        partner_patches: Float[Tensor, "B K P"],
        partner_coord: Float[Tensor, "B K 3"],
    ) -> Float[Tensor, "B M P"]:
        B = embeds.shape[0]
        M = target_coord.shape[1]
        assert target_coord.shape == (B, M, 3)

        embeds = self.decoder_proj(embeds)
        mask_tokens = self.mask_token.expand(B, M, -1)

        if self.decoder_pos_embed is not None:
            mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)

        # no partner tokens (K = 0) without a partner patch embed
        partner_tokens = mask_tokens[:, :0]
        if self.partner_patch_embed is not None:
            partner_tokens = self.partner_patch_embed(partner_patches)
            if self.decoder_pos_embed is not None:
                partner_tokens = self.decoder_pos_embed(partner_tokens, partner_coord)
        x = torch.cat([embeds, partner_tokens, mask_tokens], dim=1)
        coord = torch.cat([coord, partner_coord, target_coord], dim=1)
        if self.class_tokens:
            coord = F.pad(coord, (0, 0, self.class_tokens, 0))

        for block in self.decoder_blocks:
            x = block(x, coord=coord)

        x = x[:, -M:]
        x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x

    def forward(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        partner_images: Float[Tensor, "B X Y Z"],
        partner_mask: Float[Tensor, "B X Y Z"],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
        with_patches: bool = False,
        scores: Float[Tensor, "B S N"] | None = None,
        is_cross: Bool[Tensor, " B"] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # encode masked A, predict partner B (B = A for plain MAE) at positions masked in A
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size[0])
        mask_patches = patchify3d(mask, self.patch_size[0])
        partner_patches = patchify3d(partner_images, self.patch_size[0])
        partner_mask_patches = patchify3d(partner_mask, self.patch_size[0])
        coord = self.coord_grid.expand(B, -1, -1)

        # sample num_samples sequences per image, then flatten them into the batch
        visible_ids, predict_ids, partner_visible_ids = sample_sequences(
            mask_patches,
            partner_mask_patches,
            num_visible,
            num_predict,
            self.num_partner_visible,
            num_samples,
            scores,
            min_mask_frac=self.min_mask_frac,
        )
        batch_ids = torch.arange(B, device=images.device)[:, None, None]
        vis_patches = patches[batch_ids, visible_ids].flatten(0, 1)
        vis_coord = coord[batch_ids, visible_ids].flatten(0, 1)
        target_patches = partner_patches[batch_ids, predict_ids].flatten(0, 1)
        target_coord = coord[batch_ids, predict_ids].flatten(0, 1)
        target_mask_patches = mask_patches[batch_ids, predict_ids].flatten(0, 1)
        partner_target_mask_patches = partner_mask_patches[batch_ids, predict_ids].flatten(0, 1)
        partner_vis_patches = partner_patches[batch_ids, partner_visible_ids].flatten(0, 1)
        partner_vis_coord = coord[batch_ids, partner_visible_ids].flatten(0, 1)

        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(
            embeds, vis_coord, target_coord, partner_vis_patches, partner_vis_coord
        )
        # loss inside both brain masks
        loss_weight = target_mask_patches * partner_target_mask_patches
        error = F.mse_loss(pred_patches, target_patches, reduction="none")
        loss = (loss_weight * error).sum() / loss_weight.sum().clamp_min(1)

        state = {}
        if is_cross is not None:
            # loss split over cross-view (B != A) and self (B = A) sequences
            seq_is_cross = is_cross.repeat_interleave(num_samples)[:, None, None]
            error = error.detach()
            cross_weight = loss_weight * seq_is_cross
            self_weight = loss_weight * ~seq_is_cross
            state["cross_frac"] = is_cross.float().mean()
            state["loss_cross"] = (cross_weight * error).sum() / cross_weight.sum().clamp_min(1)
            state["loss_self"] = (self_weight * error).sum() / self_weight.sum().clamp_min(1)

        if with_patches:
            state.update(
                {
                    "vis_patches": vis_patches,
                    "target_patches": target_patches,
                    "loss_weight": loss_weight,
                    "partner_vis_patches": partner_vis_patches,
                    "embeds": embeds,
                    "pred_patches": pred_patches,
                    "vis_ids": visible_ids.flatten(0, 1),
                    "target_ids": predict_ids.flatten(0, 1),
                    "partner_vis_ids": partner_visible_ids.flatten(0, 1),
                    "image_ids": torch.arange(B, device=images.device).repeat_interleave(
                        num_samples
                    ),
                }
            )
        return loss, state

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
        values = normalize_values(values, self.clip_quantiles)
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
                tokens = self.forward_encoder(patches[:, in_mask], self.coord_grid[:, in_mask])
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


@register_model
def vit_mae_3d(checkpoint: str, random_weights: bool = False, **kwargs) -> ViTMAE3D:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    args = ckpt["args"]
    # override the checkpoint's model_kwargs
    args["model_kwargs"].update(kwargs)
    model = ViTMAE3D.from_config(args)
    if not random_weights:
        model.load_state_dict(ckpt["model"])
    return model


def patchify3d(x: Tensor, patch_size: int = 8) -> Tensor:
    p = patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=p, py=p, pz=p)
    return x


def unpatchify3d(x: Tensor, grid_size: tuple[int, int, int], patch_size: int = 8) -> Tensor:
    p = patch_size
    gx, gy, gz = [size // patch_size for size in grid_size]
    x = rearrange(
        x,
        "b (gx gy gz) (px py pz) -> b (gx px) (gy py) (gz pz)",
        gx=gx,
        gy=gy,
        gz=gz,
        px=p,
        py=p,
        pz=p,
    )
    return x


def patches_to_volume(
    patches: Float[Tensor, "B L P"],
    patch_ids: Int[Tensor, "B L"],
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
) -> Float[Tensor, "B X Y Z"]:
    B, L, P = patches.shape
    gx, gy, gz = [size // patch_size for size in grid_size]
    grid = patches.new_zeros(B, gx * gy * gz, P)
    batch_ids = torch.arange(B, device=patches.device)[:, None]
    grid[batch_ids, patch_ids] = patches
    images = unpatchify3d(grid, grid_size=grid_size, patch_size=patch_size)
    return images


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


def sample_sequences(
    mask_patches: Tensor,
    partner_mask_patches: Tensor,
    num_visible: int,
    num_predict: int | None,
    num_partner_visible: int = 0,
    num_samples: int = 1,
    scores: Tensor | None = None,
    min_mask_frac: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, num_patches, _ = mask_patches.shape
    device = mask_patches.device
    mask_frac = mask_patches.mean(dim=2)
    both_frac = (mask_patches * partner_mask_patches).mean(dim=2)
    partner_frac = partner_mask_patches.mean(dim=2)
    in_mask = mask_frac[:, None, :] >= min_mask_frac
    in_both = both_frac[:, None, :] >= min_mask_frac
    in_partner = partner_frac[:, None, :] >= min_mask_frac
    if num_predict is None:
        # one host sync to get the longest mask sequence in the batch
        num_predict = (mask_frac >= min_mask_frac).sum(dim=1).max().item() - num_visible

    # lowest scoring mask patches are visible, uniform random without scores
    if scores is None:
        scores = torch.rand(batch_size, num_samples, num_patches, device=device)
    visible_keys = torch.where(in_mask, scores, torch.inf)
    visible_ids = visible_keys.topk(num_visible, dim=2, largest=False).indices

    # predict uniformly from the rest, patches inside both masks first
    predict_keys = torch.rand(batch_size, num_samples, num_patches, device=device)
    predict_keys = torch.where(in_both, predict_keys, 2.0 + predict_keys)
    predict_keys = predict_keys.scatter(2, visible_ids, torch.inf)
    predict_ids = predict_keys.topk(num_predict, dim=2, largest=False).indices

    # partner visible uniformly from the partner mask, outside the predicted
    partner_keys = torch.rand(batch_size, num_samples, num_patches, device=device)
    partner_keys = torch.where(in_partner, partner_keys, 2.0 + partner_keys)
    partner_keys = partner_keys.scatter(2, predict_ids, torch.inf)
    partner_visible_ids = partner_keys.topk(num_partner_visible, dim=2, largest=False).indices
    return visible_ids, predict_ids, partner_visible_ids


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
