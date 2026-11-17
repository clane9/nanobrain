# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py
# vjepa2: https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/pos_embs.py

from functools import partial
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from timm.layers import DropPath

Layer = Type[nn.Module]


# Transformer modules adapted from capi (but removed the efficient residual)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        causal: bool = False,
        pos_embed_layer: Layer | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        # position embedding applied to q and k (e.g. rope)
        self.pos_embed = pos_embed_layer(self.head_dim) if pos_embed_layer else None
        context_dim = context_dim or dim

        # using separate q, k, v weights so that xavier init uses the correct dim.
        # although perhaps technically it should be initialized wrt the head dim..
        # but this is what original mae does.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def extra_repr(self):
        return f"num_heads={self.num_heads}, causal={self.causal}"

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
        coord: Float[Tensor, "B N 3"] | None = None,
        context_coord: Float[Tensor, "B M 3"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        if context is None:
            context = x
            context_coord = coord
        B, N, D = x.shape
        _, M, _ = context.shape
        h = self.num_heads

        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)
        k = self.k(context).reshape(B, M, h, D // h).transpose(1, 2)
        v = self.v(context).reshape(B, M, h, D // h).transpose(1, 2)
        if self.pos_embed is not None:
            q = self.pos_embed(q, coord[:, None])
            k = self.pos_embed(k, context_coord[:, None])
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int | float = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# timm default eps=1e-6
LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        mlp_ratio: int | float = 4,
        drop_path: float = 0.0,
        norm_layer: Layer = LayerNorm,
        pos_embed_layer: Layer | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            context_dim=context_dim,
            pos_embed_layer=pos_embed_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=proj_bias,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
        coord: Float[Tensor, "B N 3"] | None = None,
        context_coord: Float[Tensor, "B M 3"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        # should the context also be normalized? capi doesn't, so I guess not
        attn_out = self.attn(
            self.norm1(x), context=context, coord=coord, context_coord=context_coord
        )
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# Position embedding
#
# All position embeddings take coord in mm, relative to the center of the volume. Table embeddings
# look up the nearest grid cell, with grid spacing `unit` mm (the patch size).


class SeparablePosEmbed3D(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        embed_dim: int,
        unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.unit = unit

        X, Y, Z = grid_size
        self.weight_x = nn.Parameter(torch.empty(X, embed_dim))
        self.weight_y = nn.Parameter(torch.empty(Y, embed_dim))
        self.weight_z = nn.Parameter(torch.empty(Z, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_x, std=0.02)
        nn.init.trunc_normal_(self.weight_y, std=0.02)
        nn.init.trunc_normal_(self.weight_z, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B N D"]:
        X, Y, Z = self.grid_size
        x_ids = torch.round(coord[:, :, 0] / self.unit + (X - 1) / 2).to(torch.int64)
        y_ids = torch.round(coord[:, :, 1] / self.unit + (Y - 1) / 2).to(torch.int64)
        z_ids = torch.round(coord[:, :, 2] / self.unit + (Z - 1) / 2).to(torch.int64)
        x = x + self.weight_x[x_ids]
        x = x + self.weight_y[y_ids]
        x = x + self.weight_z[z_ids]
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}, unit={self.unit}"


class SinCosPosEmbed3D(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        embed_dim: int,
        unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.unit = unit

        X, Y, Z = grid_size
        weight = get_3d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=(Y, Z),
            grid_depth=X,
            uniform_power=True,
        )
        self.register_buffer("weight", torch.from_numpy(weight).float(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B N D"]:
        X, Y, Z = self.grid_size
        x_ids = torch.round(coord[:, :, 0] / self.unit + (X - 1) / 2).to(torch.int64)
        y_ids = torch.round(coord[:, :, 1] / self.unit + (Y - 1) / 2).to(torch.int64)
        z_ids = torch.round(coord[:, :, 2] / self.unit + (Z - 1) / 2).to(torch.int64)
        pos_ids = (x_ids * Y + y_ids) * Z + z_ids
        x = x + self.weight[pos_ids]
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}, unit={self.unit}"


# sincos pos embed utils from vjepa2, but fixed the confusing meshgrid indexing


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: tuple of int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_height*grid_width, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_height*grid_width, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_d, grid_h, grid_w = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Rope adapted from capi. Fixed the channel layout so that each rotated pair of channels
# (i, i + dim/2) shares one angle (capi pairs an x angle with a y angle).


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope3D(nn.Module):
    def __init__(
        self,
        dim: int,
        max_freq: float = 1 / 16,
        min_freq: float = 1 / 160_000,
    ):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq

        freqs = get_3d_rope_freqs(dim, max_freq, min_freq)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self, x: Float[Tensor, "*b d"], coord: Float[Tensor, "*b 3"]
    ) -> Float[Tensor, "*b d"]:
        # broadcast multiply, not matmul, so the angle stays fp32 under autocast
        angle = (coord.float()[..., :, None] * self.freqs).sum(dim=-2)
        angle = torch.cat([angle, angle], dim=-1)
        out = x * angle.cos() + rotate_half(x) * angle.sin()
        return out.to(x.dtype)

    def extra_repr(self):
        return f"{self.dim}, max_freq={self.max_freq}, min_freq={self.min_freq}"


def get_3d_rope_freqs(dim: int, max_freq: float, min_freq: float) -> Float[Tensor, "3 D/2"]:
    """
    Axial rope angular freqs, one row per axis, for the first half of the channels.
    dim // 6 freqs per axis, geometric from max_freq to min_freq (cycles per unit).
    Leftover pairs (e.g. 2 of 32 at dim 64) are not rotated.
    """
    num_freqs = dim // 6
    freqs_1d = max_freq * (max_freq / min_freq) ** torch.linspace(0, -1, num_freqs)
    freqs = torch.zeros(3, dim // 2)
    freqs[0, 0:num_freqs] = freqs_1d
    freqs[1, num_freqs : 2 * num_freqs] = freqs_1d
    freqs[2, 2 * num_freqs : 3 * num_freqs] = freqs_1d
    freqs = freqs * 2 * torch.pi
    return freqs
