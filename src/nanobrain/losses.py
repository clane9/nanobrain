# References:
# dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
#   and dinov2/loss/ibot_patch_loss.py
# jem: arXiv 2610.10013, eqs 5-6

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.nn.init import trunc_normal_


class PrototypeHead(nn.Module):
    """dinov2 DINOHead: mlp to a bottleneck, l2 norm, then l2-normed prototypes.

    dinov2 uses weight_norm with weight_g filled with 1 (DINO also freezes it); here the
    prototype rows are normalized explicitly, the same as weight_g fixed at 1.
    """

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int = 4096,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(_init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, num_prototypes, bias=False)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... K"]:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        weight = F.normalize(self.last_layer.weight, dim=-1)
        return F.linear(x, weight)


class PrototypeCELoss(nn.Module):
    """dinov2 iBOTPatchLoss on one gpu: teacher centering or sinkhorn, and per-token cross entropy.

    The center update is applied right away, so the teacher probs of a batch use the center
    from the batches before it (as dinov2, which applies the update on the next call).
    """

    def __init__(
        self, num_prototypes: int, student_temp: float = 0.1, center_momentum: float = 0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, num_prototypes))

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_logits: Float[Tensor, "B M K"], teacher_temp: float
    ) -> Float[Tensor, "B M K"]:
        return F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_logits: Float[Tensor, "B M K"]):
        batch_center = teacher_logits.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_logits: Float[Tensor, "N K"], teacher_temp: float, num_iterations: int = 3
    ) -> Float[Tensor, "N K"]:
        teacher_logits = teacher_logits.float()
        Q = torch.exp((teacher_logits - teacher_logits.max()) / teacher_temp)
        N, K = Q.shape
        Q /= Q.sum()
        for _ in range(num_iterations):
            # each prototype gets total weight 1 / K, then each token 1 / N
            Q /= Q.sum(dim=0, keepdim=True) * K
            Q /= Q.sum(dim=1, keepdim=True) * N
        # each token's assignment sums to 1
        return Q * N

    def forward(
        self, student_logits: Float[Tensor, "B M K"], teacher_probs: Float[Tensor, "B M K"]
    ) -> Float[Tensor, "B M"]:
        log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        return -torch.sum(teacher_probs * log_probs, dim=-1)


def entropy_regularizer(logits: Float[Tensor, "B M K"], temp: float) -> Tensor:
    """JEM R_global: minus (entropy of the mean distribution - mean entropy), over the batch.

    JEM takes the mean per patch position. Our views sample random positions, so here the mean
    is over all tokens in the batch.
    """
    probs = F.softmax(logits.float() / temp, dim=-1).flatten(0, 1)
    mean_probs = probs.mean(dim=0)
    marginal_entropy = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum()
    token_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return -(marginal_entropy - token_entropy)


def _init_weights(m: nn.Module) -> None:
    # dinov2 DINOHead init
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
