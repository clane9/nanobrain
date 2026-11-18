"""View samplers: pick the student and teacher token subsets of each volume.

A sampler module is called with the per-patch brain mask fraction [B, N] and returns patch ids
(student [B, S, num_student], teacher [B, S, num_teacher]) for S samples per volume. Fixed
sizes keep the batch rectangular.
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class RandomViews(nn.Module):
    """Uniform random brain patches. The teacher shares num_overlap of the student's patches."""

    def __init__(
        self,
        num_student: int,
        num_teacher: int,
        num_overlap: int,
        min_mask_frac: float = 0.25,
    ):
        super().__init__()
        assert 0 <= num_overlap <= min(num_student, num_teacher)
        self.num_student = num_student
        self.num_teacher = num_teacher
        self.num_overlap = num_overlap
        self.min_mask_frac = min_mask_frac

    def forward(
        self, mask_frac: Float[Tensor, "B N"], num_samples: int
    ) -> tuple[Int[Tensor, "B S Ns"], Int[Tensor, "B S Nt"]]:
        batch_size, num_patches = mask_frac.shape
        keys = torch.rand(batch_size, num_samples, num_patches, device=mask_frac.device)
        in_mask = mask_frac[:, None, :] >= self.min_mask_frac
        keys = torch.where(in_mask, keys, torch.inf)

        # one random order per sample: student first, the teacher starts num_overlap before
        # the student ends
        num_total = self.num_student + self.num_teacher - self.num_overlap
        order = keys.topk(num_total, dim=2, largest=False).indices
        student_ids = order[:, :, : self.num_student]
        teacher_ids = order[:, :, self.num_student - self.num_overlap :]
        return student_ids, teacher_ids

    def extra_repr(self):
        return (
            f"num_student={self.num_student}, num_teacher={self.num_teacher}, "
            f"num_overlap={self.num_overlap}, min_mask_frac={self.min_mask_frac}"
        )


class MaskUnitViews(nn.Module):
    """As RandomViews, but in units of unit_size^3 patches (pixio, hiera mask units)."""

    def __init__(
        self,
        num_student: int,
        num_teacher: int,
        num_overlap: int,
        patch_grid_size: tuple[int, int, int],
        unit_size: int = 2,
        min_mask_frac: float = 0.25,
    ):
        super().__init__()
        assert 0 <= num_overlap <= min(num_student, num_teacher)
        assert all(size % unit_size == 0 for size in patch_grid_size), (
            f"patch grid {patch_grid_size} not divisible by mask unit {unit_size}"
        )
        self.num_student = num_student
        self.num_teacher = num_teacher
        self.num_overlap = num_overlap
        self.patch_grid_size = patch_grid_size
        self.unit_size = unit_size
        self.min_mask_frac = min_mask_frac

    def forward(
        self, mask_frac: Float[Tensor, "B N"], num_samples: int
    ) -> tuple[Int[Tensor, "B S Ns"], Int[Tensor, "B S Nt"]]:
        batch_size, _ = mask_frac.shape
        # one random key per unit, small jitter so units stay contiguous in the order
        unit_grid_size = [size // self.unit_size for size in self.patch_grid_size]
        keys = torch.rand(batch_size, num_samples, *unit_grid_size, device=mask_frac.device)
        for dim in [2, 3, 4]:
            keys = keys.repeat_interleave(self.unit_size, dim=dim)
        keys = keys.flatten(2)
        keys = keys + 1e-6 * torch.rand_like(keys)
        in_mask = mask_frac[:, None, :] >= self.min_mask_frac
        keys = torch.where(in_mask, keys, torch.inf)

        num_total = self.num_student + self.num_teacher - self.num_overlap
        order = keys.topk(num_total, dim=2, largest=False).indices
        student_ids = order[:, :, : self.num_student]
        teacher_ids = order[:, :, self.num_student - self.num_overlap :]
        return student_ids, teacher_ids

    def extra_repr(self):
        return (
            f"num_student={self.num_student}, num_teacher={self.num_teacher}, "
            f"num_overlap={self.num_overlap}, unit_size={self.unit_size}, "
            f"min_mask_frac={self.min_mask_frac}"
        )
