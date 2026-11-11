import torch
from jaxtyping import Float
from torch import Tensor


def mask_unit_scores(
    masks: Float[Tensor, "B X Y Z"], num_samples: int, patch_size: int, mask_unit_size: int
) -> Float[Tensor, "B S N"]:
    # one random score per unit of mask_unit_size^3 patches, so whole units are visible or not
    batch_size = masks.shape[0]
    patch_grid_size = [size // patch_size for size in masks.shape[1:]]
    assert all(size % mask_unit_size == 0 for size in patch_grid_size), (
        f"patch grid {patch_grid_size} not divisible by mask unit {mask_unit_size}"
    )
    unit_grid_size = [size // mask_unit_size for size in patch_grid_size]
    scores = torch.rand(batch_size, num_samples, *unit_grid_size, device=masks.device)
    for dim in [2, 3, 4]:
        scores = scores.repeat_interleave(mask_unit_size, dim=dim)
    return scores.flatten(2)
