from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GRID_SHAPE = (192, 240, 192)
TARGET_SPACING = 1.0


class BrainNpzDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().strip().splitlines()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        with np.load(self.root / path) as npz:
            affine = npz["affine"]
            sample = {
                "path": path,
                "values": torch.from_numpy(npz["values"]),
                "mask": torch.from_numpy(npz["mask"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": torch.from_numpy(affine),
                "spacing": tuple(np.linalg.norm(affine[:3, :3], axis=0).tolist()),
            }
        return sample


def process_sample(sample: dict[str, torch.Tensor | Any]):
    values = sample["values"]
    mask_rle = sample["mask"]
    shape = sample["shape"]
    spacing = sample["spacing"]

    # normalize values
    values = values.float()
    values = values / values.max()
    values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)

    # decode mask
    mask = brle_to_dense(mask_rle, shape=shape)

    # make dense image
    image = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
    image.masked_scatter_(mask, values)

    # resizing
    image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
    image = fit_to_shape(image, target_shape=GRID_SHAPE)

    mask = mask.float()
    mask = resample_image(mask, spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest")
    mask = fit_to_shape(mask, target_shape=GRID_SHAPE)

    # apply mask
    image = image * mask

    sample = {"image": image, "mask": mask}
    return sample


def brle_to_dense(mask_rle: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    X, Y, Z = shape
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=X * Y * Z)
    return mask.reshape(shape)


def resample_image(
    image: torch.Tensor,
    spacing: tuple[float, float, float],
    target_spacing: float = 1.0,
    mode: str = "trilinear",
) -> torch.Tensor:
    new_shape = [round(size * sp / target_spacing) for size, sp in zip(image.shape, spacing)]
    if list(image.shape) == new_shape:
        return image
    image = F.interpolate(image[None, None], size=new_shape, mode=mode)
    return image[0, 0]


def fit_to_shape(
    image: torch.Tensor, target_shape: tuple[int, int, int] = GRID_SHAPE
) -> torch.Tensor:
    # x/y centered and z top aligned, matching the crop in conform_image
    # F.pad takes pads last dim first, negative pads crop
    pad = []
    for axis in reversed(range(3)):
        diff = target_shape[axis] - image.shape[axis]
        if axis < 2:
            before = diff // 2
        else:
            before = diff
        pad += [before, diff - before]
    return F.pad(image, pad)
