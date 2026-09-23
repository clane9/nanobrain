from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GRID_SHAPE = (192, 240, 192)


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


def collate_lists(samples: list[dict]) -> dict[str, list]:
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def decode_image(
    values: torch.Tensor, mask_rle: torch.Tensor, shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    numel = shape[0] * shape[1] * shape[2]

    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=numel)

    # masked_scatter avoids the host sync of boolean index assignment
    values = values.float() / (2**16 - 1)
    image = torch.zeros(numel, dtype=torch.float32, device=values.device)
    image.masked_scatter_(mask, values)
    return image.reshape(shape), mask.reshape(shape)


def resample_image(
    image: torch.Tensor, spacing: tuple[float, float, float], target_spacing: float = 1.0
) -> torch.Tensor:
    new_shape = [round(size * sp / target_spacing) for size, sp in zip(image.shape, spacing)]
    if list(image.shape) == new_shape:
        return image
    image = F.interpolate(image[None, None], size=new_shape, mode="trilinear", align_corners=False)
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
