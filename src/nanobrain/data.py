import io
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import Dataset, Sampler


class Fomo300kDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        filelist: str | Path,
        partners: str | Path | None = None,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().split()
        self.seed = seed

        # partner paths per image, empty for images without a partner
        groups = {}
        if partners is not None:
            groups = pd.read_parquet(partners).groupby("path")["partner"].agg(list)
        self.partners = [groups.get(path, []) for path in self.paths]

        all_paths = set(self.paths).union(*self.partners)
        members = pd.read_parquet(
            self.root / "tables" / "shards.parquet", columns=["path", "shard", "offset", "size"]
        )
        members = members[members["path"].isin(all_paths)]
        assert len(members) == len(all_paths), "missing shard members"
        # path -> (shard, offset, size)
        self.members = {row[0]: row[1:] for row in members.itertuples(index=False)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, key: tuple[int, int]) -> dict:
        index, draw = key
        sample = self.read(self.paths[index])
        partners = self.partners[index]
        if partners:
            rng = np.random.default_rng((self.seed, draw))
            sample["partner"] = self.read(partners[rng.integers(len(partners))])
        return sample

    def read(self, path: str) -> dict:
        # npz members are stored uncompressed, so read them straight from their offset
        shard_path, offset, size = self.members[path]
        with open(self.root / shard_path, "rb") as shard:
            shard.seek(offset)
            data = shard.read(size)
        with np.load(io.BytesIO(data)) as npz:
            sample = {
                "path": path,
                "values": torch.from_numpy(npz["values"]),
                "scale": float(npz["scale"]),
                "mask_rle": torch.from_numpy(npz["mask_rle"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": npz["affine"],
                "mni_affine": npz["mni_affine"],
            }
        return sample


class InfiniteSampler(Sampler[tuple[int, int]]):
    """Endless shuffled passes over the dataset from draw `start`, yielding (index, draw)."""

    def __init__(self, num_samples: int, seed: int = 0, start: int = 0):
        self.num_samples = num_samples
        self.seed = seed
        self.start = start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        index = 0
        while True:
            for sample_index in torch.randperm(self.num_samples, generator=generator).tolist():
                if index >= self.start:
                    yield sample_index, index
                index += 1


def brle_to_dense(mask_rle: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    X, Y, Z = shape
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=X * Y * Z)
    return mask.reshape(shape)


def crop_to_mask(
    image: Float[Tensor, "X Y Z"], mask: Bool[Tensor, "X Y Z"], affine: np.ndarray
) -> tuple[Tensor, Tensor, np.ndarray]:
    starts = []
    stops = []
    for axis in range(3):
        other_axes = tuple(other for other in range(3) if other != axis)
        ids = mask.any(dim=other_axes).nonzero()[:, 0]
        starts.append(ids.min().item())
        stops.append(ids.max().item() + 1)
    crop = tuple(slice(start, stop) for start, stop in zip(starts, stops))

    shift = np.eye(4)
    shift[:3, 3] = starts
    return image[crop], mask[crop], affine @ shift


def reorient_to_ras(
    image: Float[Tensor, "X Y Z"], mask: Bool[Tensor, "X Y Z"], affine: np.ndarray
) -> tuple[Tensor, Tensor, np.ndarray]:
    # same as nibabel apply_orientation: flip input axes, then permute
    ornt = nib.orientations.io_orientation(affine)
    flip_axes = [axis for axis in range(3) if ornt[axis, 1] == -1]
    order = np.argsort(ornt[:, 0]).tolist()
    new_affine = affine @ nib.orientations.inv_ornt_aff(ornt, image.shape)
    image = image.flip(flip_axes).permute(order)
    mask = mask.flip(flip_axes).permute(order)
    return image, mask, new_affine


def resample_to_spacing(
    image: Float[Tensor, "X Y Z"],
    mask: Bool[Tensor, "X Y Z"],
    affine: np.ndarray,
    target_spacing: float = 1.0,
) -> tuple[Tensor, Float[Tensor, "X Y Z"], np.ndarray]:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    shape = np.array(image.shape)
    new_shape = np.maximum(np.round(shape * spacing / target_spacing), 1).astype(int)

    # anti-aliasing for downsampled axes, sigma rule from skimage.transform.resize
    sigmas = np.maximum(0, (target_spacing / spacing - 1) / 2)
    image = gaussian_smooth(image, sigmas.tolist())

    new_size = new_shape.tolist()
    image = F.interpolate(image[None, None], size=new_size, mode="trilinear")[0, 0]
    mask = F.interpolate(mask[None, None].float(), size=new_size, mode="nearest-exact")[0, 0]
    image = image * mask

    # interpolate maps output voxel i to input voxel (i + 0.5) * scale - 0.5
    scale = shape / new_shape
    resample = np.eye(4)
    resample[:3, :3] = np.diag(scale)
    resample[:3, 3] = 0.5 * scale - 0.5
    return image, mask, affine @ resample


def gaussian_smooth(image: Float[Tensor, "X Y Z"], sigmas: list[float]) -> Float[Tensor, "X Y Z"]:
    # separable gaussian, sigmas in voxels per axis, zero padded
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = math.ceil(3 * sigma)
        offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        lines = image.movedim(axis, -1)
        lines_shape = lines.shape
        lines = F.conv1d(lines.reshape(-1, 1, lines_shape[-1]), kernel[None, None], padding=radius)
        image = lines.reshape(lines_shape).movedim(-1, axis)
    return image


def fit_to_shape(
    image: Float[Tensor, "X Y Z"],
    mask: Float[Tensor, "X Y Z"],
    affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[Tensor, Tensor, np.ndarray]:
    # center pad or crop. F.pad takes pads last dim first, negative pads crop
    befores = []
    pad = []
    for axis in reversed(range(3)):
        diff = target_shape[axis] - image.shape[axis]
        before = diff // 2
        befores.insert(0, before)
        pad += [before, diff - before]

    shift = np.eye(4)
    shift[:3, 3] = -np.array(befores)
    return F.pad(image, pad), F.pad(mask, pad), affine @ shift


def resample_affine(
    image: Float[Tensor, "X Y Z"],
    mask: Float[Tensor, "X Y Z"],
    voxel_map: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    # trilinear resample onto a new grid. voxel_map takes new grid voxels to input voxels
    axes = [torch.arange(size, device=image.device, dtype=torch.float32) for size in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    voxel_map = torch.as_tensor(voxel_map, device=image.device, dtype=torch.float32)
    coords = coords @ voxel_map[:3, :3].T + voxel_map[:3, 3]
    # grid_sample wants (x, y, z) = (last, middle, first) axis, normalized to [-1, 1]
    sizes = torch.tensor(image.shape, device=image.device, dtype=torch.float32)
    coords = 2 * coords / (sizes - 1).clamp_min(1) - 1
    grid = coords.flip(-1)[None]

    channels = torch.stack([image, mask.float()])
    channels = F.grid_sample(channels[None], grid, mode="bilinear", align_corners=True)[0]
    mask = (channels[1] > 0.5).float()
    return channels[0] * mask, mask


def normalize_values(
    values: Float[Tensor, " N"], clip_quantiles: tuple[float, float] = (0.005, 0.995)
) -> Float[Tensor, " N"]:
    # nearest rank clip thresholds. torch.quantile needs a q tensor, and copying it to the gpu syncs
    # cf https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/
    # nb, another option would a torch module with fixed q tensor buffer. but can't be bothered.
    sorted_values = values.sort().values
    last = len(values) - 1
    low = sorted_values[round(clip_quantiles[0] * last)]
    high = sorted_values[round(clip_quantiles[1] * last)]
    values = values.clamp(low, high)
    values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
    return values
