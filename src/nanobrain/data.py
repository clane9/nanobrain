import io
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2


class Fomo300kDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().split()
        members = pd.read_parquet(
            self.root / "tables" / "shards.parquet", columns=["path", "shard", "offset", "size"]
        )
        members = members.set_index("path").loc[self.paths]
        self.shards = members["shard"].tolist()
        self.offsets = members["offset"].to_numpy()
        self.sizes = members["size"].to_numpy()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        # npz members are stored uncompressed, so read them straight from their offset
        with open(self.root / self.shards[index], "rb") as shard:
            shard.seek(self.offsets[index])
            data = shard.read(self.sizes[index])
        with np.load(io.BytesIO(data)) as npz:
            sample = {
                "path": self.paths[index],
                "values": torch.from_numpy(npz["values"]),
                "scale": float(npz["scale"]),
                "mask_rle": torch.from_numpy(npz["mask_rle"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": npz["affine"],
            }
        return sample


class InfiniteSampler(Sampler[int]):
    """Endless shuffled passes over the dataset, starting from sample index `start`."""

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
                    yield sample_index
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


# Photometric augmentation of z-scored images on the training grid, as torch modules (composed
# with torchvision's Compose and RandomApply) over the functions below. Each module takes and
# returns a sample dict with "image" and "mask". Outside the mask stays zero. Random parameters are
# drawn on the cpu, so no device syncs. torchvision's intensity transforms
# assume [0, 1] images and don't keep the mask, so these are our own.


def adjust_gamma(
    image: Float[Tensor, "X Y Z"], mask: Tensor, gamma: float
) -> Float[Tensor, "X Y Z"]:
    # gamma on the intensities scaled to [0, 1] over the brain, then back to the original range
    mask = mask > 0
    low = torch.where(mask, image, torch.inf).amin()
    high = torch.where(mask, image, -torch.inf).amax()
    scaled = ((image - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    image = scaled**gamma * (high - low) + low
    return image * mask


def random_log_bias_field(
    shape: tuple[int, int, int], strength: float, num_freqs: int = 4, device=None
) -> Float[Tensor, "X Y Z"]:
    # smooth random field from a low frequency DCT basis, std = strength
    coefficients = torch.randn(num_freqs, num_freqs, num_freqs).to(device)
    coefficients[0, 0, 0] = 0
    bases = []
    for size in shape:
        position = (torch.arange(size, device=device) + 0.5) / size
        freqs = torch.arange(num_freqs, device=device)
        bases.append(torch.cos(math.pi * freqs[:, None] * position))
    log_field = torch.einsum("ijk,ix,jy,kz->xyz", coefficients, *bases)
    return log_field / log_field.std() * strength


def apply_bias_field(
    image: Float[Tensor, "X Y Z"], mask: Tensor, log_field: Float[Tensor, "X Y Z"]
) -> Float[Tensor, "X Y Z"]:
    # multiplicative field on the intensities shifted to start at zero over the brain
    mask = mask > 0
    low = torch.where(mask, image, torch.inf).amin()
    image = (image - low) * log_field.exp() + low
    return image * mask


def scale_shift(
    image: Float[Tensor, "X Y Z"], mask: Tensor, scale: float, shift: float
) -> Float[Tensor, "X Y Z"]:
    return (image * scale + shift) * (mask > 0)


def add_noise(image: Float[Tensor, "X Y Z"], mask: Tensor, std: float) -> Float[Tensor, "X Y Z"]:
    return (image + std * torch.randn_like(image)) * (mask > 0)


class RandomGamma(nn.Module):
    def __init__(self, max_log_gamma: float = 0.3):
        super().__init__()
        self.max_log_gamma = max_log_gamma

    def forward(self, sample: dict) -> dict:
        log_gamma = torch.empty(()).uniform_(-self.max_log_gamma, self.max_log_gamma).item()
        image = adjust_gamma(sample["image"], sample["mask"], math.exp(log_gamma))
        return {**sample, "image": image}

    def extra_repr(self):
        return f"max_log_gamma={self.max_log_gamma}"


class RandomBiasField(nn.Module):
    def __init__(self, strength: float = 0.2, num_freqs: int = 4):
        super().__init__()
        self.strength = strength
        self.num_freqs = num_freqs

    def forward(self, sample: dict) -> dict:
        image = sample["image"]
        log_field = random_log_bias_field(
            tuple(image.shape), self.strength, self.num_freqs, device=image.device
        )
        image = apply_bias_field(image, sample["mask"], log_field)
        return {**sample, "image": image}

    def extra_repr(self):
        return f"strength={self.strength}, num_freqs={self.num_freqs}"


class RandomScaleShift(nn.Module):
    def __init__(self, max_log_scale: float = 0.2, max_shift: float = 0.2):
        super().__init__()
        self.max_log_scale = max_log_scale
        self.max_shift = max_shift

    def forward(self, sample: dict) -> dict:
        log_scale = torch.empty(()).uniform_(-self.max_log_scale, self.max_log_scale).item()
        shift = torch.empty(()).uniform_(-self.max_shift, self.max_shift).item()
        image = scale_shift(sample["image"], sample["mask"], math.exp(log_scale), shift)
        return {**sample, "image": image}

    def extra_repr(self):
        return f"max_log_scale={self.max_log_scale}, max_shift={self.max_shift}"


class RandomNoise(nn.Module):
    def __init__(self, max_std: float = 0.1):
        super().__init__()
        self.max_std = max_std

    def forward(self, sample: dict) -> dict:
        std = torch.empty(()).uniform_(0, self.max_std).item()
        image = add_noise(sample["image"], sample["mask"], std)
        return {**sample, "image": image}

    def extra_repr(self):
        return f"max_std={self.max_std}"


AUGMENTATIONS = {
    "gamma": RandomGamma,
    "bias_field": RandomBiasField,
    "scale_shift": RandomScaleShift,
    "noise": RandomNoise,
}


def make_augment(config: list[dict] | None) -> nn.Module:
    """Compose from a list of {name, p, **kwargs}, e.g. [{name: gamma, p: 0.5, max_log_gamma: 0.3}]."""
    transforms = []
    for entry in config or []:
        kwargs = dict(entry)
        name = kwargs.pop("name")
        p = kwargs.pop("p", 1.0)
        transform = AUGMENTATIONS[name](**kwargs)
        if p < 1.0:
            transform = v2.RandomApply([transform], p=p)
        transforms.append(transform)
    # torchvision Compose needs at least one transform
    return v2.Compose(transforms) if transforms else nn.Identity()
