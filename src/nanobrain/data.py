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
    def __init__(self, root: str | Path, filelist: str | Path, with_age: bool = False):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().split()
        self.ages = None
        if with_age:
            images = pd.read_parquet(
                self.root / "tables" / "images.parquet", columns=["path", "age"]
            )
            # images.parquet paths are source/<stem>.nii.gz, filelist paths derivatives/<stem>.npz
            stems = images["path"].str.removeprefix("source/").str.removesuffix(".nii.gz")
            images.index = "derivatives/" + stems + ".npz"
            self.ages = images["age"].reindex(self.paths).to_numpy(dtype=np.float32)
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
        if self.ages is not None:
            # nan when unknown
            sample["age"] = float(self.ages[index])
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


def synth_lesion(
    image: Tensor,
    mask: Tensor,
    num_lesions: tuple[int, int] = (1, 1),
    radius: tuple[float, float] = (5.0, 25.0),
    roughness: float = 0.5,
    contrast: tuple[float, float] = (0.8, 2.5),
    ring_prob: float = 0.0,
    texture: float = 0.3,
    edge: float = 0.75,
    keep_anatomy: tuple[float, float] = (0.0, 0.5),
) -> tuple[Tensor, Tensor]:
    # lesions blended into a normalized 1 mm image [X, Y, Z]: a noisy ellipsoid shape, intensity of
    # the local tissue mean +- contrast (z units), optional rim, smooth texture. label = soft > 0.5
    device = image.device
    image = image.clone()
    label = torch.zeros_like(image)
    count = int(torch.randint(num_lesions[0], num_lesions[1] + 1, ()))
    # centers in 4 mm cells whose neighbors are all brain, so ~4-8 mm deep
    cells = F.avg_pool3d(mask[None, None].float(), kernel_size=4) > 0.99
    cells = -F.max_pool3d(-cells.float(), kernel_size=3, stride=1, padding=1)[0, 0]
    centers = torch.multinomial(cells.flatten() + 1e-6, count)
    centers = torch.stack(torch.unravel_index(centers, cells.shape), dim=1)
    centers = 4 * centers + torch.randint(0, 4, centers.shape, device=device)
    # one host sync for all lesion boxes
    centers = centers.tolist()
    for center in centers:
        r = math.exp(math.log(radius[0]) + torch.rand(()).item() * math.log(radius[1] / radius[0]))
        half = math.ceil(1.6 * r) + 2
        box = tuple(slice(max(c - half, 0), min(c + half, n)) for c, n in zip(center, image.shape))
        grids = [torch.arange(b.start, b.stop, device=device) - c for b, c in zip(box, center)]
        rel = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).float()
        axes = r * torch.exp(torch.rand(3, device=device) * 0.6 - 0.3)
        size = rel.shape[:3]
        shape_noise = F.interpolate(
            torch.randn(1, 1, 5, 5, 5, device=device), size=size, mode="trilinear"
        )[0, 0]
        dist = (rel / axes).norm(dim=-1) - 1 + roughness * shape_noise
        soft = torch.sigmoid(-dist * r / edge) * mask[box]

        sign = 1.0 if torch.rand(()) < 0.5 else -1.0
        amount = sign * (contrast[0] + torch.rand(()).item() * (contrast[1] - contrast[0]))
        value = torch.full(size, amount, device=device)
        if torch.rand(()) < ring_prob:
            rim = 0.2 + 0.3 * torch.rand(()).item()
            core = torch.sigmoid(-(dist + rim) * r / edge)
            value = value - core * (amount + 0.5 * amount)
        tex_size = [max(2, n // 4) for n in size]
        tex_noise = F.interpolate(
            torch.randn(1, 1, *tex_size, device=device), size=size, mode="trilinear"
        )[0, 0]
        local_mean = (image[box] * mask[box]).sum() / mask[box].sum().clamp_min(1)
        target = local_mean + value + texture * tex_noise
        # keep some of the underlying anatomy and noise, so smoothness alone doesn't give it away
        keep = keep_anatomy[0] + torch.rand(()).item() * (keep_anatomy[1] - keep_anatomy[0])
        target = target + keep * (image[box] - local_mean)
        image[box] = (1 - soft) * image[box] + soft * target
        label[box] = torch.maximum(label[box], (soft > 0.5).float())
    return image, label
