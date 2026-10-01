import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from nanobrain.eval.probe import (
    GRID_SHAPE,
    patchify3d,
    probe_binary_classification,
    probe_binary_segmentation,
    probe_regression,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="probes need cuda")


class DummyDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.labels = rng.permutation(np.arange(n_samples) % 2)
        self.random_labels = rng.permutation(np.arange(n_samples) % 2)
        self.ages = rng.uniform(20, 80, n_samples)
        self.blob_centers = rng.uniform(-40, 40, (n_samples, 3)) + np.array(GRID_SHAPE) / 2

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, index])
        x, y, z = np.meshgrid(*[np.arange(size) for size in GRID_SHAPE], indexing="ij")
        center = np.array(GRID_SHAPE) / 2
        head = ((x - center[0]) / 70) ** 2 + ((y - center[1]) / 90) ** 2 + (
            (z - center[2]) / 70
        ) ** 2 < 1

        bx, by, bz = self.blob_centers[index]
        blob = (x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2 < 10**2

        image = rng.normal(100 + self.ages[index], 10, GRID_SHAPE).astype(np.float32)
        image[blob] += 100
        if self.labels[index]:
            image[blob] += 100
        image *= head

        return {
            "t1w": image,
            "mask": head.astype(np.uint8),
            "label": self.labels[index],
            "random_label": self.random_labels[index],
            "age": self.ages[index],
            "lesion": blob.astype(np.uint8),
        }


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        quantiles = torch.tensor([0.5, 0.9, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0])
        self.register_buffer("quantiles", quantiles)

    def global_embed(self, images: Tensor, mask: Tensor) -> Tensor:
        embeddings = []
        for image, image_mask in zip(images, mask):
            values = image[image_mask > 0].float()
            embeddings.append(torch.quantile(values, self.quantiles))
        return self.scale * torch.stack(embeddings)

    def dense_embed(self, images: Tensor, mask: Tensor) -> Tensor:
        patches = patchify3d(images)
        grid = [size // 8 for size in GRID_SHAPE]
        return self.scale * patches.reshape(len(images), *grid, 512)


def test_probe_binary_classification():
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyDataset(40), image_key="t1w", label_key="label"
    )
    assert record["auroc"] > 0.95
    assert record["n_samples"] == 40
    assert record["embed_dim"] == 9
    assert len(record["probabilities"]) == 40


def test_probe_binary_classification_random_labels():
    record, _ = probe_binary_classification(
        DummyEncoder().cuda(), DummyDataset(40), image_key="t1w", label_key="random_label"
    )
    low, high = record["auroc_ci"]
    assert low < 0.5 < high


def test_probe_regression():
    record, _ = probe_regression(
        DummyEncoder().cuda(), DummyDataset(40), image_key="t1w", target_key="age"
    )
    assert record["r"] > 0.9
    assert record["mae"] < 10


def test_probe_binary_segmentation():
    record, _ = probe_binary_segmentation(
        DummyEncoder().cuda(), DummyDataset(20), image_key="t1w", label_key="lesion"
    )
    assert record["dice"] > 0.9
    assert record["average_precision"] > 0.9
    assert record["voxel_auroc"] > 0.9
    assert record["embed_dim"] == 512
    assert len(record["subject_dice"]) == 20
