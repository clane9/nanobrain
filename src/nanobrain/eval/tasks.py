import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import datasets as hfds
import nibabel as nib
import numpy as np
import zstandard
from torch.utils.data import Dataset

from nanobrain.eval.probe import (
    probe_binary_classification,
    probe_binary_segmentation,
    probe_regression,
)

EVAL_ROOT = Path(os.getenv("NANOBRAIN_EVAL_ROOT", "data/eval"))
Probe = Callable[..., dict[str, Any]]

SYNTHSEG_CORTEX = (3, 42)
SYNTHSEG_HIPPOCAMPUS = (17, 53)
TASK4_NERVE = 1
TASK4_VESSEL = 2
AP_EXTENT_MM = 133
SEGMENTATION_SUBSET_SIZE = 40
SUBSET_SEED = 0
MAX_NEGATIVE_RATIO = 10.0


class BrainEvalDataset(Dataset):
    """Prepared eval dataset. Decodes the requested images to float32 arrays."""

    def __init__(
        self,
        dataset: hfds.Dataset,
        image_keys: list[str],
        transform: Callable[[dict], dict] | None = None,
    ):
        # drop unused image columns so they are never read
        unused_keys = [
            key
            for key, feature in dataset.features.items()
            if feature.dtype == "large_binary" and key not in image_keys
        ]
        self.dataset = dataset.remove_columns(unused_keys)
        self.image_keys = image_keys
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        for key in self.image_keys:
            img = decode_nifti_zst(sample[key])
            sample[key] = img.get_fdata(dtype=np.float32)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def load_dataset(
    name: str,
    image_keys: list[str],
    transform: Callable[[dict], dict] | None = None,
) -> BrainEvalDataset:
    dataset = hfds.load_from_disk(EVAL_ROOT / name)
    return BrainEvalDataset(dataset, image_keys, transform)


def decode_nifti_zst(data: bytes) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(zstandard.decompress(data))


def select_labels(sample: dict, source_key: str, label_ids: tuple[int, ...]) -> dict:
    sample["target"] = np.isin(sample[source_key], label_ids).astype(np.uint8)
    return sample


def strip_and_crop_ap(sample: dict, image_keys: list[str]) -> dict:
    brain = sample["synthseg"] > 0
    for key in image_keys:
        sample[key] = sample[key] * brain
    brain_ap = np.flatnonzero(brain.any(axis=(0, 2)))
    start = round((brain_ap[0] + brain_ap[-1] - AP_EXTENT_MM) / 2)
    window = np.zeros(brain.shape[1], dtype=bool)
    window[max(start, 0) : start + AP_EXTENT_MM] = True
    for key in image_keys:
        sample[key] = sample[key] * window[None, :, None]
    return sample


def fomo_task1_dwi_cls() -> Probe:
    dataset = load_dataset("fomo_task1", ["dwi_b1000", "mask"])
    return partial(
        probe_binary_classification, dataset=dataset, image_key="dwi_b1000", label_key="label"
    )


def fomo_task1_lesion(image_key: str) -> Probe:
    dataset = hfds.load_from_disk(EVAL_ROOT / "fomo_task1")
    positive_ids = [ii for ii, label in enumerate(dataset["label"]) if label == 1]
    dataset = BrainEvalDataset(dataset.select(positive_ids), [image_key, "mask", "seg"])
    return partial(
        probe_binary_segmentation,
        dataset=dataset,
        image_key=image_key,
        label_key="seg",
        max_negative_ratio=MAX_NEGATIVE_RATIO,
    )


def fomo_task1_adc_lesion() -> Probe:
    return fomo_task1_lesion("adc")


def fomo_task1_dwi_lesion() -> Probe:
    return fomo_task1_lesion("dwi_b1000")


def fomo_task2_tumor(image_key: str) -> Probe:
    dataset = load_dataset("fomo_task2", [image_key, "mask", "seg"])
    return partial(
        probe_binary_segmentation,
        dataset=dataset,
        image_key=image_key,
        label_key="seg",
        max_negative_ratio=MAX_NEGATIVE_RATIO,
    )


def fomo_task2_dwi_tumor() -> Probe:
    return fomo_task2_tumor("dwi_b1000")


def fomo_task2_flair_tumor() -> Probe:
    return fomo_task2_tumor("flair")


def fomo_task3_t1w_age() -> Probe:
    dataset = load_dataset("fomo_task3", ["t1w", "mask"])
    return partial(probe_regression, dataset=dataset, image_key="t1w", target_key="age")


def fomo_task3_synthseg(label_ids: tuple[int, ...], max_negative_ratio: float | None) -> Probe:
    dataset = hfds.load_from_disk(EVAL_ROOT / "fomo_task3")
    rng = np.random.default_rng(SUBSET_SEED)
    subset_ids = sorted(rng.choice(len(dataset), SEGMENTATION_SUBSET_SIZE, replace=False))
    transform = partial(select_labels, source_key="synthseg", label_ids=label_ids)
    dataset = BrainEvalDataset(dataset.select(subset_ids), ["t1w", "mask", "synthseg"], transform)
    return partial(
        probe_binary_segmentation,
        dataset=dataset,
        image_key="t1w",
        label_key="target",
        max_negative_ratio=max_negative_ratio,
    )


def fomo_task3_t1w_cortex() -> Probe:
    return fomo_task3_synthseg(SYNTHSEG_CORTEX, None)


def fomo_task3_t1w_hippocampus() -> Probe:
    return fomo_task3_synthseg(SYNTHSEG_HIPPOCAMPUS, MAX_NEGATIVE_RATIO)


def fomo_task4_structure(label_id: int) -> Probe:
    transform = partial(select_labels, source_key="seg", label_ids=(label_id,))
    dataset = load_dataset("fomo_task4", ["t2w", "mask", "seg"], transform)
    return partial(
        probe_binary_segmentation,
        dataset=dataset,
        image_key="t2w",
        label_key="target",
        max_negative_ratio=MAX_NEGATIVE_RATIO,
    )


def fomo_task4_t2w_nerve() -> Probe:
    return fomo_task4_structure(TASK4_NERVE)


def fomo_task4_t2w_vessel() -> Probe:
    return fomo_task4_structure(TASK4_VESSEL)


def fomo_task5_t1w_cls() -> Probe:
    transform = partial(strip_and_crop_ap, image_keys=["t1w", "mask"])
    dataset = load_dataset("fomo_task5", ["t1w", "mask", "synthseg"], transform)
    return partial(probe_binary_classification, dataset=dataset, image_key="t1w", label_key="label")


def fomo_task5_t1w_cortex() -> Probe:
    transform = partial(select_labels, source_key="synthseg", label_ids=SYNTHSEG_CORTEX)
    dataset = load_dataset("fomo_task5", ["t1w", "mask", "synthseg"], transform)
    return partial(probe_binary_segmentation, dataset=dataset, image_key="t1w", label_key="target")


TASKS: dict[str, Callable[[], Probe]] = {
    task.__name__: task
    for task in [
        fomo_task1_dwi_cls,
        fomo_task1_adc_lesion,
        fomo_task1_dwi_lesion,
        fomo_task2_dwi_tumor,
        fomo_task2_flair_tumor,
        fomo_task3_t1w_age,
        fomo_task3_t1w_cortex,
        fomo_task3_t1w_hippocampus,
        fomo_task4_t2w_nerve,
        fomo_task4_t2w_vessel,
        fomo_task5_t1w_cls,
        fomo_task5_t1w_cortex,
    ]
}

HEADLINE_METRICS: dict[str, str] = {
    "fomo_task1_dwi_cls": "auroc",
    "fomo_task1_adc_lesion": "voxel_auroc",
    "fomo_task1_dwi_lesion": "voxel_auroc",
    "fomo_task2_dwi_tumor": "voxel_auroc",
    "fomo_task2_flair_tumor": "voxel_auroc",
    "fomo_task3_t1w_age": "r2",
    "fomo_task3_t1w_cortex": "dice",
    "fomo_task3_t1w_hippocampus": "dice",
    "fomo_task4_t2w_nerve": "dice",
    "fomo_task4_t2w_vessel": "dice",
    "fomo_task5_t1w_cls": "auroc",
    "fomo_task5_t1w_cortex": "dice",
}
