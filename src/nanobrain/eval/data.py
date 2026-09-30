import os
from pathlib import Path

import datasets as hfds
import nibabel as nib
import numpy as np
import zstandard
from torch.utils.data import Dataset

EVAL_ROOT = Path(os.getenv("NANOBRAIN_EVAL_ROOT", "data/eval"))


class BrainEvalDataset(Dataset):
    """Prepared eval dataset. Decodes the requested images to float32 arrays."""

    def __init__(self, dataset: hfds.Dataset, image_keys: list[str]):
        # drop unused image columns so they are never read
        unused_keys = [
            key
            for key, feature in dataset.features.items()
            if feature.dtype == "large_binary" and key not in image_keys
        ]
        self.dataset = dataset.remove_columns(unused_keys)
        self.image_keys = image_keys

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        for key in self.image_keys:
            img = decode_nifti_zst(sample[key])
            sample[key] = img.get_fdata(dtype=np.float32)
        return sample


def load_dataset(name: str, image_keys: list[str]) -> BrainEvalDataset:
    dataset = hfds.load_from_disk(EVAL_ROOT / name)
    return BrainEvalDataset(dataset, image_keys)


def decode_nifti_zst(data: bytes) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(zstandard.decompress(data))
