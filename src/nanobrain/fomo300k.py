import zipfile
import gzip
import importlib.resources
from pathlib import Path

import fsspec
import nibabel as nib
import numpy as np
from torch.utils.data import IterableDataset
from huggingface_hub.utils import disable_progress_bars

from nanobrain.utils.prefetch import prefetch

disable_progress_bars()

DEFAULT_ROOT = "s3://nanobrain/FOMO300K"
DEFAULT_SUFFIXES = ("T1w", "T2w", "FLAIR", "bval0", "bval1000")
DEFAULT_FILELIST = importlib.resources.files("nanobrain.config").joinpath(
    "fomo300k_filelist_openneuro.txt"
)


class Fomo300K(IterableDataset):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        filelist: str = DEFAULT_FILELIST,
        suffixes: list[str] | None = DEFAULT_SUFFIXES,
        shuffle: bool = False,
        random_state: int | np.random.Generator | None = None,
        cache_dir: str | Path | None = None,
        prefetch_threads: int = 1,
        storage_options: dict | None = None,
    ):
        self.root = root
        self.filelist = filelist
        self.suffixes = suffixes
        self.shuffle = shuffle
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.prefetch_threads = prefetch_threads
        self.storage_options = storage_options

        with fsspec.open(filelist, "rt") as f:
            paths = f.read().strip().splitlines()
        self.paths_ = np.array(paths)
        self.rng_ = np.random.default_rng(random_state)

    def __iter__(self):
        paths = self.paths_.copy()
        if self.shuffle:
            self.rng_.shuffle(paths)

        for path, fullpath in prefetch(
            self.root,
            paths,
            cache_dir=self.cache_dir,
            max_workers=self.prefetch_threads,
            storage_options=self.storage_options,
        ):
            for name, img in read_fomo300_zip(fullpath, suffixes=self.suffixes):
                fullname = f"{path.removesuffix('.zip')}/{name}"
                yield fullname, img


def read_fomo300_zip(path: str, suffixes: list[str] | None = DEFAULT_SUFFIXES):
    if suffixes is not None:
        suffixes = set(suffixes)

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".nii.gz"):
                continue
            stem = name.removesuffix(".nii.gz")
            suf = stem.split("_")[-1]
            if suffixes is not None and suf not in suffixes:
                continue

            with z.open(name) as zf:
                with gzip.open(zf) as gz:
                    img = nib.Nifti1Image.from_bytes(gz.read())

            yield name, img


def list_fomo300_files(
    root: str = DEFAULT_ROOT,
    pattern: str | list[str] = "**/*.zip",
    storage_options: dict | None = None,
):
    fs, root_ = fsspec.url_to_fs(str(root), **(storage_options or {}))
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths = sorted(
        p.removeprefix(root_).lstrip("/") for pat in patterns for p in fs.glob(f"{root_}/{pat}")
    )
    return paths
