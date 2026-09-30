import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path

import datasets as hfds
import fsspec
import nibabel as nib
import numpy as np
import zstandard
from nibabel.processing import resample_from_to
from SynthSeg_pytorch import SynthSegPredictor
from torch.utils.data import Dataset
from tqdm import tqdm

import nanobrain.utils.preprocessing as preproc

logger = logging.getLogger(__name__)

EVAL_ROOT = Path(os.getenv("NANOBRAIN_EVAL_ROOT", "data/eval"))
FOMO_RAW_ROOT = Path(os.getenv("FOMO_RAW_ROOT", "data/eval_raw/fomo"))
FOMO_URL = "https://sid.erda.dk/share_redirect/fmeuvo1EdF"
GRID_FOV = (192.0, 240.0, 192.0)
VMIN_QUANTILE = 0.005
VMAX_QUANTILE = 0.995


class EvalDataset(Dataset):
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


def load_dataset(name: str, image_keys: list[str]) -> EvalDataset:
    dataset = hfds.load_from_disk(EVAL_ROOT / name)
    return EvalDataset(dataset, image_keys)


def prepare_fomo_task1(num_workers: int, device: str) -> hfds.Dataset:
    task_dir = fetch_fomo_task(1)
    subjects = sorted(path.name for path in (task_dir / "preprocessed").iterdir())
    prepare_subject = partial(prepare_fomo_task1_subject, task_dir, device=device)
    with ProcessPoolExecutor(num_workers) as executor:
        rows = list(tqdm(executor.map(prepare_subject, subjects), total=len(subjects)))

    features = hfds.Features(
        {
            "subject": hfds.Value("string"),
            "label": hfds.Value("int32"),
            "mask": hfds.Value("large_binary"),
            "adc": hfds.Value("large_binary"),
            "dwi_b1000": hfds.Value("large_binary"),
            "flair": hfds.Value("large_binary"),
            "synthseg": hfds.Value("large_binary"),
            "seg": hfds.Value("large_binary"),
        }
    )
    return hfds.Dataset.from_list(rows, features=features)


def prepare_fomo_task1_subject(task_dir: Path, subject: str, device: str) -> dict:
    label = int((task_dir / "labels" / subject / "ses-01" / "label.txt").read_text())
    images = {
        modality: nib.load(task_dir / "preprocessed" / subject / "ses-01" / f"{modality}.nii.gz")
        for modality in ("adc", "dwi_b1000", "flair")
    }
    seg_path = task_dir / "labels" / subject / "ses-01" / "seg.nii.gz"
    assert seg_path.exists() == (label == 1), f"{subject}: seg does not match label"
    segs = {"seg": nib.load(seg_path) if label == 1 else None}
    encoded = prepare_images(images, segs, reference="flair", device=device)
    return {"subject": subject, "label": label, **encoded}


def prepare_fomo_task2(num_workers: int, device: str) -> hfds.Dataset:
    task_dir = fetch_fomo_task(2)
    subjects = sorted(path.name for path in (task_dir / "preprocessed").iterdir())
    prepare_subject = partial(prepare_fomo_task2_subject, task_dir, device=device)
    with ProcessPoolExecutor(num_workers) as executor:
        rows = list(tqdm(executor.map(prepare_subject, subjects), total=len(subjects)))

    features = hfds.Features(
        {
            "subject": hfds.Value("string"),
            "mask": hfds.Value("large_binary"),
            "dwi_b1000": hfds.Value("large_binary"),
            "flair": hfds.Value("large_binary"),
            "synthseg": hfds.Value("large_binary"),
            "seg": hfds.Value("large_binary"),
        }
    )
    return hfds.Dataset.from_list(rows, features=features)


def prepare_fomo_task2_subject(task_dir: Path, subject: str, device: str) -> dict:
    images = {
        modality: nib.load(task_dir / "preprocessed" / subject / "ses-01" / f"{modality}.nii.gz")
        for modality in ("dwi_b1000", "flair")
    }
    segs = {"seg": nib.load(task_dir / "labels" / subject / "ses-01" / "seg.nii.gz")}
    encoded = prepare_images(images, segs, reference="flair", device=device)
    return {"subject": subject, **encoded}


def prepare_fomo_task3(num_workers: int, device: str) -> hfds.Dataset:
    task_dir = fetch_fomo_task(3)
    subjects = sorted(path.name for path in (task_dir / "preprocessed").iterdir())
    prepare_subject = partial(prepare_fomo_task3_subject, task_dir, device=device)
    with ProcessPoolExecutor(num_workers) as executor:
        rows = list(tqdm(executor.map(prepare_subject, subjects), total=len(subjects)))

    features = hfds.Features(
        {
            "subject": hfds.Value("string"),
            "age": hfds.Value("float32"),
            "mask": hfds.Value("large_binary"),
            "t1w": hfds.Value("large_binary"),
            "synthseg": hfds.Value("large_binary"),
        }
    )
    return hfds.Dataset.from_list(rows, features=features)


def prepare_fomo_task3_subject(task_dir: Path, subject: str, device: str) -> dict:
    age = float((task_dir / "labels" / subject / "ses-01" / "labels.txt").read_text())
    images = {"t1w": nib.load(task_dir / "preprocessed" / subject / "ses-01" / "t1w.nii.gz")}
    encoded = prepare_images(images, {}, reference="t1w", device=device)
    return {"subject": subject, "age": age, **encoded}


def prepare_fomo_task4(num_workers: int, device: str) -> hfds.Dataset:
    task_dir = fetch_fomo_task(4)
    subjects = sorted(path.name for path in (task_dir / "preprocessed").iterdir())
    prepare_subject = partial(prepare_fomo_task4_subject, task_dir, device=device)
    with ProcessPoolExecutor(num_workers) as executor:
        rows = list(tqdm(executor.map(prepare_subject, subjects), total=len(subjects)))

    features = hfds.Features(
        {
            "subject": hfds.Value("string"),
            "mask": hfds.Value("large_binary"),
            "t2w": hfds.Value("large_binary"),
            "synthseg": hfds.Value("large_binary"),
            "seg": hfds.Value("large_binary"),
        }
    )
    return hfds.Dataset.from_list(rows, features=features)


def prepare_fomo_task4_subject(task_dir: Path, subject: str, device: str) -> dict:
    images = {"t2w": nib.load(task_dir / "preprocessed" / subject / "ses-01" / "t2w.nii.gz")}
    segs = {"seg": nib.load(task_dir / "labels" / subject / "ses-01" / "seg.nii.gz")}
    encoded = prepare_images(images, segs, reference="t2w", device=device)
    return {"subject": subject, **encoded}


def prepare_fomo_task5(num_workers: int, device: str) -> hfds.Dataset:
    task_dir = fetch_fomo_task(5)
    subjects = sorted(path.name for path in (task_dir / "preprocessed").iterdir())
    prepare_subject = partial(prepare_fomo_task5_subject, task_dir, device=device)
    with ProcessPoolExecutor(num_workers) as executor:
        rows = list(tqdm(executor.map(prepare_subject, subjects), total=len(subjects)))

    features = hfds.Features(
        {
            "subject": hfds.Value("string"),
            "label": hfds.Value("int32"),
            "mask": hfds.Value("large_binary"),
            "t1w": hfds.Value("large_binary"),
            "synthseg": hfds.Value("large_binary"),
        }
    )
    return hfds.Dataset.from_list(rows, features=features)


def prepare_fomo_task5_subject(task_dir: Path, subject: str, device: str) -> dict:
    label = int((task_dir / "labels" / subject / "ses_01" / "labels.txt").read_text())
    images = {"t1w": nib.load(task_dir / "preprocessed" / subject / "ses_01" / "t1.nii.gz")}
    encoded = prepare_images(images, {}, reference="t1w", device=device)
    return {"subject": subject, "label": label, **encoded}


PREPARE_FUNCTIONS = {
    "fomo_task1": prepare_fomo_task1,
    "fomo_task2": prepare_fomo_task2,
    "fomo_task3": prepare_fomo_task3,
    "fomo_task4": prepare_fomo_task4,
    "fomo_task5": prepare_fomo_task5,
}


def fetch_fomo_task(task: int) -> Path:
    """Extracted FOMO task directory, downloaded and extracted if missing."""
    task_dir = FOMO_RAW_ROOT / f"Task_{task}"
    if task_dir.exists():
        return task_dir

    FOMO_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if task == 5:
        # task 5 ships as the source dataset plus a script that builds Task_5/ from it
        for filename in ("Task_5_extract.py", "Zhang_Lingfeng_2022_PPMR_Dataset.zip"):
            download_file(f"{FOMO_URL}/{filename}", FOMO_RAW_ROOT / filename)
        subprocess.run([sys.executable, "Task_5_extract.py"], cwd=FOMO_RAW_ROOT, check=True)
    else:
        zip_path = FOMO_RAW_ROOT / f"Task_{task}.zip"
        download_file(f"{FOMO_URL}/Task_{task}.zip", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(FOMO_RAW_ROOT)
    assert task_dir.exists(), f"{task_dir} missing after extraction"
    return task_dir


def download_file(url: str, path: Path):
    if path.exists():
        return
    logger.info(f"downloading {url} to {path}")
    tmp_path = path.with_name(path.name + ".tmp")
    with fsspec.open(url) as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.rename(path)


def prepare_images(
    images: dict[str, nib.Nifti1Image],
    segs: dict[str, nib.Nifti1Image | None],
    reference: str,
    device: str = "cuda",
) -> dict[str, bytes | None]:
    """
    Conform all images to the fixed 1mm grid using one head mask from the reference image.
    Images are clipped and scaled to uint16 inside the mask like prepare.py, zero outside.
    Also runs SynthSeg on the conformed reference. Missing segs stay None.
    """
    mask_img = preproc.threshold_mask(images[reference])
    reference_img = preproc.conform_image(
        images[reference], mask_img, max_fov=GRID_FOV, fixed_grid=True
    )
    affine = reference_img.affine
    fit_mask = np.asarray(resample_from_to(mask_img, reference_img, order=0).dataobj) > 0
    fit_mask_img = nib.Nifti1Image(fit_mask.astype(np.uint8), affine)

    encoded = {"mask": encode_nifti_zst(fit_mask_img)}
    for key, img in images.items():
        # modalities share the reference voxel grid, so the same mask gives the same fit grid.
        # conform rather than resample_from_to so every modality gets the same anti-alias smoothing
        fit_img = preproc.conform_image(img, mask_img, max_fov=GRID_FOV, fixed_grid=True)
        assert np.allclose(fit_img.affine, affine, atol=1e-3), f"{key} grid mismatch"
        fit_img = truncate_image(fit_img, fit_mask_img, (VMIN_QUANTILE, VMAX_QUANTILE))
        encoded[key] = encode_nifti_zst(fit_img)
        if key == reference:
            encoded["synthseg"] = encode_nifti_zst(run_synthseg(fit_img, device=device))

    for key, img in segs.items():
        if img is None:
            encoded[key] = None
            continue
        fit_seg = resample_from_to(img, reference_img, order=0)
        seg = np.asarray(fit_seg.dataobj).round().astype(np.uint8)
        encoded[key] = encode_nifti_zst(nib.Nifti1Image(seg, affine))
    return encoded


@lru_cache(maxsize=1)
def synthseg_predictor(device: str) -> SynthSegPredictor:
    return SynthSegPredictor(device=device)


def run_synthseg(image: nib.Nifti1Image, device: str) -> nib.Nifti1Image:
    predictor = synthseg_predictor(device=device)
    seg, _, _, seg_affine, _ = predictor.segment(image)
    seg_image = nib.Nifti1Image(seg.astype(np.uint8), seg_affine)
    seg_image = resample_from_to(seg_image, image, order=0)
    seg_image.set_data_dtype(np.uint8)
    return seg_image


def truncate_image(
    img: nib.Nifti1Image, mask_img: nib.Nifti1Image, qs: tuple[float, float]
) -> nib.Nifti1Image:
    data = img.get_fdata(dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0
    values = data[mask]
    vmin, vmax = np.quantile(values, qs)
    assert vmax > vmin, "degenerate value range"
    values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    values = (values * (2**16 - 1)).astype(np.uint16)
    data = np.zeros(data.shape, dtype=np.uint16)
    data[mask] = values
    img = nib.Nifti1Image(data, img.affine)
    img.set_data_dtype(np.uint16)
    return img


def encode_nifti_zst(img: nib.Nifti1Image) -> bytes:
    return zstandard.compress(img.to_bytes())


def decode_nifti_zst(data: bytes) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(zstandard.decompress(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=list(PREPARE_FUNCTIONS))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    out_path = EVAL_ROOT / args.name
    assert not out_path.exists(), f"{out_path} exists, delete it first"
    dataset = PREPARE_FUNCTIONS[args.name](args.num_workers, args.device)
    dataset.save_to_disk(out_path)
    logger.info(f"saved {len(dataset)} samples to {out_path}")
