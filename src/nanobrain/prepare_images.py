import argparse
import io
import json
import logging
import time
import secrets
import zipfile
import gzip
import importlib.resources
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from nibabel.processing import resample_from_to
from PIL import Image

import nanobrain.utils.misc as misc
import nanobrain.utils.preprocessing as preproc
from nanobrain.utils.prefetch import prefetch
from nanobrain.utils.rle import dense_to_brle

logger = logging.getLogger()

DEFAULT_ROOT = "s3://nanobrain/FOMO300K"
DEFAULT_SUFFIXES = ("T1w", "T2w", "FLAIR", "bval0", "bval1000")
DEFAULT_FILELIST = importlib.resources.files("nanobrain.config").joinpath(
    "fomo300k_filelist_openneuro_subsampled.txt"
)

QUANTILES = (0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)
VMIN_QUANTILE = 0.005
VMAX_QUANTILE = 0.995

TOTAL_BATCHES = 64
SHUFFLE_SEED = 0
LOG_INTERVAL = 100
PREFETCH_THREADS = 1
JPEG_QUALITY = 85


def main(args: argparse.Namespace):
    assert 0 <= args.index < TOTAL_BATCHES, f"invalid {args.index=}"

    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    misc.setup_logging(logger, log_dir / f"log-{args.index:03d}.txt")

    logger.info(f"preparing batch {args.index:03d}/{TOTAL_BATCHES}")
    logger.info(f"args:\n{json.dumps(vars(args), default=str)}")
    logger.info(f"sha: {misc.git_sha()}")

    paths = Path(args.filelist).read_text().strip().splitlines()

    # shuffle so batches get a balanced mix of datasets and suffixes, otherwise the
    # slowest batch runs ~2x the mean
    rng = np.random.default_rng(SHUFFLE_SEED)
    paths = [paths[i] for i in rng.permutation(len(paths))]

    offsets = np.linspace(0, len(paths), TOTAL_BATCHES + 1, dtype=np.int64).tolist()
    start, stop = offsets[args.index : args.index + 2]
    paths = paths[start:stop]

    # limit concurrency to prevent fsspec timeout errors
    # the script is going to be parallelized anyway
    dataset = iter_fomo300k(
        root=args.root,
        filelist=paths,
        prefetch_threads=PREFETCH_THREADS,
        storage_options={"max_concurrency": 1},
    )

    logger.info("example paths:\n\t" + "\n\t".join(paths[:5]))

    num_images = 0
    num_errors = 0
    t0 = time.time()

    for name, img in dataset:
        try:
            process_image(
                name,
                img,
                out_root=args.out_root,
                min_voxel_size=args.min_voxel_size,
                max_fov=args.max_fov,
            )
        except Exception as exc:
            logger.warning(f"Error {name}; {exc!r}", exc_info=True)
            num_errors += 1
        num_images += 1

        if num_images % LOG_INTERVAL == 0:
            sps = num_images / (time.time() - t0)
            logger.info(
                f"done [{num_images:05d}] {name}; tput: {sps:.1f} im/s, errors: {num_errors}"
            )

    elapsed = time.time() - t0
    sps = num_images / elapsed
    logger.info(
        f"finished batch {args.index:03d}; "
        f"images: {num_images}, errors: {num_errors}, "
        f"tput: {sps:.1f} im/s, elapsed: {elapsed / 3600:.2f} hr"
    )


def process_image(
    name: str,
    img: nib.Nifti1Image,
    *,
    out_root: str | Path,
    min_voxel_size: float,
    max_fov: tuple[float, float, float],
):
    prefix = Path(name).name.removesuffix(".nii.gz")
    out_dir = Path(out_root) / Path(name).parent
    out_path = out_dir / f"{prefix}.npz"
    if out_path.exists():
        logger.info(f"output {out_path} exists; skipping")
        return

    # minimal preprocessing
    #   1. compute a fast threshold based head mask
    #   2. fit the image to the size/resolution constraints, cropping z from the top of the mask
    #   3. quantize values to uint16 over a quantile range
    t0 = time.time()
    mask_img = preproc.threshold_mask(img)
    t1 = time.time()
    fit_img = preproc.conform_image(img, mask_img, min_voxel_size=min_voxel_size, max_fov=max_fov)
    mask_img = resample_from_to(mask_img, fit_img, order=0)
    t2 = time.time()

    data = fit_img.get_fdata(dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0
    values = data[mask]
    quantiles = np.quantile(values, QUANTILES)
    qs = {str(q): float(v) for q, v in zip(QUANTILES, quantiles)}
    vmin = qs[str(VMIN_QUANTILE)]
    vmax = qs[str(VMAX_QUANTILE)]

    buf = encode_image(values, mask, fit_img.affine, vmin, vmax)
    t3 = time.time()

    # thumbnail images of 3 main views to inspect without downloading the full images.
    thumbnails = make_thumbnails(data, mask, fit_img.header.get_zooms(), vmin, vmax)

    # lots of misc info that could be useful for filtering the data, idk
    # first info about the original image
    dtype = str(img.get_data_dtype())
    orient = "".join(nib.aff2axcodes(img.affine))
    shape = img.shape
    spacing = [float(sz) for sz in img.header.get_zooms()]
    fov = np.array(shape) * spacing

    # info about the fit image and value distribution
    fit_shape = fit_img.shape
    fit_spacing = [float(sz) for sz in fit_img.header.get_zooms()]
    fit_fov = np.array(fit_shape) * fit_spacing
    counts, edges = np.histogram(values, bins=100, range=(quantiles[0], quantiles[-1]))

    # mask info
    mask_indices = np.stack(mask.nonzero(), 1)
    mask_bbox = mask_indices.max(axis=0) - mask_indices.min(axis=0) + 1
    mask_fov = mask_bbox * fit_spacing
    mask_volume = mask.sum() * np.prod(fit_spacing)

    t4 = time.time()

    info = {
        "name": name,
        "dtype": dtype,
        "orient": orient,
        "shape": shape,
        "spacing": spacing,
        "fov": fov.tolist(),
        "fit_shape": fit_shape,
        "fit_spacing": fit_spacing,
        "fit_fov": fit_fov.tolist(),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "qs": qs,
        "mask_fov": mask_fov.tolist(),
        "mask_size": float(mask.sum()),
        "mask_frac": float(mask.mean()),
        "mask_volume": float(mask_volume),
        "vmin": vmin,
        "vmax": vmax,
        "size_bytes": len(buf),
        "mask_time": t1 - t0,
        "fit_time": t2 - t1,
        "encode_time": t3 - t2,
        "total_time": t4 - t0,
    }

    # save output
    # save to a tmp dir first and then atomically rename each file (we don't want
    # corrupt partial results).
    tmp_dir = out_dir.with_name(f".tmp-{secrets.token_hex(3)}-{out_dir.name}")
    tmp_dir.mkdir(parents=True)

    with (tmp_dir / f"{prefix}.npz").open("wb") as f:
        f.write(buf)
    for view, square in thumbnails.items():
        square.save(tmp_dir / f"{prefix}.{view}.jpg", quality=JPEG_QUALITY)
    with (tmp_dir / f"{prefix}.meta.json").open("w") as f:
        print(json.dumps(info), file=f)

    out_dir.mkdir(exist_ok=True)
    for p in tmp_dir.iterdir():
        p.rename(out_dir / p.name)
    tmp_dir.rmdir()

    return


def iter_fomo300k(
    root: str = DEFAULT_ROOT,
    filelist: str | Path | list[str] = DEFAULT_FILELIST,
    suffixes: list[str] | None = DEFAULT_SUFFIXES,
    shuffle: bool = False,
    random_state: int | np.random.Generator | None = None,
    cache_dir: str | Path | None = None,
    prefetch_threads: int = 1,
    storage_options: dict | None = None,
):
    if isinstance(filelist, (str, Path)):
        paths = Path(filelist).read_text().strip().splitlines()
    else:
        paths = filelist
    paths = np.array(paths)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(paths)

    for path, fullpath in prefetch(
        root,
        paths,
        cache_dir=cache_dir,
        max_workers=prefetch_threads,
        storage_options=storage_options,
    ):
        for name, img in read_fomo300k_zip(fullpath, suffixes=suffixes):
            fullname = f"{Path(path).parent}/{name}"
            yield fullname, img


def read_fomo300k_zip(path: str, suffixes: list[str] | None = DEFAULT_SUFFIXES):
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


def make_thumbnails(
    data: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    vmin: float,
    vmax: float,
    size: int = 224,
) -> dict[str, Image.Image]:
    sx, sy, sz = spacing
    cx, cy, cz = [int(c) for c in ndi.center_of_mass(mask)]

    planes = {"sag": data[cx, :, :], "cor": data[:, cy, :], "ax": data[:, :, cz]}
    mask_planes = {"sag": mask[cx, :, :], "cor": mask[:, cy, :], "ax": mask[:, :, cz]}
    spacings = {"sag": (sy, sz), "cor": (sx, sz), "ax": (sx, sy)}

    images = {}
    for view, plane in planes.items():
        plane = np.flipud(plane.T)
        mask_plane = np.flipud(mask_planes[view].T)
        plane = np.clip((plane - vmin) / (vmax - vmin), 0, 1)
        plane = (plane * 255).astype(np.uint8)
        plane = np.where(mask_plane, plane, 127)  # mask as gray
        plane = Image.fromarray(plane)
        images[view] = rescale(plane, spacing=spacings[view], size=size)

    return images


def rescale(img: Image.Image, spacing: tuple[float, float], size: int = 224):
    """Scale image to a target square size while preserving aspect."""
    w, h = img.size
    col_sp, row_sp = spacing
    height = width = size

    scale = height / max(h * row_sp, w * col_sp)
    new_h = max(1, round(h * row_sp * scale))
    new_w = max(1, round(w * col_sp * scale))
    off = ((width - new_w) // 2, (height - new_h) // 2)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    square = Image.new("L", (width, height))
    square.paste(img, off)
    return square


def encode_image(
    values: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    vmin: float,
    vmax: float,
) -> bytes:
    assert vmax > vmin, f"degenerate value range {vmin=}, {vmax=}"

    values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    values = (values * (2**16 - 1)).astype(np.uint16)
    mask_rle = dense_to_brle(mask.ravel(), dtype=np.uint8)

    f = io.BytesIO()
    np.savez(
        f,
        values=values,
        mask=mask_rle,
        vminmax=np.array([vmin, vmax]),
        shape=np.array(mask.shape),
        affine=np.array(affine),
    )
    buf = f.getvalue()
    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help=f"batch index in [0, {TOTAL_BATCHES})")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument("--min-voxel-size", type=float, default=1.0)
    parser.add_argument("--max-fov", type=float, nargs=3, default=(192.0, 240.0, 192.0))
    args = parser.parse_args()
    main(args)
