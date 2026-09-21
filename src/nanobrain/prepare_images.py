import argparse
import json
import logging
import time
import secrets
from compression import zstd
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import scipy.ndimage as ndi
from PIL import Image

from nanobrain.fomo300k import DEFAULT_ROOT, DEFAULT_FILELIST, Fomo300K
import nanobrain.utils.preprocessing as preproc
import nanobrain.utils.misc as misc

logger = logging.getLogger()

TOTAL_BATCHES = 64
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

    offsets = np.linspace(0, len(paths), TOTAL_BATCHES + 1, dtype=np.int64).tolist()
    start, stop = offsets[args.index : args.index + 2]
    paths = paths[start:stop]

    # limit concurrency to prevent fsspec timeout errors
    # the script is going to be parallelized anyway
    dataset = Fomo300K(
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
                nbits=args.nbits,
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
    nbits: int,
):
    prefix = Path(name).name.removesuffix(".nii.gz")
    out_dir = Path(out_root) / Path(name).parent
    out_path = out_dir / f"{prefix}.nii.zst"
    if out_path.exists():
        logger.info(f"output {out_path} exists; skipping")
        return

    # minimal preprocessing
    #   1. compute a fast threshold based head mask
    #   2. fit the image to the size/resolution constraints, cropping z from the top of the mask
    #   3. quantize values to a fixed number of bits
    t0 = time.time()
    mask_img = preproc.threshold_mask(img)
    t1 = time.time()
    fit_img = preproc.conform_image(img, mask_img, min_voxel_size=min_voxel_size, max_fov=max_fov)
    mask_img = resample_from_to(mask_img, fit_img, order=0)
    t2 = time.time()
    q_img, (vmin, vmax) = quantize_image(fit_img, mask_img, nbits=nbits)
    t3 = time.time()

    # thumbnail images of 3 main views to inspect without downloading the full images.
    thumbnails = make_thumbnails(q_img)

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
    data = fit_img.get_fdata(dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0
    values = data[mask]
    counts, edges = np.histogram(values, bins=50, range=(vmin, vmax))
    qs = np.quantile(values, [0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995])

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
        "qs": qs.tolist(),
        "mask_fov": mask_fov.tolist(),
        "mask_size": float(mask.sum()),
        "mask_frac": float(mask.mean()),
        "mask_volume": float(mask_volume),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "mask_time": t1 - t0,
        "fit_time": t2 - t1,
        "q_time": t3 - t2,
        "total_time": t4 - t0,
    }

    # save output
    # save to a tmp dir first and then atomically rename each file (we don't want
    # corrupt partial results).
    tmp_dir = out_dir.with_name(f".tmp-{secrets.token_hex(3)}-{out_dir.name}")
    tmp_dir.mkdir(parents=True)

    with (tmp_dir / f"{prefix}.nii.zst").open("wb") as f:
        # nb, zstd decoding much faster than gzip
        # the main compression that happens is due to:
        #   1. compressing the all zero background
        #   2. compressing the size of each value to nbits rather than 16 bits
        # also briefly considered more bespoke encoding schemes, but they weren't
        # obviously much better.
        f.write(zstd.compress(q_img.to_bytes()))
    for view, square in thumbnails.items():
        square.save(tmp_dir / f"{prefix}.{view}.jpg", quality=JPEG_QUALITY)
    with (tmp_dir / f"{prefix}.meta.json").open("w") as f:
        print(json.dumps(info), file=f)

    out_dir.mkdir(exist_ok=True)
    for p in tmp_dir.iterdir():
        p.rename(out_dir / p.name)
    tmp_dir.rmdir()

    return


def make_thumbnails(img: nib.Nifti1Image, size: int = 224) -> dict[str, Image.Image]:
    data = img.get_fdata(dtype=np.float32)
    mask = data > 0
    vmax = data[mask].max()

    sx, sy, sz = img.header.get_zooms()
    cx, cy, cz = [int(c) for c in ndi.center_of_mass(mask)]

    planes = {"sag": data[cx, :, :], "cor": data[:, cy, :], "ax": data[:, :, cz]}
    spacings = {"sag": (sy, sz), "cor": (sx, sz), "ax": (sx, sy)}

    images = {}
    for view, plane in planes.items():
        plane = np.flipud(plane.T)
        plane = (plane * (255 / vmax)).astype(np.uint8)
        plane = np.where(plane > 0, plane, 127)  # mask as gray
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


def quantize_image(
    img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    nbits: int = 12,
    qs: tuple[float, float] = (0.005, 0.995),
) -> tuple[nib.Nifti1Image, tuple[float, float]]:
    """Quantize image values to a max number of bits and return as int16.

    Values are truncated to a quantile range and then quantized to [1, 2**nbits).
    Background mask is set to 0.

    (Nb, int16 used instead of uint16 bc it seems more typical for nifti.)
    """
    assert 0 < nbits <= 15, f"invalid {nbits=}, expected in (0, 15)"
    data = img.get_fdata(dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0

    values = data[mask]
    vmin, vmax = np.quantile(values, qs)
    assert vmax > vmin, f"degenerate value range {vmin=}, {vmax=}"
    values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    values = (values * (2**nbits - 2)).astype(np.int16) + 1

    data = np.zeros(data.shape, dtype=np.int16)
    data[mask] = values
    return nib.Nifti1Image(data, img.affine), (vmin, vmax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help=f"batch index in [0, {TOTAL_BATCHES})")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument("--min-voxel-size", type=float, default=1.0)
    parser.add_argument("--max-fov", type=float, nargs=3, default=(208.0, 240.0, 208.0))
    parser.add_argument("--nbits", type=int, default=12)
    args = parser.parse_args()
    main(args)
