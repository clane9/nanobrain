import argparse
import os
import shutil
import sys
import time
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from image_codecs import CODECS

CORPUS = None


def load_source(path: Path):
    img = nib.Nifti1Image.from_bytes(zstd.decompress(path.read_bytes()))
    return np.asarray(img.dataobj, dtype=np.int16), img.affine


def init_worker(corpus: Path):
    global CORPUS
    CORPUS = corpus


def read_only(task):
    name, index = task
    path = CORPUS / name / f"{index:05d}.bin"
    with open(path, "rb") as f:
        buf = f.read()
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return len(buf), 0


def read_decode(task):
    name, index = task
    path = CORPUS / name / f"{index:05d}.bin"
    with open(path, "rb") as f:
        buf = f.read()
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    data, _ = CODECS[name][1](buf)
    return len(buf), data.size


def noop(x):
    return x


def evict(paths):
    for path in paths:
        with open(path, "rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main(args: argparse.Namespace):
    names = args.codecs.split(",") if args.codecs else list(CODECS)
    for name in names:
        assert name in CODECS, f"unknown codec {name}, have {list(CODECS)}"

    all_paths = Path(args.filelist).read_text().split()
    rng = np.random.default_rng(args.seed)
    src_paths = [Path(args.root) / p for p in rng.choice(all_paths, args.num_images, replace=False)]

    corpus = Path(args.corpus)
    for name in names:
        (corpus / name).mkdir(parents=True, exist_ok=True)

    sizes = {name: [] for name in names}
    encode_times = {name: [] for name in names}
    decode_times = {name: [] for name in names}
    num_voxels = 0
    num_masked = 0

    print(f"encoding {len(src_paths)} images to {len(names)} formats in {corpus}")
    for index, src in enumerate(src_paths):
        data, affine = load_source(src)
        num_voxels += data.size
        num_masked += int((data != 0).sum())

        for name in names:
            encode, decode = CODECS[name]
            t0 = time.perf_counter()
            buf = encode(data, affine)
            encode_times[name].append(time.perf_counter() - t0)

            best = np.inf
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out, out_affine = decode(buf)
                best = min(best, time.perf_counter() - t0)
            assert np.array_equal(out, data), f"{name} roundtrip mismatch"
            assert np.allclose(out_affine, affine), f"{name} affine mismatch"

            decode_times[name].append(best)
            sizes[name].append(len(buf))
            (corpus / name / f"{index:05d}.bin").write_bytes(buf)

        if (index + 1) % 32 == 0:
            print(f"  [{index + 1}/{len(src_paths)}]")

    baseline = np.sum(sizes["nii.zst"]) if "nii.zst" in sizes else np.sum(sizes[names[0]])
    print(
        f"\n{len(src_paths)} images, {num_voxels / 1e6:.0f} Mvoxels, mask frac {num_masked / num_voxels:.2f}"
    )
    print(
        f"{'codec':<28} {'MB/im':>7} {'vs zst':>7} {'bits/vox':>9} "
        f"{'enc MB/s':>9} {'enc im/s':>9} {'dec MB/s':>9} {'dec im/s':>9}"
    )
    for name in names:
        total = np.sum(sizes[name])
        encode_total = np.sum(encode_times[name])
        decode_total = np.sum(decode_times[name])
        print(
            f"{name:<28} {total / len(src_paths) / 1e6:7.2f} {total / baseline:7.3f} "
            f"{8 * total / num_masked:9.2f} "
            f"{total / encode_total / 1e6:9.1f} {len(src_paths) / encode_total:9.1f} "
            f"{total / decode_total / 1e6:9.1f} {len(src_paths) / decode_total:9.1f}"
        )

    print(
        f"\n{'codec':<28} {'workers':>7} {'read MB/s':>10} {'read im/s':>10} "
        f"{'e2e MB/s':>9} {'e2e im/s':>9} {'e2e Mvox/s':>11}"
    )
    for num_workers in args.workers:
        with ProcessPoolExecutor(
            num_workers, initializer=init_worker, initargs=(corpus,)
        ) as executor:
            list(executor.map(noop, range(4 * num_workers)))
            for name in names:
                paths = [corpus / name / f"{index:05d}.bin" for index in range(len(src_paths))]
                tasks = [(name, index) for index in range(len(src_paths))]

                evict(paths)
                t0 = time.perf_counter()
                results = list(executor.map(read_only, tasks, chunksize=1))
                read_elapsed = time.perf_counter() - t0
                read_bytes = sum(r[0] for r in results)

                evict(paths)
                t0 = time.perf_counter()
                results = list(executor.map(read_decode, tasks, chunksize=1))
                e2e_elapsed = time.perf_counter() - t0
                e2e_voxels = sum(r[1] for r in results)

                print(
                    f"{name:<28} {num_workers:>7} "
                    f"{read_bytes / read_elapsed / 1e6:10.0f} {len(paths) / read_elapsed:10.1f} "
                    f"{read_bytes / e2e_elapsed / 1e6:9.0f} {len(paths) / e2e_elapsed:9.1f} "
                    f"{e2e_voxels / e2e_elapsed / 1e6:11.0f}"
                )

    if not args.keep_corpus:
        shutil.rmtree(corpus)
        print(f"\nremoved {corpus}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/FOMO300K_images"))
    parser.add_argument(
        "--filelist", type=Path, default=Path("experiments/filter_images/filelist_subsampled.txt")
    )
    parser.add_argument("--corpus", type=Path, default=Path("experiments/codec_benchmark/corpus"))
    parser.add_argument("--codecs", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-corpus", action="store_true")
    main(parser.parse_args())
