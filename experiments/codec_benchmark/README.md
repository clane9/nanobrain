# Codec benchmark

Compare encodings for the prepared images in `data/FOMO300K_images`. Current format is `.nii.zst`.

```bash
uv run python experiments/codec_benchmark/benchmark.py
uv run --with imagecodecs python experiments/codec_benchmark/benchmark.py  # adds npz-jxl
```

Codecs are in `image_codecs.py`. All use a standard container (nifti or npz) and store the affine.
Sparse variants store a head mask plus in-mask values. `rle` encodes the mask as uint8 binary run
lengths (`nanobrain.utils.rle`), `pack12` splits values into a low byte plane plus packed high
nibbles.

## Size and single thread throughput

256 random subsampled images, 1,639 Mvoxels, mask frac 0.42.

| codec | MB/im | vs zst | bits/vox | enc im/s | dec im/s |
|---|---|---|---|---|---|
| `nii.gz` | 4.15 | 0.993 | 12.37 | 4.5 | 26.3 |
| `nii.zst` (current) | 4.18 | 1.000 | 12.46 | 24.3 | 98.3 |
| `npz` | 12.80 | 3.062 | 38.15 | 203.0 | 201.0 |
| `npz-zlib` | 4.15 | 0.993 | 12.37 | 4.5 | 25.2 |
| `npz-sparse` | 6.17 | 1.476 | 18.39 | 39.7 | 187.1 |
| `npz-sparse-rle` | 5.44 | 1.301 | 16.21 | 16.3 | 189.3 |
| `npz-sparse-rle-pack12` | 4.10 | 0.980 | 12.21 | 15.9 | 143.8 |
| `npz-sparse-rle-zstd` | 3.73 | 0.893 | 11.13 | 6.7 | 72.3 |
| `npz-sparse-rle-pack12-zstd` | 3.45 | 0.824 | 10.27 | 10.0 | 98.5 |
| `npz-jxl` | 2.86 | 0.683 | 8.51 | 7.1 | 11.8 |

## End to end (read + decode)

Images/s off virtiofs, page cache evicted per file, warm process pool.

| codec | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| `nii.zst` (current) | 34.9 | 249.7 | 425.0 | 499.5 |
| `npz` | 54.1 | 312.1 | 413.8 | 493.0 |
| `npz-sparse` | 57.5 | 381.0 | 613.6 | 797.4 |
| `npz-sparse-rle` | 62.6 | **409.2** | 666.1 | 865.7 |
| `npz-sparse-rle-pack12` | 51.8 | **350.4** | 557.2 | 506.9 |
| `npz-sparse-rle-pack12-zstd` | 38.7 | 268.6 | 432.2 | 471.1 |
| `npz-jxl` | 8.1 | 62.8 | 126.3 | 163.8 |

## Conclusions

Go with `npz-sparse-rle`. Small enough, simple, fast loading.

Measurement notes: build the process pool outside the timed region (python 3.14 defaults to
forkserver, spawn costs 0.3-1.0 s), and evict the page cache with `posix_fadvise` before each pass.
