# Filter images

Remove corrupt and non-brain images from `data/FOMO300K_images`, then keep one image per suffix.
Subject and session subsampling happens upstream in `experiments/subsample_sessions/`.

## Summary

```bash
uv run python experiments/filter_images/index_metadata.py  # data/FOMO300K_images/metadata.parquet
uv run python experiments/filter_images/plot_summary.py    # figures/summary/
```

![suffixes](figures/summary/suffixes.png)

![datasets](figures/summary/datasets.png)

![subjects](figures/summary/subjects.png)

![spacing](figures/summary/spacing.png)

![fov](figures/summary/fov.png)

![mask](figures/summary/mask.png)

![intensity](figures/summary/intensity.png)

![contrast](figures/summary/contrast.png)

![file_size](figures/summary/file_size.png)

## Filtering

```bash
uv run python experiments/filter_images/filter_images.py   # filter.parquet, filelist_filtered.txt
uv run python experiments/filter_images/plot_montages.py   # figures/montages/
```

| Rule | Catches | Excluded | Only this rule |
|---|---|---|---|
| `adc_map`: bval1000 with 99.5% quantile < 1 | ADC-like maps labeled bval1000 | 2,742 | 2,213 |
| `partial_coverage`: mask extent < 100 mm on any axis | thin slabs, neck/spine scans | 2,579 | 2,139 |
| `bad_mask`: mask > 75% of volume | noisy background in mask | 754 | 36 |
| `low_contrast`: (p90 - p10) / median < 0.65 in mask | flat images, phantoms | 183 | 88 |
| `saturated`: median / 99.5% quantile > 0.9 | inverted masks on phantoms, clipped images | 77 | 24 |

**Kept 54,676 / 60,079 images (278 GB).**

**Excluded `adc_map`**

![adc_map](figures/montages/excluded_adc_map.png)

**Excluded `partial_coverage`**

![partial_coverage](figures/montages/excluded_partial_coverage.png)

**Excluded `low_contrast`**

![low_contrast](figures/montages/excluded_low_contrast.png)

**Excluded `saturated`**

![saturated](figures/montages/excluded_saturated.png)

**Random kept**

![kept](figures/montages/kept_0.png)

## Subsampling

```bash
uv run python experiments/filter_images/subsample_images.py  # subsample.parquet, filelist_subsampled.txt
uv run python experiments/filter_images/plot_subsample.py    # figures/subsample/
```

Keep the first image per suffix in each session.

| Stage | Images | Subjects | GB |
|---|---|---|---|
| Filtered | 54,676 | 32,099 | 278 |
| One image per suffix | 46,594 | 32,099 | 250 |

Final: T1w 30,999, T2w 5,695, FLAIR 1,998, bval0 5,462, bval1000 2,440. Filtering drops 710 of the
32,809 subjects in the split entirely, and 16 of 880 datasets.

![suffixes](figures/subsample/suffixes.png)

![datasets](figures/subsample/datasets.png)

![subjects](figures/subsample/subjects.png)
