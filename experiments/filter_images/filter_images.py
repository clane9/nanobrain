import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ADC_MAX_VALUE = 1.0
MIN_MASK_EXTENT = 100.0
MAX_MASK_FRACTION = 0.75
RULES = ["adc_map", "partial_coverage", "bad_mask"]


def main(args: argparse.Namespace):
    df = pd.read_parquet(args.metadata)
    mask_extent = np.stack(df.mask_fov)

    filters = df[["name", "path", "dataset", "suffix", "size_bytes"]].copy()
    filters["adc_map"] = (df.suffix == "bval1000") & (df.vmax < ADC_MAX_VALUE)
    filters["partial_coverage"] = mask_extent.min(axis=1) < MIN_MASK_EXTENT
    filters["bad_mask"] = df.mask_frac > MAX_MASK_FRACTION
    filters["keep"] = ~filters[RULES].any(axis=1)

    for rule in RULES:
        count = filters[rule].sum()
        print(f"{rule}: {count:,} ({count / len(filters):.1%})")
    kept = filters[filters.keep]
    print(f"kept: {len(kept):,} / {len(filters):,} ({len(kept) / len(filters):.1%})")
    print(f"kept size: {kept.size_bytes.sum() / 1e9:.0f} GB")
    print(kept.groupby("suffix").size().to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    filters.to_parquet(args.out_dir / "filter.parquet")

    filelist = [name.replace(".nii.gz", ".nii.zst") for name in kept.name]
    (args.out_dir / "filelist_filtered.txt").write_text("\n".join(filelist) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/FOMO300K_images/metadata.parquet")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/filter_images"))
    args = parser.parse_args()
    main(args)
