import argparse
from pathlib import Path

import pandas as pd

SUFFIXES = ["T1w", "T2w", "FLAIR", "bval0", "bval1000"]
COLUMNS = ["name", "path", "dataset", "subject", "session", "suffix", "size_bytes"]


def main(args: argparse.Namespace):
    metadata = pd.read_parquet(args.metadata, columns=COLUMNS)
    filters = pd.read_parquet(args.filters, columns=["name", "keep"])
    filtered = metadata.merge(filters, on="name")
    filtered = filtered[filtered.keep].drop(columns="keep")
    filtered = filtered.sort_values("name").reset_index(drop=True)

    subsampled = filtered.groupby(["session", "suffix"]).head(1)

    stages = {"filtered": filtered, "one per suffix": subsampled}
    for label, df in stages.items():
        dataset_counts = df.groupby("dataset").size().sort_values(ascending=False)
        top_10_share = dataset_counts.iloc[:10].sum() / len(df)
        suffix_counts = df.groupby("suffix").size().reindex(SUFFIXES)
        print(
            f"{label}: {len(df):,} images, {df.subject.nunique():,} subjects, "
            f"{df.dataset.nunique()} datasets, {df.size_bytes.sum() / 1e9:.0f} GB, "
            f"top 10 datasets {top_10_share:.0%}"
        )
        print("  " + ", ".join(f"{suffix} {count:,}" for suffix, count in suffix_counts.items()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subsampled.to_parquet(args.out_dir / "subsample.parquet")
    filelist = [name.replace(".nii.gz", ".npz") for name in subsampled.name]
    (args.out_dir / "filelist_subsampled.txt").write_text("\n".join(filelist) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/FOMO300K_images/metadata.parquet")
    )
    parser.add_argument(
        "--filters", type=Path, default=Path("experiments/filter_images/filter.parquet")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/filter_images"))
    args = parser.parse_args()
    main(args)
