import argparse
from pathlib import Path

import pandas as pd

from plot_summary import plot_datasets, plot_subjects, plot_suffixes


def main(args: argparse.Namespace):
    df = pd.read_parquet(args.subsample)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_suffixes(df, args.out_dir)
    plot_datasets(df, args.out_dir)
    plot_subjects(df, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsample", type=Path, default=Path("experiments/filter_images/subsample.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("experiments/filter_images/figures/subsample")
    )
    args = parser.parse_args()
    main(args)
