import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUFFIXES = ["T1w", "T2w", "FLAIR", "bval0", "bval1000"]
AXES = ["x", "y", "z"]


def plot_suffixes(df: pd.DataFrame, out_dir: Path):
    counts = df.groupby("suffix").size().reindex(SUFFIXES)
    gigabytes = df.groupby("suffix").size_bytes.sum().reindex(SUFFIXES) / 1e9

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(SUFFIXES, counts)
    axs[0].set_title(f"Images per suffix (total {len(df):,})")
    axs[1].bar(SUFFIXES, gigabytes)
    axs[1].set_title(f"GB per suffix (total {gigabytes.sum():.0f} GB)")
    for ax, values in zip(axs, [counts, gigabytes]):
        for ii, value in enumerate(values):
            ax.text(ii, value, f"{value:,.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "suffixes.png", dpi=150)
    plt.close(fig)


def plot_datasets(df: pd.DataFrame, out_dir: Path):
    counts = df.groupby("dataset").size().sort_values(ascending=False)
    cumulative = counts.cumsum() / counts.sum()
    top = counts.head(20)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].barh(top.index[::-1], top.values[::-1])
    axs[0].set_title("Top 20 datasets")
    axs[0].set_xlabel("Images")

    ranks = np.arange(1, len(counts) + 1)
    axs[1].plot(ranks, counts.values)
    axs[1].set_yscale("log")
    axs[1].set_title(f"Images per dataset ({len(counts)} datasets)")
    axs[1].set_xlabel("Dataset rank")
    axs[1].set_ylabel("Images")

    axs[2].plot(ranks, cumulative.values)
    axs[2].set_title("Cumulative fraction of images")
    axs[2].set_xlabel("Dataset rank")
    axs[2].grid()
    for frac in [0.5, 0.9]:
        rank = int(np.searchsorted(cumulative.values, frac)) + 1
        axs[2].annotate(
            f"{frac:.0%} in top {rank}", (rank, frac), xytext=(10, -15), textcoords="offset points"
        )
    fig.tight_layout()
    fig.savefig(out_dir / "datasets.png", dpi=150)
    plt.close(fig)


def plot_subjects(df: pd.DataFrame, out_dir: Path):
    images_per_subject = df.groupby("subject").size()
    sessions_per_subject = df.groupby("subject").session.nunique()
    subjects_per_dataset = df.groupby("dataset").subject.nunique()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(images_per_subject, bins=np.arange(1, 52) - 0.5)
    axs[0].set_title(f"Images per subject ({len(images_per_subject):,} subjects)")
    axs[1].hist(sessions_per_subject, bins=np.arange(1, 22) - 0.5)
    axs[1].set_title("Sessions per subject")
    axs[2].hist(subjects_per_dataset, bins=np.logspace(0, 4, 41))
    axs[2].set_xscale("log")
    axs[2].set_title("Subjects per dataset")
    for ax in axs:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "subjects.png", dpi=150)
    plt.close(fig)


def plot_spacing(df: pd.DataFrame, out_dir: Path):
    spacing = np.stack(df.spacing)
    min_spacing = spacing.min(axis=1)
    max_spacing = spacing.max(axis=1)
    bins = np.linspace(0, 8, 81)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(min_spacing, bins=bins)
    axs[0].set_title("Min voxel spacing (mm)")
    axs[1].hist(max_spacing, bins=bins)
    axs[1].set_title("Max voxel spacing (mm)")
    axs[1].text(
        0.95, 0.9, f"≥3mm: {np.mean(max_spacing >= 3):.1%}", transform=axs[1].transAxes, ha="right"
    )
    axs[2].hist(max_spacing / min_spacing, bins=np.linspace(1, 10, 91))
    axs[2].set_title("Anisotropy (max / min spacing)")
    for ax in axs:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "spacing.png", dpi=150)
    plt.close(fig)


def plot_fov(df: pd.DataFrame, out_dir: Path):
    fov = np.stack(df.fov)
    mask_fov = np.stack(df.mask_fov)
    bins = np.linspace(0, 400, 81)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ii, axis in enumerate(AXES):
        axs[0, ii].hist(fov[:, ii], bins=bins)
        axs[0, ii].set_title(f"Original FOV {axis} (mm)")
        axs[1, ii].hist(mask_fov[:, ii], bins=bins)
        axs[1, ii].set_title(f"Mask extent {axis} (mm)")
    fig.tight_layout()
    fig.savefig(out_dir / "fov.png", dpi=150)
    plt.close(fig)


def plot_mask(df: pd.DataFrame, out_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for suffix in SUFFIXES:
        sub = df[df.suffix == suffix]
        axs[0].hist(sub.mask_frac, bins=np.linspace(0, 1, 51), histtype="step", label=suffix)
        axs[1].hist(
            sub.mask_volume / 1e6, bins=np.linspace(0, 7, 71), histtype="step", label=suffix
        )
    axs[0].set_title("Mask fraction of volume")
    axs[1].set_title("Mask volume (L)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mask.png", dpi=150)
    plt.close(fig)


def plot_intensity(df: pd.DataFrame, out_dir: Path):
    dtypes = df.dtype.value_counts()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for suffix in SUFFIXES:
        sub = df[df.suffix == suffix]
        axs[0].hist(np.log10(sub.vmax), bins=np.linspace(-3, 7, 101), histtype="step", label=suffix)
        axs[1].hist(
            sub["mean"] / sub.vmax, bins=np.linspace(0, 1, 51), histtype="step", label=suffix
        )
    axs[0].set_title("log10 of 99.5% quantile in mask")
    axs[0].legend()
    axs[1].set_title("Mean / 99.5% quantile in mask")
    axs[2].bar(dtypes.index, dtypes.values)
    axs[2].set_yscale("log")
    axs[2].set_title("Original dtype")
    axs[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity.png", dpi=150)
    plt.close(fig)


def plot_contrast(df: pd.DataFrame, out_dir: Path):
    quantiles = np.stack(df.qs)
    p10 = quantiles[:, 2]
    median = quantiles[:, 4]
    p90 = quantiles[:, 6]
    contrast = (p90 - p10) / median

    fig, ax = plt.subplots(figsize=(6, 4))
    for suffix in SUFFIXES:
        sub = contrast[df.suffix == suffix]
        ax.hist(sub, bins=np.linspace(0, 4, 81), histtype="step", label=suffix)
    ax.set_yscale("log")
    ax.set_title("Contrast in mask: (p90 - p10) / median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "contrast.png", dpi=150)
    plt.close(fig)


def plot_file_size(df: pd.DataFrame, out_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for suffix in SUFFIXES:
        sub = df[df.suffix == suffix]
        axs[0].hist(
            sub.size_bytes / 1e6, bins=np.linspace(0, 12, 61), histtype="step", label=suffix
        )
        axs[1].hist(sub.total_time, bins=np.linspace(0, 10, 51), histtype="step", label=suffix)
    axs[0].set_title("File size (MB)")
    axs[1].set_title("Processing time (s)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "file_size.png", dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace):
    df = pd.read_parquet(args.metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_suffixes(df, args.out_dir)
    plot_datasets(df, args.out_dir)
    plot_subjects(df, args.out_dir)
    plot_spacing(df, args.out_dir)
    plot_fov(df, args.out_dir)
    plot_mask(df, args.out_dir)
    plot_intensity(df, args.out_dir)
    plot_contrast(df, args.out_dir)
    plot_file_size(df, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/FOMO300K_images/metadata.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("experiments/filter_images/figures/summary")
    )
    args = parser.parse_args()
    main(args)
