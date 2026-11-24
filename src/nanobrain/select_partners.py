"""Select partner images (same subject, any session) for the images of a pretraining filelist.

uv run python -m nanobrain.select_partners src/nanobrain/config/fomo300k_images.txt \
    src/nanobrain/config/fomo300k_partners.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nanobrain.select_images import ANAT_SUFFIXES, DATA_ROOT, DWI_SHELL_RANGE, MIN_BRAIN_EXTENT

# partners are read whole and resampled on the gpu, skip the few very large stored grids
MAX_STORED_VOXELS = 30e6
SUBJECT = ["collection", "subject"]


def main(args: argparse.Namespace):
    images = pd.read_parquet(DATA_ROOT / "fomo300k" / "tables" / "images.parquet")
    images["npz"] = (
        "derivatives/"
        + images["path"].str.removeprefix("source/").str.removesuffix(".nii.gz")
        + ".npz"
    )
    # same sequences and qc filters as select_images
    bval = pd.to_numeric(images["suffix"].str.extract(r"^dwi_bval(\d+)$")[0])
    is_shell = (images["dwi_type"] == "mean") & bval.between(*DWI_SHELL_RANGE)
    sequence = pd.Series(None, index=images.index, dtype=object)
    sequence[images["suffix"].isin(ANAT_SUFFIXES)] = images["suffix"]
    sequence[images["dwi_type"] == "b0"] = "dwi_b0"
    sequence[is_shell] = "dwi_b1000"
    images["sequence"] = sequence

    filelist = args.filelist.read_text().split()
    anchors = images[images["npz"].isin(set(filelist))]
    assert len(anchors) == len(filelist), "filelist images missing from images.parquet"
    print(f"filelist images: {len(filelist)}")

    pool = images[images["sequence"].notna()]
    pool = pool[pool["warnings"].map(len) == 0]
    brain_extent = np.array(pool["mask_extent"].tolist())
    pool = pool[brain_extent.min(axis=1) >= MIN_BRAIN_EXTENT]
    print(f"qc images of the {pool['sequence'].nunique()} sequences: {len(pool)}")

    # stored grids are cropped to the brain mask
    brain_extent = np.array(pool["mask_extent"].tolist())
    spacing = np.array(pool["spacing"].tolist())
    stored_voxels = np.prod(brain_extent / spacing + 1, axis=1)
    pool = pool[stored_voxels <= MAX_STORED_VOXELS]
    print(f"stored grid <= {MAX_STORED_VOXELS / 1e6:.0f}M voxels: {len(pool)}")

    pairs = anchors[SUBJECT + ["npz", "session", "sequence"]].merge(
        pool[SUBJECT + ["npz", "session", "sequence"]], on=SUBJECT, suffixes=("", "_partner")
    )
    pairs = pairs[pairs["npz"] != pairs["npz_partner"]]
    num_partners = pairs.groupby("npz").size()
    print(f"pairs: {len(pairs)}")
    print(
        f"images with a partner: {len(num_partners)} ({len(num_partners) / len(filelist):.1%}), "
        f"partners per image: median {num_partners.median():.0f}, max {num_partners.max()}"
    )
    # fractions of all draws. a draw picks each partner of an image with equal chance
    weight = 1 / pairs["npz"].map(num_partners) / len(filelist)
    same_session = pairs["session"] == pairs["session_partner"]
    print(f"draws with a partner from the same session: {weight[same_session].sum():.1%}")
    other_sequence = pairs["sequence"] != pairs["sequence_partner"]
    print(f"draws with a partner of another sequence: {weight[other_sequence].sum():.1%}")
    print("draws by partner sequence:")
    print(weight.groupby(pairs["sequence_partner"]).sum().round(3).to_string())

    table = pd.DataFrame({"path": pairs["npz"], "partner": pairs["npz_partner"]})
    table = table.sort_values(["path", "partner"])
    table.to_parquet(args.out, index=False, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", type=Path, help="pretraining filelist of npz paths")
    parser.add_argument("out", type=Path, help="output parquet of (path, partner) pairs")
    args = parser.parse_args()
    main(args)
