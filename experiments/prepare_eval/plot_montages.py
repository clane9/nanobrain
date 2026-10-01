import argparse
from pathlib import Path

import datasets as hfds
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanobrain.eval.tasks import EVAL_ROOT, decode_nifti_zst

DATASETS = ["fomo_task1", "fomo_task2", "fomo_task3", "fomo_task4", "fomo_task5"]
REFERENCE_KEYS = {
    "fomo_task1": "flair",
    "fomo_task2": "flair",
    "fomo_task3": "t1w",
    "fomo_task4": "t2w",
    "fomo_task5": "t1w",
}
VIEWS = ["sag", "cor", "ax"]
PANEL_SIZE = 1.4

# synthseg labels with colors from FreeSurferColorLUT.txt
# https://github.com/freesurfer/freesurfer/blob/dev/distribution/FreeSurferColorLUT.txt
SYNTHSEG_COLORS = {
    2: (245, 245, 245),
    3: (205, 62, 78),
    4: (120, 18, 134),
    5: (196, 58, 250),
    7: (220, 248, 164),
    8: (230, 148, 34),
    10: (0, 118, 14),
    11: (122, 186, 220),
    12: (236, 13, 176),
    13: (12, 48, 255),
    14: (204, 182, 142),
    15: (42, 204, 164),
    16: (119, 159, 176),
    17: (220, 216, 20),
    18: (103, 255, 255),
    24: (60, 60, 60),
    26: (255, 165, 0),
    28: (145, 42, 42),
    41: (245, 245, 245),
    42: (205, 62, 78),
    43: (120, 18, 134),
    44: (196, 58, 250),
    46: (220, 248, 164),
    47: (230, 148, 34),
    49: (0, 118, 14),
    50: (122, 186, 220),
    51: (236, 13, 176),
    52: (13, 48, 255),
    53: (220, 216, 20),
    54: (103, 255, 255),
    58: (255, 165, 0),
    60: (165, 42, 42),
}
SEG_COLORS = {1: (255, 0, 0), 2: (255, 255, 0)}


def take_slices(volume: np.ndarray, center: np.ndarray) -> list[np.ndarray]:
    x, y, z = center
    return [volume[x].T, volume[:, y].T, volume[:, :, z].T]


def label_overlay(gray: np.ndarray, labels: np.ndarray, colors: dict, alpha: float) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for label, color in colors.items():
        selected = labels == label
        rgb[selected] = (1 - alpha) * rgb[selected] + alpha * np.array(color) / 255
    return rgb


def plot_dataset(name: str, num_subjects: int, seed: int) -> plt.Figure:
    dataset = hfds.load_from_disk(EVAL_ROOT / name)
    image_keys = [
        key
        for key, feature in dataset.features.items()
        if feature.dtype == "large_binary" and key not in ("mask", "synthseg", "seg")
    ]
    has_seg = "seg" in dataset.column_names
    panel_keys = image_keys + ["synthseg"] + (["seg"] if has_seg else [])
    reference_key = REFERENCE_KEYS[name]

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), min(num_subjects, len(dataset)), replace=False))

    num_cols = len(panel_keys) * len(VIEWS)
    fig, axes = plt.subplots(
        len(indices),
        num_cols,
        figsize=(num_cols * PANEL_SIZE, len(indices) * PANEL_SIZE),
        squeeze=False,
    )
    for row, index in enumerate(indices):
        sample = dataset[int(index)]
        images = {
            key: np.asarray(decode_nifti_zst(sample[key]).dataobj) / 65535 for key in image_keys
        }
        mask = np.asarray(decode_nifti_zst(sample["mask"]).dataobj) > 0
        synthseg = np.asarray(decode_nifti_zst(sample["synthseg"]).dataobj)
        seg = None
        if has_seg and sample["seg"] is not None:
            seg = np.asarray(decode_nifti_zst(sample["seg"]).dataobj)

        # slice where each view shows the most label, otherwise through the head mask center.
        # not the label centroid, which misses bilateral structures like the task 4 nerves
        if seg is not None and seg.any():
            labeled = seg > 0
            center = np.array(
                [
                    labeled.sum(axis=(1, 2)).argmax(),
                    labeled.sum(axis=(0, 2)).argmax(),
                    labeled.sum(axis=(0, 1)).argmax(),
                ]
            )
        else:
            nonzero = np.argwhere(mask)
            center = (nonzero.min(axis=0) + nonzero.max(axis=0)) // 2

        reference_slices = take_slices(images[reference_key], center)
        panels = {key: take_slices(images[key], center) for key in image_keys}
        panels["synthseg"] = [
            label_overlay(gray, labels, SYNTHSEG_COLORS, alpha=0.6)
            for gray, labels in zip(reference_slices, take_slices(synthseg, center))
        ]
        if has_seg:
            seg_volume = seg if seg is not None else np.zeros_like(synthseg)
            panels["seg"] = [
                label_overlay(gray, labels, SEG_COLORS, alpha=0.6)
                for gray, labels in zip(reference_slices, take_slices(seg_volume, center))
            ]

        for key_id, key in enumerate(panel_keys):
            for view_id, view in enumerate(VIEWS):
                ax = axes[row, key_id * len(VIEWS) + view_id]
                ax.imshow(panels[key][view_id], cmap="gray", vmin=0, vmax=1, origin="lower")
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(f"{key} {view}", fontsize=8)

        targets = [f"{key} {sample[key]}" for key in ("label", "age") if key in sample]
        axes[row, 0].text(
            -0.05,
            0.5,
            "\n".join([sample["subject"]] + targets),
            transform=axes[row, 0].transAxes,
            fontsize=7,
            ha="right",
            va="center",
        )

    fig.suptitle(name)
    fig.tight_layout()
    return fig


def main(args: argparse.Namespace):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        fig = plot_dataset(name, args.num_subjects, args.seed)
        fig.savefig(args.out_dir / f"montage_{name}.png", dpi=args.dpi)
        plt.close(fig)
        print(f"saved {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--num-subjects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/prepare_eval/figures"))
    args = parser.parse_args()
    main(args)
