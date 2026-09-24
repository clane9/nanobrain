from typing import Literal

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")

VIEW_AXES = {"sag": 0, "cor": 1, "ax": 2}
BACKGROUND_GRAY = 0.0
MASKED_GRAY = 0.5


def plot_mask_pred(
    images: np.ndarray,
    pred: np.ndarray,
    visible_mask: np.ndarray,
    pred_mask: np.ndarray,
    img_mask: np.ndarray,
    view: Literal["sag", "cor", "ax"] = "ax",
    nrow: int = 4,
    ncol: int = 2,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> Figure:
    # all volumes shape [B, X, Y, Z]
    visible_mask = visible_mask > 0
    img_mask = img_mask > 0
    pred_mask = pred_mask > 0
    visible_mask = visible_mask & img_mask
    pred_mask = pred_mask & img_mask

    # center slice, oriented with superior/anterior up
    axis = VIEW_AXES[view] + 1
    index = images.shape[axis] // 2
    images = np.take(images, index, axis=axis)
    pred = np.take(pred, index, axis=axis)
    visible_mask = np.take(visible_mask, index, axis=axis)
    pred_mask = np.take(pred_mask, index, axis=axis)
    img_mask = np.take(img_mask, index, axis=axis)

    # gray levels as a fraction of the display range
    background = vmin + BACKGROUND_GRAY * (vmax - vmin)
    masked = vmin + MASKED_GRAY * (vmax - vmin)

    visible_panel = np.where(visible_mask, images, masked)
    visible_panel = np.where(img_mask, visible_panel, background)
    pred_panel = np.where(pred_mask, pred, images)
    pred_panel = np.where(img_mask, pred_panel, background)
    image_panel = np.where(img_mask, images, background)

    height, width = images.shape[1:]
    size = max(height, width)
    num_examples = min(len(images), nrow * ncol)

    # roughly one pixel per voxel
    dpi = 100
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(1.05 * ncol * 3 * size / dpi, 1.05 * nrow * size / dpi),
        dpi=dpi,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03, hspace=0.03)

    for ii, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if ii >= num_examples:
            continue

        # three panels padded to square and stacked with no gap.
        strip = np.full((size, 3 * size), background, dtype=np.float32)
        for jj, panel in enumerate([visible_panel, pred_panel, image_panel]):
            plane = np.flipud(panel[ii].T)
            plane_height, plane_width = plane.shape
            top = (size - plane_height) // 2
            left = jj * size + (size - plane_width) // 2
            strip[top : top + plane_height, left : left + plane_width] = plane

        ax.imshow(strip, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    return fig
