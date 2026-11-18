from typing import Literal

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

matplotlib.use("Agg")

VIEW_AXES = {"sag": 0, "cor": 1, "ax": 2}
# pca components are whitened and scaled by the first component's std; clip range for rgb
PCA_CLIP = 2.3


def feature_pca_rgb(features: Tensor) -> Tensor:
    """Top 3 pcs of one image's tokens [N, D] as rgb [N, 3] in [0, 1].

    As pos_embed/pca_embed.py (capi eval_visualizations): whitened pca, each component flipped
    to positive skew, scaled by the first component's std, clipped to +-PCA_CLIP.
    """
    centered = (features - features.mean(dim=0)).double()
    covariance = centered.T @ centered / (len(features) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)[:3]
    eigenvectors = eigenvectors.flip(1)[:, :3]
    projected = centered @ eigenvectors / eigenvalues.sqrt()
    skew = projected.pow(3).mean(dim=0) / projected.pow(2).mean(dim=0).pow(1.5)
    projected = projected * torch.where(skew < 0, -1.0, 1.0)
    projected = projected / projected[:, 0].std()
    rgb = ((projected + PCA_CLIP) / (2 * PCA_CLIP)).clamp(0, 1)
    return rgb.float()


def plot_feature_pca(
    images: np.ndarray,
    img_mask: np.ndarray,
    pca_maps: np.ndarray,
    view: Literal["sag", "cor", "ax"] = "ax",
    nrow: int = 4,
    ncol: int = 4,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> Figure:
    """Center slice of each image next to its feature pca rgb, as the mae plot.

    images, img_mask: [B, X, Y, Z]. pca_maps: [B, gx, gy, gz, 3] on the patch grid, in [0, 1].
    """
    img_mask = img_mask > 0
    patch_size = images.shape[1] // pca_maps.shape[1]

    # center slice, oriented with superior/anterior up. pca upsampled from patches to voxels
    axis = VIEW_AXES[view] + 1
    index = images.shape[axis] // 2
    images = np.take(images, index, axis=axis)
    img_mask = np.take(img_mask, index, axis=axis)
    pca = np.take(pca_maps, index // patch_size, axis=axis)
    pca = pca.repeat(patch_size, axis=1).repeat(patch_size, axis=2)

    # gray image and pca panels as rgb, black outside the brain
    gray = np.clip((images - vmin) / (vmax - vmin), 0, 1)
    image_panel = np.where(img_mask[..., None], gray[..., None].repeat(3, axis=-1), 0.0)
    pca_panel = np.where(img_mask[..., None], pca, 0.0)

    height, width = images.shape[1:]
    size = max(height, width)
    num_examples = min(len(images), nrow * ncol)

    # roughly one pixel per voxel
    dpi = 100
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(1.05 * ncol * 2 * size / dpi, 1.05 * nrow * size / dpi),
        dpi=dpi,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03, hspace=0.03)

    for ii, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if ii >= num_examples:
            continue

        # two panels padded to square and stacked with no gap
        strip = np.zeros((size, 2 * size, 3), dtype=np.float32)
        for jj, panel in enumerate([image_panel, pca_panel]):
            plane = np.flipud(panel[ii].transpose(1, 0, 2))
            plane_height, plane_width = plane.shape[:2]
            top = (size - plane_height) // 2
            left = jj * size + (size - plane_width) // 2
            strip[top : top + plane_height, left : left + plane_width] = plane

        ax.imshow(strip, interpolation="nearest")
    return fig
