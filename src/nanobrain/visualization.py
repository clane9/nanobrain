import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")

VIEW_AXES = {"sag": 0, "cor": 1, "ax": 2}


def plot_feature_pca(
    images: np.ndarray,
    masks: np.ndarray,
    pca_maps: np.ndarray,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> Figure:
    """Center slices of each image next to its feature pca colors, one row per image.

    images, masks: [B, X, Y, Z]. pca_maps: [B, gx, gy, gz, 3] on the patch grid, in [0, 1].
    """
    num_images = len(images)
    fig, axes = plt.subplots(
        num_images, 6, figsize=(12, 2.2 * num_images), squeeze=False, layout="constrained"
    )
    for row in range(num_images):
        for col, (view, axis) in enumerate(VIEW_AXES.items()):
            image = np.take(images[row], images.shape[axis + 1] // 2, axis=axis)
            mask = np.take(masks[row], masks.shape[axis + 1] // 2, axis=axis) > 0
            pca = np.take(pca_maps[row], pca_maps.shape[axis + 1] // 2, axis=axis)
            # patch grid colors stretched over the image extent
            height, width = image.shape
            extent = (0, height, 0, width)
            axes[row, 2 * col].imshow(
                np.where(mask, image, vmin).T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower"
            )
            axes[row, 2 * col + 1].imshow(
                pca.transpose(1, 0, 2), origin="lower", extent=extent, interpolation="nearest"
            )
            if row == 0:
                axes[row, 2 * col].set_title(view, fontsize=9)
                axes[row, 2 * col + 1].set_title(f"{view} pca", fontsize=9)
    for ax in axes.flat:
        ax.set_axis_off()
    return fig
