import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nibabel.processing import resample_to_output, resample_from_to


def threshold_mask(
    img: nib.Nifti1Image,
    resolution: float | None = 2.0,
    sigma: float | None = 4.0,
    remove_islands: bool = True,
) -> nib.Nifti1Image:
    """Threshold based head masking.

    Args:
        resolution: target resolution to compute the mask at.
        sigma: gaussian smoothing sigma in mm.
        remove_islands: keep only largest connected component.

    Returns:
        mask nibabel image

    Notes:
        The recipe of smoothing + mean threshold seemed pretty robust on OpenNeuro test
        images. Also tried median filtering instead of gaussian, otsu/minimum threshold
        instead of mean, which weren't any better.

    References:
        https://github.com/dipy/dipy/blob/1.12.1/dipy/segment/mask.py#L132
    """
    orig_img = img

    if resolution is not None:
        voxel_sizes = 3 * (resolution,)
        img = resample_to_output(img, voxel_sizes=voxel_sizes, order=1)

    data = img.get_fdata(dtype=np.float32)

    # filter to remove high frequency noise
    if sigma is not None:
        spacing = np.array(img.header.get_zooms()[:3])
        voxel_sigmas = sigma / spacing
        data = ndi.gaussian_filter(data, sigma=voxel_sigmas)

    # threshold, simple mean works fine, esp after smoothing
    mask = data > data.mean()

    # morphology cleanup
    if remove_islands:
        mask = largest_component(mask)

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
    mask_img = resample_from_to(mask_img, orig_img, order=0)
    return mask_img


def largest_component(mask: np.ndarray):
    label, count = ndi.label(mask)
    if count > 1:
        sizes = ndi.sum_labels(mask, label, range(1, count + 1))
        mask = label == (np.argmax(sizes) + 1)
    return mask
