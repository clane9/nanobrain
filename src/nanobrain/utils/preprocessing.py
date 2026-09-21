import math

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nibabel.affines import rescale_affine
from nibabel.processing import resample_to_output, resample_from_to, smooth_image


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


def conform_image(
    img: nib.Nifti1Image,
    min_voxel_size: float = 1.0,
    max_fov: tuple[float, float, float] = (256.0, 256.0, 256.0),
    order: int = 1,
):
    """Conform image to target minimum voxel size and max FOV.

    References:
        https://github.com/nipy/nibabel/blob/5.4.2/nibabel/processing.py#L318
    """
    assert img.ndim == 3, f"expected 3D image, got {img.ndim}"
    img = nib.as_closest_canonical(img)

    # new voxel size no smaller than min voxel size
    voxel_sizes = img.header.get_zooms()
    new_voxel_sizes = [max(min_voxel_size, sz) for sz in voxel_sizes]
    # new fov no bigger than target
    fov = [w * sz for w, sz in zip(img.shape, voxel_sizes)]
    new_fov = [min(w_, w) for w_, w in zip(max_fov, fov)]
    new_shape = [math.ceil(w / sz) for w, sz in zip(new_fov, new_voxel_sizes)]

    # smooth if downsampling
    if any(sz_ > sz for sz_, sz in zip(new_voxel_sizes, voxel_sizes)):
        fwhm = [math.sqrt(sz_**2 - sz**2) for sz_, sz in zip(new_voxel_sizes, voxel_sizes)]
        img = smooth_image(img, fwhm=fwhm)

    # resample with cropping
    new_affine = rescale_affine(img.affine, img.shape, new_voxel_sizes, new_shape)
    new_img = resample_from_to(img, (new_shape, new_affine), order=order)
    return new_img
