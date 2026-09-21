import math

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nibabel.processing import resample_to_output, resample_from_to, smooth_image


def threshold_mask(
    img: nib.Nifti1Image,
    resolution: float | None = 2.0,
    sigma: float | None = 4.0,
    remove_islands: bool = True,
    fill_holes: bool = True,
) -> nib.Nifti1Image:
    """Threshold based head masking.

    Args:
        resolution: target resolution to compute the mask at.
        sigma: gaussian smoothing sigma in mm.
        remove_islands: keep only largest connected component.
        fill_holes: fill holes enclosed by the mask.

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
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

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
    mask_img: nib.Nifti1Image,
    min_voxel_size: float = 1.0,
    max_fov: tuple[float, float, float] = (208.0, 240.0, 208.0),
    top_margin: float = 5.0,
    order: int = 1,
):
    """Conform image to target minimum voxel size and max FOV.

    The z crop is taken from the top of the head mask (plus top_margin mm) down.

    References:
        https://github.com/nipy/nibabel/blob/5.4.2/nibabel/processing.py#L318
    """
    assert img.ndim == 3, f"expected 3D image, got {img.ndim}"
    img = nib.as_closest_canonical(img)
    mask_img = nib.as_closest_canonical(mask_img)

    voxel_sizes = img.header.get_zooms()
    new_voxel_sizes = [max(min_voxel_size, sz) for sz in voxel_sizes]
    fov = [w * sz for w, sz in zip(img.shape, voxel_sizes)]
    new_fov = [min(w_, w) for w_, w in zip(max_fov, fov)]
    new_shape = [math.ceil(w / sz) for w, sz in zip(new_fov, new_voxel_sizes)]

    if any(sz_ > sz for sz_, sz in zip(new_voxel_sizes, voxel_sizes)):
        fwhm = [math.sqrt(sz_**2 - sz**2) for sz_, sz in zip(new_voxel_sizes, voxel_sizes)]
        img = smooth_image(img, fwhm=fwhm)

    # update center to be relative to the top of the mask to leave out more neck
    centroid = (np.array(img.shape) - 1) // 2
    mask = np.asarray(mask_img.dataobj) > 0
    mask_z_ids = np.nonzero(mask.any(axis=(0, 1)))[0]
    top = mask_z_ids.max() + top_margin / voxel_sizes[2]
    top = min(top, img.shape[2])
    half_height = new_fov[2] / voxel_sizes[2] / 2
    centroid[2] = round(top - half_height)

    # resample with cropping
    new_affine = rescale_affine(
        img.affine, img.shape, new_voxel_sizes, new_shape=new_shape, centroid=centroid
    )
    new_img = resample_from_to(img, (new_shape, new_affine), order=order)
    return new_img


def rescale_affine(
    affine: np.ndarray,
    shape: tuple[int, int, int],
    zooms: tuple[float, float, float],
    new_shape: tuple[int, int, int] | None = None,
    centroid: tuple[int, int, int] | None = None,
):
    """Return a new affine matrix with updated voxel sizes.

    Accepts a new target shape (defaulting to input shape) and source centroid
    (defaulting to input voxel grid center).

    Reference:
        nibabel.affines.rescale_affine
    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)
    centroid = np.array(centroid if centroid is not None else (shape - 1) // 2)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, centroid)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return nib.affines.from_matvec(rzs_out, t_out)
