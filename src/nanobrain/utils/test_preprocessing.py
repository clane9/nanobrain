import pytest
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nibabel.processing import resample_to_output
from templateflow import api as tflow

import nanobrain.utils.preprocessing as preproc


@pytest.fixture(scope="module")
def test_img() -> nib.Nifti1Image:
    client = tflow.TemplateFlowClient()
    path = client.get("MNI152NLin6Asym", desc=None, resolution=1, suffix="T1w", extension="nii.gz")
    img = nib.load(path)
    return img


def test_threshold_mask(test_img: nib.Nifti1Image):
    mask_img = preproc.threshold_mask(test_img)
    assert mask_img.shape == test_img.shape
    assert (mask_img.affine == test_img.affine).all()

    mask = np.asarray(mask_img.dataobj) > 0
    label, count = ndi.label(mask)
    assert count == 1


def test_conform_image(test_img: nib.Nifti1Image):
    test_img = resample_to_output(test_img, (0.5, 1.0, 3.0), order=1)
    mvs = 1.0
    max_fov = (208.0, 240.0, 120.0)
    mask_img = preproc.threshold_mask(test_img)
    fit_img = preproc.conform_image(test_img, mask_img, min_voxel_size=mvs, max_fov=max_fov)
    spacing = fit_img.header.get_zooms()
    fov = [w * sz for w, sz in zip(fit_img.shape, spacing)]
    assert all(sz >= mvs for sz in spacing)
    assert all(w <= w_ for w, w_ in zip(fov, max_fov))
