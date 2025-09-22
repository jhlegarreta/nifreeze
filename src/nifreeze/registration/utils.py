# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Utilities to aid in performing and evaluating image registration.

This module provides functions to compute displacements of image coordinates
under a transformation, useful for assessing the accuracy of image registration
processes.

"""

from __future__ import annotations

from itertools import product
from typing import Tuple

import nibabel as nb
import nitransforms as nt
import numpy as np

RADIUS = 50.0
"""Typical radius (in mm) of a sphere mimicking the size of a typical human brain."""


def displacements_within_mask(
    mask_img: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.BaseTransform,
    reference_xfm: nt.base.BaseTransform | None = None,
) -> np.ndarray:
    """
    Compute the distance between voxel coordinates mapped through two transforms.

    Parameters
    ----------
    mask_img : :obj:`~nibabel.spatialimages.SpatialImage`
        A mask image that defines the region of interest. Voxel coordinates
        within the mask are transformed.
    test_xfm : :obj:`~nitransforms.base.BaseTransform`
        The transformation to test. This transformation is applied to the
        voxel coordinates.
    reference_xfm : :obj:`~nitransforms.base.BaseTransform`, optional
        A reference transformation to compare with. If ``None``, the identity
        transformation is assumed (no transformation).

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of displacements (in mm) for each voxel within the mask.

    """
    # Mask data as boolean (True for voxels inside the mask)
    maskdata = np.asanyarray(mask_img.dataobj) > 0
    # Convert voxel coordinates to world coordinates using affine transform
    xyz = nb.affines.apply_affine(
        mask_img.affine,
        np.argwhere(maskdata),
    )
    # Apply the test transformation
    targets = test_xfm.map(xyz)

    # Compute the difference (displacement) between the test and reference transformations
    diffs = targets - xyz if reference_xfm is None else targets - reference_xfm.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


def compute_fd_from_transform(
    img: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.BaseTransform,
    radius: float = RADIUS,
) -> float:
    """
    Compute the framewise displacement (FD) for a given transformation.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        The reference image. Used to extract the center coordinates.
    test_xfm : :obj:`~nitransforms.base.BaseTransform`
        The transformation to test. Applied to coordinates around the image center.
    radius : :obj:`float`, optional
        The radius (in mm) of the spherical neighborhood around the center of the image.

    Returns
    -------
    :obj:`float`
        The average framewise displacement (FD) for the test transformation.

    """
    affine = img.affine
    # Compute the center of the image in voxel space
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    # Convert to world coordinates
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    # Generate coordinates of points at radius distance from center
    fd_coords = np.array(list(product(*((radius, -radius),) * 3))) + center_xyz
    # Compute the average displacement from the test transformation
    return np.mean(np.linalg.norm(test_xfm.map(fd_coords) - fd_coords, axis=-1))


def compute_fd_from_motion(motion_parameters: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """Compute framewise displacement (FD) from motion parameters.

    Each row in the motion parameters represents one frame, and columns
    represent each coordinate axis ``x``, `y``, and ``z``. Translation
    parameters are followed by rotation parameters column-wise.

    Parameters
    ----------
    motion_parameters : :obj:`numpy.ndarray`
        Motion parameters.
    radius : :obj:`float`, optional
        Radius (in mm) of a sphere mimicking the size of a typical human brain.

    Returns
    -------
    :obj:`numpy.ndarray`
        The framewise displacement (FD) as the sum of absolute differences
        between consecutive frames.
    """

    translations = motion_parameters[:, :3]
    rotations_deg = motion_parameters[:, 3:]
    rotations_rad = np.deg2rad(rotations_deg)

    # Compute differences between consecutive frames
    d_translations = np.vstack([np.zeros((1, 3)), np.diff(translations, axis=0)])
    d_rotations = np.vstack([np.zeros((1, 3)), np.diff(rotations_rad, axis=0)])

    # Convert rotations from radians to displacement on a sphere
    rotation_displacement = d_rotations * radius

    # Compute FD as sum of absolute differences
    return np.sum(np.abs(d_translations) + np.abs(rotation_displacement), axis=1)


def extract_motion_parameters(affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation (mm) and rotation (degrees) parameters from an affine matrix.

    Parameters
    ----------
    affine : :obj:`~numpy.ndarray`
        The affine transformation matrix.

    Returns
    -------
    :obj:`tuple`
        Extracted translation and rotation parameters.
    """

    translation = affine[:3, 3]
    rotation_rad = np.arctan2(
        [affine[2, 1], affine[0, 2], affine[1, 0]], [affine[2, 2], affine[0, 0], affine[1, 1]]
    )
    rotation_deg = np.rad2deg(rotation_rad)
    return *translation, *rotation_deg
