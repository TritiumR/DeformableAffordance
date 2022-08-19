#!/usr/bin/env python

import sys
import time
import struct
import threading

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import transformations

#-----------------------------------------------------------------------------
# MATH UTILS
#-----------------------------------------------------------------------------

def sample_distribution(prob, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = np.ndarray.flatten(prob) / np.sum(prob)
    rand_ind = np.random.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())

#-------------------------------------------------------------------------
# Transformation Helper Functions
#-------------------------------------------------------------------------



def get_pybullet_quaternion_from_rot(rotation):
    """Abstraction for converting from a 3-parameter rotation to quaterion.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      rotation: a 3-parameter rotation, in xyz order tuple of 3 floats
    Returns:
      quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = transformations.quaternion_from_euler(*euler_zxy, axes='szxy')
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw


def get_rot_from_pybullet_quaternion(quaternion_xyzw):
    """Abstraction for converting from quaternion to a 3-parameter toation.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      quaternion, in xyzw order, tuple of 4 floats
    Returns:
      rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
    """
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = transformations.euler_from_quaternion(quaternion_wxyz, axes='szxy')
    euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
    return euler_xyz


def apply_transform(transform_to_from, points_from):
  r"""Transforms points (3D) into new frame.

  Using transform_to_from notation.

  Args:
    transform_to_from: numpy.ndarray of shape [B,4,4], SE3
    points_from: numpy.ndarray of shape [B,3,N]

  Returns:
    points_to: numpy.ndarray of shape [B,3,N]
  """
  num_points = points_from.shape[-1]

  # non-batched
  if len(transform_to_from.shape) == 2:
    ones = np.ones((1, num_points))

    # makes these each into homogenous vectors
    points_from = np.vstack((points_from, ones))  # [4,N]
    points_to = transform_to_from @ points_from  # [4,N]
    return points_to[0:3, :]  # [3,N]

  # batched
  else:
    assert len(transform_to_from.shape) == 3
    batch_size = transform_to_from.shape[0]
    zeros = np.ones((batch_size, 1, num_points))
    points_from = np.concatenate((points_from, zeros), axis=1)
    assert points_from.shape[1] == 4
    points_to = transform_to_from @ points_from
    return points_to[:, 0:3, :]

#-----------------------------------------------------------------------------
# IMAGE UTILS
#-----------------------------------------------------------------------------


def get_image_transform(theta, trans, pivot=[0, 0]):
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_T_image = np.array([[1., 0., -pivot[0]],
                              [0., 1., -pivot[1]],
                              [0., 0.,        1.]])
    image_T_pivot = np.array([[1., 0., pivot[0]],
                              [0., 1., pivot[1]],
                              [0., 0.,       1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]],
                          [0.,            0.,            1.]])
    return np.dot(image_T_pivot, np.dot(transform, pivot_T_image))


def check_transform(image, pixel, transform):
    """Valid transform only if pixel locations are still in FoV after transform."""
    new_pixel = np.flip(np.int32(np.round(np.dot(transform, np.float32(
        [pixel[1], pixel[0], 1.]).reshape(3, 1))))[:2].squeeze())
    valid = np.all(new_pixel >= 0) and new_pixel[0] < image.shape[
        0] and new_pixel[1] < image.shape[1]
    return valid, new_pixel


def get_se3_from_image_transform(theta, trans, pivot, heightmap, bounds, pixel_size):
    position_center = pixel_to_position(np.flip(np.int32(np.round(pivot))),
        heightmap, bounds, pixel_size, skip_height=False)
    new_position_center = pixel_to_position(np.flip(np.int32(np.round(pivot + trans))),
        heightmap, bounds, pixel_size, skip_height=True)
    # Don't look up the z height, it might get augmented out of frame
    new_position_center = (new_position_center[0], new_position_center[1], position_center[2])

    delta_position = np.array(new_position_center) - np.array(position_center)

    t_world_center = np.eye(4)
    t_world_center[0:3, 3] = np.array(position_center)

    t_centernew_center = np.eye(4)
    euler_zxy = (-theta, 0, 0)
    t_centernew_center[0:3, 0:3] = transformations.euler_matrix(*euler_zxy, axes='szxy')[0:3, 0:3]

    t_centernew_center_Tonly = np.eye(4)
    t_centernew_center_Tonly[0:3, 3] = - delta_position
    t_centernew_center = t_centernew_center @ t_centernew_center_Tonly

    t_world_centernew = t_world_center @ np.linalg.inv(t_centernew_center)
    return t_world_center, t_world_centernew


def get_random_image_transform_params(image_size):
    theta_sigma = 2 * np.pi / 6
    theta = np.random.normal(0, theta_sigma)

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


# -------------------------------------------------------------------------------------- #
# this is slightly different in their updated code, where they return rounded
# pixels. After carefully checking their updated code, their `rounded_pixels` corresponds
# to the OLD way of creating `pixels`. We originally returned `input_image, new_pixels`,
# and the new pixels were already rounded. Later, they must have wanted to return the
# non-rounded versions, so they re-interpret `pixels` to be the non-rounded version, and
# added `rounded_pixels` to make the distinction more explicit.
# -------------------------------------------------------------------------------------- #
# Their code also calls `theta, trans, pivot = get_random_image_transform_params`. We use
# the EXACT code; I assume they wanted to call it from `agents/gt_state.py`, because that
# agent should use the same transformation, except it doesn't use an input image. Thus, it
# just needs the parameters so that it can change all the ground-truth poses.
# -------------------------------------------------------------------------------------- #
# Code gets called from agents/{conv_mlp,form2fit,transporter}.py. We only have transporter
# here, and in all cases, set_theta_zero=False by default so that's fine. Finally, consider
# return values. Their transporter.py uses `transform_params` but ONLY for the 6 DoF agent,
# thus we don't need to return it. Their code also doesn't even return the non-rounded
# pixels, so again it should be safe to ignore. :)
# -------------------------------------------------------------------------------------- #

def perturb(input_image, pixels, set_theta_zero=False):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]
    i = 0
    # Compute random rigid transform.
    while True:
        i += 1
        theta, trans, pivot = get_random_image_transform_params(image_size)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)
            pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            pixel = np.flip(pixel)
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]
            is_valid = is_valid and np.all(pixel >= 0) and in_fov
            new_pixels.append(pixel)
        if is_valid:
            break
        if i > 100:
            print('out of range')
            return None, None

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(input_image, transform[:2, :],
                                 (image_size[1], image_size[0]),
                                 flags=cv2.INTER_NEAREST)
    return input_image, new_pixels
