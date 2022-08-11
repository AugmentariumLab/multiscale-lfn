# Lint as: python3
"""A file of helper functions.

This file contains helper functions for working with spherical images.
This includes functions to convert between spherical and cartesian coordinates as well as
a function to create rotation matrices and rotate equirectangular images.
"""

import argparse
import os
import shutil

import numpy as np

from utils.turbo_colormap import TURBO_COLORMAP


def spherical_to_cartesian(theta, phi, r=None):
    """Spherical to cartesian

    Args:
      theta: Azimuthal value or array between 0 and 2*pi.
      phi: Zenith value or array between 0 and pi.
      r:  (Default value = None) Radius.

    Returns:
      nx3 array of cartesian coordinates.

    """
    if type(theta) is list:
        theta = np.array(theta)
    if type(phi) is list:
        phi = np.array(phi)
    if r is None:
        if type(theta) is np.ndarray:
            r = np.broadcast_to(1, theta.shape)
        else:
            r = 1
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    if type(x) is int:
        return np.array([x, y, z])
    return np.stack([x, y, z], axis=-1)


def cartesian_to_spherical(xyz):
    """Spherical to cartesian.

    Args:
      xyz: nx3 array of Cartesian coordinates.

    Returns:
      Spherical coordintes as an nx3 array.

    """
    theta = np.arctan2(xyz[..., 2], xyz[..., 0])
    r = np.linalg.norm(xyz, axis=-1)
    phi = np.arccos(xyz[..., 1] / r)
    return np.stack([theta, phi, r], axis=-1)


def bilinear_interpolate(image, x, y):
    """Applies bilinear interpolation on numpy encoded images.

    Assumes a channel_last format.
    Based on https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python

    Args:
      image: Input image.
      x: x-coordinates.
      x: y-coordinates.

    Returns:
      Interpolated image.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    top_left = image[y0, x0]
    bottom_left = image[y1, x0]
    top_right = image[y0, x1]
    bottom_right = image[y1, x1]

    tl_weight = (x1 - x) * (y1 - y)
    bl_weight = (x1 - x) * (y - y0)
    tr_weight = (x - x0) * (y1 - y)
    br_weight = (x - x0) * (y - y0)

    if len(top_left.shape) > len(tl_weight.shape):
        tl_weight = tl_weight[..., np.newaxis]
        bl_weight = bl_weight[..., np.newaxis]
        tr_weight = tr_weight[..., np.newaxis]
        br_weight = br_weight[..., np.newaxis]

    return tl_weight * top_left + bl_weight * bottom_left + tr_weight * top_right + br_weight * bottom_right


def rotate_equirectangular_image(image, rot_mat):
    """Applies a rotation matrix to an equirectangular image to rotate it.

    Note that the rotation is performed using bilinear interpolation so applying
    this function several times will create a blurry image.
    You should always accumulate rotations.

    Args:
      image: Input erp image.
      rot_mat: 3x3 Rotation matrix.

    Returns:
      Rotated equirectangular image of the same resolution.

    """
    h = image.shape[0]
    w = image.shape[1]

    xx, yy = np.meshgrid(-(np.arange(0, w) + 0.5) * (2 * np.pi / w) - np.pi / 2,
                         (np.arange(0, h) + 0.5) * (np.pi / h))
    xyz = spherical_to_cartesian(xx, yy)
    xyz = xyz @ rot_mat
    sp = cartesian_to_spherical(xyz)[..., :2]
    sp[:, :, 0] = (-(sp[:, :, 0] + np.pi / 2) + 4 * np.pi) % (2 * np.pi)
    sp[:, :, 0] = w * sp[:, :, 0] / (2 * np.pi) - 0.5
    sp[:, :, 0] = (sp[:, :, 0] + w) % w
    sp[:, :, 1] = h * sp[:, :, 1] / np.pi - 0.5
    image_extended = np.concatenate([image, image[:, -1:, :]], axis=1)
    new_image = bilinear_interpolate(image_extended, sp[:, :, 0], sp[:, :, 1])
    return new_image


TAG_FLOAT = 202021.25


def read_flo_file(flo_path):
    """Reads a flo file.

    Based on https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py

    Args:
      flo_path: Path to the .flo file.

    Returns:
      Flow array as [h,w,2]] tensor.

    """
    with open(flo_path, "rb") as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        assert tag == TAG_FLOAT, "flo tag incorrect"
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data, (h, w, 2))
        return flow


def flow_to_image(flow):
    """Draw the optical flow as an rgb image.

    R shows horizontal flow. G shows vertical flow.

    Args:
      flow: Input flow as [h,w,2] array.

    Returns:
      RGB image as [h,w,3] array.

    """
    h = flow.shape[0]
    w = flow.shape[1]
    img = np.zeros((h, w, 3))
    max_val = 2 * np.median(flow)
    # print("Flow max", np.max(flow), "median", np.median(flow))
    img[:, :, 0] = np.abs(flow[:, :, 0]) / max_val
    img[:, :, 1] = np.abs(flow[:, :, 1]) / max_val
    img = np.clip(img, 0, 1)
    return img


def rotate_around_axis(axis, rad):
    """Creates a rotation matrix of angle rad around a specified axis

    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Args:
      axis: Axis around which to rotate.
      rad: Radians to rotate.

    Returns:
      3x3 Rotation matrix

    """
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    iden = np.identity(3)
    rot = iden + np.sin(rad) * k + (1 - np.cos(rad)) * (k @ k)
    return rot


def lerp(a, b, x):
    """Basic linear interpolation function.

    Args:
      a: Left value.
      x: Right value.
      b: Distance between the left and right values.

    Returns:
      Lerped value.

    """
    return (1 - x) * a + x * b


def join_and_make(*args):
    """Joins args using os.path.join and then ensures the directory exists using os.makedirs.

    Args:
      args: Same args as os.path.join
      *args:

    Returns:
      Joined path.

    """
    my_folder = os.path.join(*args)
    os.makedirs(my_folder, exist_ok=True)
    return my_folder


def angle_diff_deg(angle1, angle2):
    """Calculates the difference between two angles in degrees

    Args:
      angle1: First angle.
      angle2: Second angle.

    Returns:
      Signed angle between [-180, 180]

    """
    return (angle1 - angle2 + 180 + 3600) % 360 - 180


def spherical_to_gnomonic(theta, phi, theta_1, phi_1):
    """Performs gnomonic projection.

    Args:
      theta: Longitude (azimuthal angle) in radians.
      phi: Latitude (altitude angle) in radians.
      theta_1: Longitude of reference plane.
      phi_1: Latitude of reference plane.

    Returns:
      xy: in channel_last format

    """
    cos_c = np.sin(phi_1) * np.sin(phi) + np.cos(phi_1) * np.cos(phi) * np.cos(
        theta - theta_1)
    x = np.cos(phi) * np.sin(theta - theta_1) / cos_c
    y = (np.cos(phi_1) * np.sin(phi) -
         np.sin(phi_1) * np.cos(phi) * np.cos(theta - theta_1)) / cos_c
    return np.stack((x, y), axis=-1)


def cubemap_to_spherical(uv, side=0):
    """Converts cubemap coordinates to spherical coordinates.

    Args:
      uv: uv coordinates from 0 to 1.
      side: Side of the cubemap from 0 to 5. 0=top, 5=bottom.

    Returns:
      Spherical coordinates a channel-last tensor.

    """
    up_vec = np.array([0, 1, 0])
    right_vec = np.array([1, 0, 0])
    u = uv[..., 0] * 2 - 1
    v = uv[..., 1] * 2 - 1
    ones = np.ones(u.shape)
    xyz = np.stack((u, v, ones), axis=-1)
    if side == 0:
        rot_mat = rotate_around_axis(right_vec, np.pi / 2)
    elif side == 1:
        rot_mat = rotate_around_axis(right_vec, -np.pi / 2)
    else:
        rot_mat = rotate_around_axis(up_vec, side * np.pi / 2)
    xyz = rot_mat @ xyz[..., np.newaxis]
    xyz = xyz[..., 0]
    sph = cartesian_to_spherical(xyz)
    theta = sph[..., 0]
    phi = np.pi - sph[..., 1]
    return np.stack((theta, phi), axis=-1)


def spherical_to_cubemap(theta, phi):
    """Converts spherical coordinates to cubemap coordinates.

    Args:
      theta: Longitude (azimuthal angle) in radians. [0, 2pi]
      phi: Latitude (altitude angle) in radians. [0, pi]

    Returns:
      uv: UVS in channel_last format
      idx: Side of the cubemap

    """
    u = np.zeros(theta.shape, dtype=np.float32)
    v = np.zeros(theta.shape, dtype=np.float32)
    side = np.zeros(theta.shape, dtype=np.float32)
    side[:] = -1

    for i in range(0, 4):
        indices = np.logical_or(
            np.logical_and(theta >= i * np.pi / 2 - np.pi / 4, theta <=
                           (i + 1) * np.pi / 2 - np.pi / 4),
            np.logical_and(theta >= i * np.pi / 2 - np.pi / 4 + 2 * np.pi, theta <=
                           (i + 1) * np.pi / 2 - np.pi / 4 + 2 * np.pi))
        u[indices] = np.tan(theta[indices] - i * np.pi / 2)
        v[indices] = 1 / (np.tan(phi[indices]) *
                          np.cos(theta[indices] - i * np.pi / 2))
        side[indices] = i + 1
    top_indices = np.logical_or(phi < np.pi / 4, v >= 1)
    u[top_indices] = -np.tan(phi[top_indices]) * np.sin(theta[top_indices] -
                                                        np.pi)
    v[top_indices] = np.tan(phi[top_indices]) * np.cos(theta[top_indices] - np.pi)
    side[top_indices] = 0
    bottom_indices = np.logical_or(phi >= 3 * np.pi / 4, v <= -1)
    u[bottom_indices] = -np.tan(phi[bottom_indices]) * np.sin(
        theta[bottom_indices])
    v[bottom_indices] = -np.tan(phi[bottom_indices]) * np.cos(
        theta[bottom_indices])
    side[bottom_indices] = 5

    assert not np.any(side < 0), "Side less than 0"

    return np.stack(((u + 1) / 2, (-v + 1) / 2), axis=-1), side


def panobasic_im2sphere(fov, sphere_width, sphere_height, im_width, im_height,
                        theta_0, phi_0):
    """This is a python port of im2sphere in Yinda's panobasic.

    Args:
      fov: Field of view
      sphere_w: Width of image
      sphere_h: Height of image
      theta_0: Theta of plane
      phi_0: Phi of plane

    Returns:
      Image coordinates and a valid map
    """
    theta, phi = np.meshgrid(
        (np.arange(sphere_width) - sphere_width / 2 + 0.5) *
        (2 * np.pi / sphere_width),
        -(np.arange(sphere_height) - sphere_height / 2 + 0.5) *
        (np.pi / sphere_height))
    radius = (im_width / 2) / np.tan(fov / 2)

    x0 = radius * np.cos(phi_0) * np.sin(theta_0)
    y0 = radius * np.cos(phi_0) * np.cos(theta_0)
    z0 = radius * np.sin(phi_0)

    alpha = radius * np.cos(phi) * np.sin(theta)
    beta = radius * np.cos(phi) * np.cos(theta)
    gamma = radius * np.sin(phi)

    divisor = x0 * alpha + y0 * beta + z0 * gamma
    x1 = radius * radius * alpha / divisor
    y1 = radius * radius * beta / divisor
    z1 = radius * radius * gamma / divisor

    vec = np.stack((x1 - x0, y1 - y0, z1 - z0), axis=2).reshape((-1, 3))
    vecpos_x = np.array([np.cos(theta_0), -np.sin(theta_0), 0])[np.newaxis, :]
    delta_x = (vecpos_x @ vec.transpose()) / np.sqrt(
        vecpos_x @ vecpos_x.transpose())
    vecpos_y = np.cross(np.array([x0, y0, z0]), vecpos_x)
    delta_y = (vecpos_y @ vec.transpose()) / np.sqrt(
        vecpos_y @ vecpos_y.transpose())

    delta_x = delta_x.reshape((sphere_height, sphere_width)) + (im_width + 1) / 2
    delta_y = delta_y.reshape((sphere_height, sphere_width)) + (im_height + 1) / 2

    return np.stack((delta_x, delta_y), axis=-1), divisor > 0


def depth_to_turbo_colormap(depth, min_depth=None, max_depth=None):
    """Returns the depth colors according to the turbo colormap.

    Args:
      depth: numpy array of size BxHxWx1
      min_depth: minimum depth adjustment (optional)

    Returns:
      numpy array of size BxHxWx3

    """
    batch_size, height, width = depth.shape[:3]
    if np.any(np.isnan(depth)):
        raise ValueError("Nan depth values")
    if min_depth is None:
        normalized_depth = np.divide(1,
                                     depth,
                                     out=np.zeros_like(depth),
                                     where=depth != 0)
        normalized_depth = normalized_depth / np.max(normalized_depth)
        normalized_depth = np.clip(normalized_depth * TURBO_COLORMAP.shape[0], 0,
                                   TURBO_COLORMAP.shape[0] - 1)
    else:
        normalized_depth = min_depth / np.maximum(depth, 1e-5)
        normalized_depth = np.clip(normalized_depth * TURBO_COLORMAP.shape[0], 0,
                                   TURBO_COLORMAP.shape[0] - 1)
    normalized_depth_floor = np.floor(normalized_depth).astype(int)
    normalized_depth_ceil = np.ceil(normalized_depth).astype(int)
    normalized_depth_round = normalized_depth - normalized_depth_floor
    colored_depth = lerp(
        TURBO_COLORMAP[normalized_depth_floor.reshape(-1)],
        TURBO_COLORMAP[normalized_depth_ceil.reshape(-1)],
        np.tile(normalized_depth_round.reshape(-1)[:, np.newaxis], [1, 3]))
    colored_depth = colored_depth.reshape((batch_size, height, width, 3))
    return colored_depth


def str2bool(val):
    """Converts the string value to a bool.

    Args:
      val: string representing true or false
    Returns:
      bool
    """
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rmdir_sync(*inputs):
    """Deletes a directory and its files synchronously.

    Args:
      *inputs: List of folders to join.

    Returns:
      None

    """
    dir_path = os.path.join(*inputs)
    if not os.path.isdir(dir_path):
        return
    shutil.rmtree(dir_path)
    while os.path.isdir(dir_path):
        pass
