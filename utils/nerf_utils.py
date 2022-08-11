import math
from typing import Optional, Tuple

import torch


def meshgrid_xy(
        tensor1: torch.Tensor, tensor2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)
    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2, indexing='ij')
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def add_positional_encoding(tensor, num_encoding_functions=6,
                            include_input=True,
                            log_sampling=True):
    """Applies positional encoding to the input.

    Copied from krrish94/nerf-pytorch.

    Args:
        tensor: Input tensor.
        num_encoding_functions: Number of encoding functions.
        include_input: Whether to include the input tensor in the output.
        log_sampling: Sample frequency bands with log offsets.

    Returns:
        An output tensor with feature size
        input_dim * include_input + 2 * input_dim * num_encoding_functions
    """
    # Special case, for no positional encoding
    if include_input and num_encoding_functions == 0:
        return tensor
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    encoding = [tensor] if include_input else []
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_ray_bundle(
        height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    """Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Copied from krrish94/nerf-pytorch.

    Args:
        height (int): Height of an image (number of pixels).
        width (int): Width of an image (number of pixels).
        focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
        tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
          transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
        ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
          each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
          row index `j` and column index `i`.
        ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
          direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
          passing through the pixel at row index `j` and column index `i`.
    """
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions
