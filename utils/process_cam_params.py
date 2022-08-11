import json

import torch
import numpy as np
from PIL import Image

DATA_DIR = "datasets/StudioV2CaptureTest0921NormalizedColor/"


def create_meshgrid(height, width, normalized_coordinates=True, device=torch.device('cpu'), dtype=torch.float32):
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys], indexing='ij')).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def get_ray_directions(H, W, focal_W, focal_H, p_W, p_H):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i - p_W) / focal_W, (j - p_H) / focal_H, torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_rays_world(directions, R):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ R.T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_d


f = open(f'{DATA_DIR}/cameraParametersColmapAlignedNew.txt')
cam_params = json.load(f)

R, T, foc, pos, p_p = [], [], [], [], []
for idx in range(240):
    half_x_len, half_y_len = cam_params[idx]["intrinsics"][8], cam_params[idx]["intrinsics"][9]
    foc_x, foc_y = cam_params[idx]["intrinsics"][0], cam_params[idx]["intrinsics"][5]
    E = np.transpose(np.asarray(cam_params[idx]["transform"]).reshape(1, 4, 4), (0, 2, 1))
    _R, _T = np.transpose(E[:, :3, :3], (0, 2, 1)), E[:, :3, 3]
    _pos = -_R[0] @ _T.reshape(3, 1)
    _at = _R[0].T @ np.asarray([0, 0, 1])
    R.append(_R)
    T.append(_T)
    foc.append([foc_x, foc_y])
    pos.append(_pos)
    p_p.append([half_x_len, half_y_len])

for i in range(len(R)):
    R_i = R[i]
    foc_i = foc[i]
    pos_i = pos[i]
    p_i = p_p[i]
    cam_id = i + 1

    im_path = f'{DATA_DIR}/Extracted/Despilled_f_0_{cam_id}_extracted.png'
    mask_path = f'{DATA_DIR}/Mask/f_0_{cam_id}_mask.jpg'

    _rgb_im = (np.asarray(Image.open(im_path).convert('RGB')).astype(np.uint8))
    mask = np.asarray(Image.open(mask_path).convert('RGB')) / 255.
    rgb_im = (_rgb_im * mask).astype(np.uint8)

    image_height, image_width = rgb_im.shape[0], rgb_im.shape[1]

    _ray_dir = get_ray_directions(H=image_height, W=image_width,
                                  focal_W=foc_i[0], focal_H=foc_i[1],
                                  p_W=p_i[0], p_H=p_i[1])

    ray_dir = get_rays_world(_ray_dir, R_i.squeeze())
    # ray_dir = ray_dir.view(-1, 3)
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
    ray_dir = ray_dir.numpy()
