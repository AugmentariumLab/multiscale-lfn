from typing import Optional

import torch

from utils import my_torch_utils


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0,
         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes the PSNR.

    Args:
        img1: Image
        img2: Image
        max_val: Maximum possible value.
        mask: Mask of pixels.

    Returns:
        PSNR value as torch tensor.
    """
    if mask is not None:
        filtered_mask = (mask > 0.0001).float()
        channels = img1.shape[1]
        if channels > 4:
            raise ValueError("Too many channels")
        per_pixel_error = ((img1 - img2) * filtered_mask) ** 2
        per_channel_error = torch.sum(per_pixel_error, dim=(
            2, 3)) / torch.sum(filtered_mask, dim=(2, 3))
        mse = torch.mean(per_channel_error)
    else:
        mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(max_val * max_val / mse)


def cropped_psnr(gt_img: torch.Tensor, pred_img: torch.Tensor, alpha_threshold: float = 0.10,
                 percent_pixels: float = 0.01) -> torch.Tensor:
    """Crops the image to only non-transparent portions and computes the PSNR of RGB channels.

    Args:
        gt_img: GT image as (B, C, H, W).
        pred_img: Pred image.
        alpha_threshold: Alpha value to use for cropping.
        percent_pixels: Percentage of pixels above alpha to keep the row.

    Returns:
        The psnr value.
    """
    batch_size, channels, _, _ = gt_img.shape
    if channels < 4:
        raise ValueError("Alpha channel not provided")
    total_psnr = []
    for i in range(batch_size):
        alpha_ok = gt_img[i, 3, :, :] > alpha_threshold
        alpha_ok_h = (torch.sum(alpha_ok, dim=1) /
                      alpha_ok.shape[1] > percent_pixels).nonzero()
        alpha_ok_w = (torch.sum(alpha_ok, dim=0) /
                      alpha_ok.shape[0] > percent_pixels).nonzero()
        alpha_ok_h_min = torch.min(alpha_ok_h)
        alpha_ok_h_max = torch.max(alpha_ok_h) + 1
        alpha_ok_w_min = torch.min(alpha_ok_w)
        alpha_ok_w_max = torch.max(alpha_ok_w) + 1
        gt_img_cropped = gt_img[i, :, alpha_ok_h_min:alpha_ok_h_max,
                                alpha_ok_w_min:alpha_ok_w_max]
        pred_img_cropped = pred_img[i, :,
                                    alpha_ok_h_min:alpha_ok_h_max, alpha_ok_w_min:alpha_ok_w_max]
        total_psnr.append(
            psnr(gt_img_cropped[None, :3], pred_img_cropped[None, :3]))
    return torch.mean(torch.stack(total_psnr))
