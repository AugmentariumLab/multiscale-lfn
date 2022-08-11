# https://github.com/Po-Hsun-Su/pytorch-ssim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    if channel > 4:
        raise ValueError("Too many channels")
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def cropped_ssim(gt_img: torch.Tensor, pred_img: torch.Tensor, alpha_threshold: float = 0.10,
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
    total_ssim = []
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
        total_ssim.append(
            ssim(gt_img_cropped[None, :3], pred_img_cropped[None, :3]))
    return torch.mean(torch.stack(total_ssim))
