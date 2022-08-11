"""The idea is to learn to bake feature values into two 360 ERP images."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import my_utils

lerp = my_utils.lerp


def nan_to_num(a: torch.Tensor, val: float = 0.0):
    return torch.where(torch.isfinite(a), a, torch.zeros_like(a) + val)


def solve_quadratic(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Solution to quadratic formula."""
    return (-b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)


def grid_sample(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Custom bilinear grid sample"""
    height, width, channels = image.shape
    x = x * (width - 1)
    y = y * (height - 1)
    x_floor = torch.floor(x).long()
    x_ceil = torch.ceil(x).long()
    x_frac = x - x_floor
    y_floor = torch.floor(y).long()
    y_ceil = torch.ceil(y).long()
    y_frac = y - y_floor
    image1 = lerp(image[y_floor, x_floor], image[y_floor, x_ceil], x_frac[:, None])
    image2 = lerp(image[y_ceil, x_floor], image[y_ceil, x_ceil], x_frac[:, None])
    return lerp(image1, image2, y_frac[:, None])


class LearnedRayModel(nn.Module):
    def __init__(self, ray_features: int = 4, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 32,
                 use_layernorm: bool = True, width: int = 512, height: int = 512,
                 sphere_radius=None):
        """Initialize an MLP.

        Args:
            ray_features: Channels per sphere.
            out_features: Number of output features.
            layers: Number of layers.
            hidden_features: Number of hidden features.
            use_layernorm: Whether to include layer norm.
        """
        super().__init__()

        if sphere_radius is None:
            sphere_radius = np.arange(1, 2, 4)

        self.sphere_features = nn.Parameter(
            torch.rand((len(sphere_radius), height, width, ray_features), dtype=torch.float32))

        model_layers = [nn.Linear(in_features=len(sphere_radius) * ray_features,
                                  out_features=hidden_features)]
        if use_layernorm:
            model_layers.append(nn.LayerNorm(hidden_features))
        model_layers.append(nn.ReLU())
        for i in range(layers - 2):
            model_layers.append(nn.Linear(in_features=hidden_features,
                                          out_features=hidden_features))
            if use_layernorm:
                model_layers.append(nn.LayerNorm(hidden_features))
            model_layers.append(nn.ReLU())
        model_layers.append(
            nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        self.out_features = out_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.sphere_radius = sphere_radius

    def ray_to_indices(self, ray_origins, ray_directions):
        """Convert rays to input indices.

        Args:
            ray_origins: Ray origins as (B, 3) tensor
            ray_directions: Ray directions as (B, 3) tensor

        Returns:
            Tensor as (B, 4).
        """
        theta_phis = []
        for radius in self.sphere_radius:
            ro_norm_sq = ray_origins.pow(2).sum(-1)
            rd_norm_sq = ray_directions.pow(2).sum(-1)
            t1 = solve_quadratic(
                rd_norm_sq,
                (2 * ray_origins * ray_directions).sum(-1),
                ro_norm_sq - radius * radius
            )
            outer_points = ray_origins + t1[..., None] * ray_directions
            r = torch.sqrt(outer_points.pow(2).sum(-1))
            outer_theta = torch.atan2(outer_points[..., 2], outer_points[..., 0])
            outer_theta = torch.fmod(outer_theta + 2 * np.pi, 2 * np.pi)
            outer_phi = torch.acos(outer_points[..., 1] / r)
            theta_phis.append(outer_theta)
            theta_phis.append(outer_phi)
        return torch.stack(theta_phis, dim=-1)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs: (B, 4) tensor of spherical coordinates.

        Returns:
            Features as (B, out_features) tensor.
        """
        features = []
        for i in range(len(self.sphere_radius)):
            theta_1 = inputs[:, 2 * i] / (2 * np.pi)
            phi_1 = inputs[:, 2 * i + 1] / np.pi
            features_1 = grid_sample(self.sphere_features[i],
                                     nan_to_num(theta_1),
                                     nan_to_num(phi_1))
            features_1 = torch.where(
                torch.logical_and(torch.isfinite(theta_1), torch.isfinite(phi_1))[:, None], features_1,
                torch.zeros_like(features_1))
            features.append(features_1)
        x = torch.cat(features, dim=-1)
        return self.model(x)
