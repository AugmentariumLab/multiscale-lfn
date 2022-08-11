"""Layers for videos."""
import numpy as np
import torch
from torch import nn


class LatentCodeLayer(nn.Module):
    """A layer containing latent codes for each frame."""

    def __init__(self, num_latent_codes: int = 10, max_time: float = 10,
                 latent_code_dim: int = 256):
        """Initializes a latent code layer.

        Args:
            num_latent_codes: Number of latent codes.
            max_time: Time corresponding to the last latent code.
            latent_code_dim: Size of the latent codes.
        """
        super().__init__()
        self.num_latent_codes = num_latent_codes
        self.max_time = max(max_time, 0.01)
        self.latent_code_dim = latent_code_dim
        if num_latent_codes >= 0:
            latent_codes = torch.normal(0, 0.01 / np.sqrt(max(latent_code_dim, 1)),
                                        size=(num_latent_codes, latent_code_dim))
            self.latent_codes = nn.Parameter(latent_codes)

    def interpolate_latent_code(self, t):
        """Interpolate the latent code.

        Args:
            t: Tensor of size N containing one or more times.

        Returns:
            Tensor containing the linearly interpolated latent code.
        """
        t_clipped = torch.clamp(t, min=0, max=self.max_time)
        index = (self.num_latent_codes - 1) * t_clipped / self.max_time
        index_floored = torch.clamp(torch.floor(index), min=0,
                                    max=self.num_latent_codes - 1).long()
        index_ceiled = torch.clamp(torch.ceil(index), min=0,
                                   max=self.num_latent_codes - 1).long()
        index_fraction = index - index_floored
        latent_code_floored = self.latent_codes[index_floored]
        latent_code_ceiled = self.latent_codes[index_ceiled]
        return ((1.0 - index_fraction)[:, None] * latent_code_floored +
                index_floored[:, None] * latent_code_ceiled)

    def forward(self, x) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Inputs as (B, D) tensor with last entry in D vector the time.

        Returns:
            Tensor x with the time replaced with the interpolated latent code.
        """
        if self.num_latent_codes == 0:
            return x[:, :-1]
        times = x[:, -1]
        sampled_latent_codes = self.interpolate_latent_code(times)
        x_with_latents = torch.cat((x[:, :-1], sampled_latent_codes), dim=1)
        return x_with_latents
