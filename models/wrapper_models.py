import torch
from torch import nn

from utils import my_torch_utils
from utils import nerf_utils


class MLPWrapper(nn.Module):
    def __init__(self, model: nn.Module, latent_code_layer: nn.Module, positional_encoding_functions: int = 6):
        super().__init__()
        self.model = model
        self.latent_code_layer = latent_code_layer
        self.positional_encoding_functions = positional_encoding_functions

    def _render_full_image(self, rays: torch.Tensor):
        rays_positional_encoding = nerf_utils.add_positional_encoding(
            rays[:, :-1],
            num_encoding_functions=self.positional_encoding_functions)
        rays_with_positional = torch.cat(
            (rays_positional_encoding, rays[:, -1:]), dim=-1)
        features = self.latent_code_layer(rays_with_positional)
        return self.model(features)

    def forward(self, height, width, focal, pose_target, t):
        ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal, pose_target)
        rays_plucker = my_torch_utils.convert_rays_to_plucker(ray_origins, ray_directions, use_manual_cross=True)
        rays_with_t = torch.cat(
            (rays_plucker, t * torch.ones_like(rays_plucker[:, :, :1])),
            dim=-1)
        return self._render_full_image(rays_with_t.reshape(-1, 7)).reshape((height, width, 3))
