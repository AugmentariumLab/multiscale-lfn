"""This contains a basic MLP with layernorm and relu"""
import dataclasses

from torch import nn
import torch

from models import mlp
from utils import nerf_utils


@dataclasses.dataclass
class AtlasNetOutputs:
    model_output: torch.Tensor
    uv_map: torch.Tensor


class AtlasNet(nn.Module):

    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 32,
                 intermediate_features: int = 16,
                 atlas_layers: int = 10, atlas_features: int = 256):
        """Initialize a LFN + Atlas network.

        Args:
            in_features: Input features.
            out_features: Output features.
            layers: Mapping layers.
            hidden_features: Mapping hidden features.
            intermediate_features: Intermediate size.
            atlas_layers: Atlas network layers.
            atlas_features: Atlas network features.
        """
        super().__init__()

        self.mapping_network = mlp.MLP(in_features=in_features,
                                       out_features=intermediate_features,
                                       layers=layers,
                                       hidden_features=hidden_features,
                                       use_layernorm=True)
        self.atlas_network = mlp.MLP(in_features=intermediate_features,
                                     out_features=out_features,
                                     layers=atlas_layers,
                                     hidden_features=atlas_features,
                                     use_layernorm=True)

        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.hidden_features = hidden_features
        self.intermediate_features = intermediate_features
        self.atlas_layers = atlas_layers
        self.atlas_features = atlas_features

    def forward(self, inputs: torch.Tensor) -> AtlasNetOutputs:
        uvs = self.mapping_network(inputs)
        result = self.atlas_network(uvs)
        return AtlasNetOutputs(
            model_output=result,
            uv_map=uvs
        )
