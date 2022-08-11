"""This module contains the adaptive mlp model."""
import dataclasses
from typing import Optional

import torch
from torch import nn

from protos_compiled import model_pb2


class LinearBlock(nn.Module):
    """Linear-norm-relu block"""

    def __init__(self, in_features: int = 256, out_features: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LayerNorm(out_features),
            nn.ReLU()
        )

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs: Input tensor.

        Returns:
            A tuple containing the outputs before norm and relu and the outputs after norm and relu.
        """
        return self.model(inputs)


@dataclasses.dataclass
class AdaptiveResnetOutputs:
    """ Outputs for the AdaptiveMLP.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class AdaptiveResnet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 output_every: int = 3, layers: int = 10, hidden_features: int = 256,
                 use_layernorm: bool = True):
        """Initialize an adaptive resnet.

        Args:
            in_features: Input features.
            out_features: Output features.
            output_every: Skip block size.
            layers: Number of layers.
            hidden_features: Number of hidden features.
        """
        super().__init__()
        if not use_layernorm:
            raise NotImplementedError("No layernorm not implemented")
        model_layers = [
            LinearBlock(in_features=in_features, out_features=hidden_features)
        ]
        for i in range(layers - 2):
            model_layers.append(LinearBlock(in_features=hidden_features,
                                            out_features=hidden_features))
        self.model_layers = nn.ModuleList(model_layers)
        self.final_layer = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.output_every = output_every
        self.num_outputs = (layers + output_every - 2) // output_every

    def forward(self, inputs, lods=None) -> AdaptiveResnetOutputs:
        """Forward pass.

        Args:
            inputs: Input tensor.
            lods: Which lods to output.

        Returns:
            Object containing the expected output and the final output.
        """
        outputs = []
        current_output = inputs
        if lods is None:
            lods = list(range(self.num_outputs))
        sorted_lods = sorted(lods, reverse=True)
        for i, layer in enumerate(self.model_layers):
            current_output = layer(current_output)
            current_lod = i // self.output_every
            if ((i + 1) % self.output_every == 0 and i != len(self.model_layers) - 1 and
                    sorted_lods and sorted_lods[-1] == current_lod):
                outputs.append(self.final_layer(current_output))
                sorted_lods.pop()
            if not sorted_lods:
                break
        current_lod = (len(self.model_layers) - 1) // self.output_every
        if sorted_lods and sorted_lods[-1] == current_lod:
            outputs.append(self.final_layer(current_output))
        outputs = torch.stack(outputs, dim=1)
        return AdaptiveResnetOutputs(outputs=outputs)

    def num_params(self, lod: int):
        total_params = 0
        for i, layer in enumerate(self.model_layers):
            total_params += (layer.model[0].weight.numel() + layer.model[0].bias.numel() +
                             layer.model[1].weight.numel() + layer.model[1].bias.numel())
            current_lod = i // self.output_every
            if ((i + 1) % self.output_every == 0 and i != len(self.model_layers) - 1 and
                    lod == current_lod):
                total_params += self.final_layer.weight.numel() + self.final_layer.bias.numel()
                return total_params
        current_lod = (len(self.model_layers) - 1) // self.output_every
        if lod == current_lod:
            total_params += self.final_layer.weight.numel() + self.final_layer.bias.numel()
        return total_params

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto representation.

        Returns:
            Proto containing the network weights.
        """
        network = model_pb2.Model.Network()
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            v_np = v.cpu().numpy()
            network.state_dict[k].shape.extend(v_np.shape)
            network.state_dict[k].values.extend(v_np.ravel())
        network.resnet.in_features = self.in_features
        network.resnet.out_features = self.out_features
        network.resnet.layers = self.layers
        network.resnet.hidden_features = self.hidden_features
        network.resnet.use_layernorm = self.use_layernorm
        network.resnet.output_every = self.output_every
        return network
