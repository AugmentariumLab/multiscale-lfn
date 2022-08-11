"""This is a variant of mipnet where each layer's weights are only partially used so we get all LoD in one forward."""
import dataclasses
from typing import Tuple, Sequence, Optional

import torch
from torch import nn
import numpy as np

from protos_compiled import model_pb2


@dataclasses.dataclass
class SMipNetOutputs:
    """Outputs for the MipNet.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class SMipNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 factors: Optional[Sequence[Tuple[float, float]]] = None):
        """Initialize a mipnet.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers:
            hidden_features:
            use_layernorm:
            factors:
        """
        super().__init__()
        if layers == 1:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=out_features)]
        else:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=hidden_features),
                            nn.ReLU()]
            for i in range(layers - 2):
                model_layers.append(nn.Linear(in_features=hidden_features,
                                              out_features=hidden_features))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.factors = [0.25, 0.5, 0.75, 1.0] if factors is None else [x[0] for x in factors]
        self.num_outputs = len(self.factors)

    def _forward_linear(self, layer: nn.Linear, inputs: torch.Tensor,
                        factors: Tuple[float, float]) -> torch.Tensor:
        """Forward through a linear layer.

        Args:
            layer: The linear layer.
            inputs: Inputs.
            factors: Factor to subset the layer.

        Returns:
            The output of the forward pass.
        """
        subset_shapes = (int(round(layer.weight.shape[0] * factors[0])), int(round(layer.weight.shape[1] * factors[1])))
        m_weight = layer.weight[:subset_shapes[0], :subset_shapes[1]]
        m_bias = layer.bias[:subset_shapes[0]]
        return inputs @ m_weight.T + m_bias[None]

    def _forward_linear2(self, layer: nn.Linear, inputs: torch.Tensor,
                         factors: Sequence[float]) -> torch.Tensor:
        """Forward through a linear layer.

        Args:
            layer: The linear layer.
            inputs: Inputs.
            factors: Factor to subset the layer.
            subfactors: Factor for frozen weights.

        Returns:
            The output of the forward pass.
        """
        weight = layer.weight
        bias = layer.bias
        out_vecs = []
        for i, my_factor in enumerate(factors):
            my_subfactor = 0 if i == 0 else factors[i - 1]
            subset_shapes = (int(round(weight.shape[0] * my_factor)), int(round(weight.shape[1] * my_factor)))
            subset_shapes2 = (int(round(weight.shape[0] * my_subfactor)), int(round(weight.shape[1] * my_subfactor)))
            m_weight = weight[subset_shapes2[0]:subset_shapes[0], :subset_shapes[1]]
            m_bias = bias[subset_shapes2[0]:subset_shapes[0]]
            out_vecs.append(inputs[:, :subset_shapes[1]] @ m_weight.T + m_bias[None])
        return torch.cat(out_vecs, dim=1)

    def forward(self, inputs, factors: Optional[Sequence[float]] = None) -> SMipNetOutputs:
        if factors is None:
            factors = self.factors
        max_factor = np.max(factors)
        x = self._forward_linear(self.model[0], inputs, (max_factor, 1))
        x = self.model[1](x)
        for i in range(1, self.layers - 1):
            x = self._forward_linear2(self.model[2 * i], x, factors)
            x = self.model[2 * i + 1](x)
        final_features = x
        model_outputs = []
        for factor in factors:
            v_length = int(round(self.model[-1].weight.shape[1] * factor))
            x = self._forward_linear(self.model[-1], final_features[:, :v_length], (1, factor))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return SMipNetOutputs(outputs=model_outputs)

    def _numel_linear(self, layer: nn.Linear, factors: Tuple[float, float]) -> int:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return subweight.numel() + subbias.numel()

    def num_params(self, lod: int):
        factor = lod if isinstance(lod, Tuple) else self.factors[lod]
        total_params = 0
        total_params += self._numel_linear(self.model[0], (factor, 1))
        for i in range(1, self.layers - 1):
            total_params += self._numel_linear(self.model[2 * i], (factor, factor))
        total_params += self._numel_linear(self.model[-1], (1, factor))
        return total_params

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto containing weights.

        Returns:
            The proto.
        """
        network = model_pb2.Model.Network()
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            v_np = v.cpu().numpy()
            network.state_dict[k].shape.extend(v_np.shape)
            network.state_dict[k].values.extend(v_np.ravel())
        exit(0)
