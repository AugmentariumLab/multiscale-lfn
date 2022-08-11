"""This is a network with multiple sized layers."""
import dataclasses
from typing import Tuple, Sequence, Optional

import torch
from torch import nn

from protos_compiled import model_pb2


@dataclasses.dataclass
class MipNetOutputs:
    """Outputs for the MipNet.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class MipNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 factors: Optional[Sequence[Tuple[float, float]]] = None,
                 share_gradients: bool = False):
        """Initialize a mipnet.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers:
            hidden_features:
            use_layernorm:
            factors:
            share_gradients: Whether gradients should propagate across LoD weights.
        """
        super().__init__()
        if layers == 1:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=out_features)]
        else:
            model_layers = [nn.Linear(in_features=in_features,
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
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.factors = [
            (0.5, 0.5),
            (1, 0.5),
            (1, 1)
        ] if factors is None else factors
        self.num_outputs = len(self.factors)
        self.share_gradients = share_gradients

    def _forward_linear(self, layer: nn.Linear, inputs: torch.Tensor,
                        factors: Tuple[float, float], subfactors: Tuple[float, float]) -> torch.Tensor:
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
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subset_shapes2 = (int(round(weight.shape[0] * subfactors[0])), int(round(weight.shape[1] * subfactors[1])))
        if self.share_gradients:
            m_weight = weight[:subset_shapes[0], :subset_shapes[1]]
            m_bias = bias[:subset_shapes[0]]
        else:
            base_weights = weight[:subset_shapes2[0], :subset_shapes2[1]]
            m_weight = torch.cat((base_weights.detach(),
                                  weight[:subset_shapes2[0], subset_shapes2[1]:subset_shapes[1]]), dim=1)
            m_weight = torch.cat((m_weight,
                                  weight[subset_shapes2[0]:subset_shapes[0], :subset_shapes[1]]), dim=0)
            base_bias = bias[:subset_shapes2[0]]
            m_bias = torch.cat((base_bias.detach(),
                                bias[subset_shapes2[0]:subset_shapes[0]]))
        return inputs @ m_weight.T + m_bias[None]

    def _forward_layernorm(self, layer: nn.LayerNorm, inputs: torch.Tensor,
                           factor: float, subfactor: float) -> torch.Tensor:
        var, mean = torch.var_mean(inputs, dim=1, unbiased=False, keepdim=True)
        x = (inputs - mean) / torch.sqrt(var + 1e-5)
        if layer.elementwise_affine:
            weight = layer.weight
            bias = layer.bias
            subshape = int(round(weight.shape[0] * factor))
            subshape2 = int(round(weight.shape[0] * subfactor))
            if self.share_gradients:
                subweight = weight[:subshape]
                subbias = bias[:subshape]
            else:
                base_weight = weight[:subshape2]
                subweight = torch.cat((base_weight.detach(), weight[subshape2:subshape]))
                base_bias = bias[:subshape2]
                subbias = torch.cat((base_bias.detach(), bias[subshape2:subshape]))
            x = subweight[None] * x + subbias[None]
        return x

    def forward(self, inputs, factors: Optional[Sequence[Tuple[float, float, float, float]]] = None) -> MipNetOutputs:
        if factors is None:
            factors = [self.factors[0] + (0, 0)] + [x + y for x, y in zip(self.factors[1:], self.factors[:-1])]
        model_outputs = []
        for factor in factors:
            x = self._forward_linear(self.model[0], inputs, (factor[0], 1), (factor[2], 1))
            x = self._forward_layernorm(self.model[1], x, factor[0], factor[2])
            x = self.model[2](x)
            for i in range(1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                my_subfactor = (factor[2 + i % 2], factor[2 + (i + 1) % 2])
                x = self._forward_linear(self.model[3 * i], x, my_factor, my_subfactor)
                x = self._forward_layernorm(self.model[3 * i + 1], x, my_factor[0], my_subfactor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(self.model[-1], x, (1, factor[self.layers % 2]), (1, factor[2 + self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return MipNetOutputs(outputs=model_outputs)

    def _numel_linear(self, layer: nn.Linear, factors: Tuple[float, float]) -> int:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return subweight.numel() + subbias.numel()

    def _numel_layernorm(self, layer: nn.LayerNorm, factor: float) -> int:
        if layer.elementwise_affine:
            weight = layer.weight
            bias = layer.bias
            subshape = int(round(weight.shape[0] * factor))
            subweight = weight[:subshape]
            subbias = bias[:subshape]
            return subweight.numel() + subbias.numel()
        return 0

    def num_params(self, lod: int):
        factor = lod if isinstance(lod, Tuple) else self.factors[lod]
        total_params = 0
        total_params += self._numel_linear(self.model[0], (factor[0], 1))
        total_params += self._numel_layernorm(self.model[1], factor[0])
        for i in range(1, self.layers - 1):
            my_factor = (factor[i % 2], factor[(i + 1) % 2])
            total_params += self._numel_linear(self.model[3 * i], my_factor)
            total_params += self._numel_layernorm(self.model[3 * i + 1], my_factor[0])
        total_params += self._numel_linear(self.model[-1], (1, factor[self.layers % 2]))
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
