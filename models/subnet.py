"""This is a network with multiple sized layers."""
import dataclasses
from typing import Tuple, Sequence, Optional

import torch
from torch import nn

from protos_compiled import model_pb2


@dataclasses.dataclass
class SubNetOutputs:
    """ Outputs for the SubNet.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class SubNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 factors: Optional[Sequence[Tuple[float, float]]] = None):
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
        self.svd_dict = {}

    def _forward_linear(self, layer: nn.Linear, inputs: torch.Tensor,
                        factors: Tuple[float, float]) -> torch.Tensor:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return inputs @ subweight.T + subbias[None]

    def _forward_layernorm(self, layer: nn.LayerNorm, inputs: torch.Tensor,
                           factor: float) -> torch.Tensor:
        var, mean = torch.var_mean(inputs, dim=1, unbiased=False, keepdim=True)
        x = (inputs - mean) / torch.sqrt(var + 1e-5)
        if layer.elementwise_affine:
            weight = layer.weight
            bias = layer.bias
            subshape = int(round(weight.shape[0] * factor))
            subweight = weight[:subshape]
            subbias = bias[:subshape]
            x = subweight[None] * x + subbias[None]
        return x

    def forward(self, inputs, factors=None, svd=False, **options) -> SubNetOutputs:
        if svd:
            return self.forward_svd(inputs, factors, **options)
        if factors is None:
            factors = self.factors
        model_outputs = []
        for factor in factors:
            x = self._forward_linear(self.model[0], inputs, (factor[0], 1))
            x = self._forward_layernorm(self.model[1], x, factor[0])
            x = self.model[2](x)
            for i in range(1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                x = self._forward_linear(self.model[3 * i], x, my_factor)
                x = self._forward_layernorm(self.model[3 * i + 1], x, my_factor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(self.model[-1], x, (1, factor[self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return SubNetOutputs(outputs=model_outputs)

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
        network.subnet.in_features = self.in_features
        network.subnet.out_features = self.out_features
        network.subnet.layers = self.layers
        network.subnet.hidden_features = self.hidden_features
        network.subnet.use_layernorm = self.use_layernorm
        return network

    def _forward_linear_svd(self, weight_svd, bias, inputs: torch.Tensor,
                            factors: Tuple[float, float], svd_components: int) -> torch.Tensor:
        wu, ws, wvh = weight_svd
        weight = wu[:, :svd_components] @ torch.diag(ws[:svd_components]) @ wvh[:svd_components, :]
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return inputs @ subweight.T + subbias[None]

    def forward_svd(self, inputs, factors=None, svd_components: int = 32) -> SubNetOutputs:
        if factors is None:
            factors = self.factors
        model_outputs = []
        for factor in factors:
            x = self._forward_linear(self.model[0], inputs, (factor[0], 1))
            x = self._forward_layernorm(self.model[1], x, factor[0])
            x = self.model[2](x)
            for i in range(1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                if f"{i}_{svd_components}" not in self.svd_dict:
                    self.svd_dict[f"{i}_{svd_components}"] = torch.linalg.svd(self.model[3 * i].weight)
                x = self._forward_linear_svd(self.svd_dict[f"{i}_{svd_components}"], self.model[3 * i].bias, x,
                                             my_factor, svd_components=svd_components)
                x = self._forward_layernorm(self.model[3 * i + 1], x, my_factor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(self.model[-1], x, (1, factor[self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return SubNetOutputs(outputs=model_outputs)
