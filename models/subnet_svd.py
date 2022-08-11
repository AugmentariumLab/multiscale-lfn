"""This is a network with multiple sized layers."""
import dataclasses
from typing import Tuple, Sequence, Optional

import torch
from torch import nn
import numpy as np

from utils import torch_checkpoint_manager
from protos_compiled import model_pb2


@dataclasses.dataclass
class SubNetSVDOutputs:
    """ Outputs for the SubNet.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class LinearSVD(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3, svd_components: int = 2):
        super().__init__()
        sqrt_k = np.sqrt(1 / out_features)
        self.wu = nn.Parameter(
            2 * sqrt_k * torch.rand((out_features, svd_components), dtype=torch.float32) - sqrt_k)
        self.wvh = self.weights = nn.Parameter(
            2 * sqrt_k * torch.rand((svd_components, in_features), dtype=torch.float32) - sqrt_k)
        self.bias = nn.Parameter(
            2 * sqrt_k * torch.rand(out_features, dtype=torch.float32) - sqrt_k)

    def forward(self, x):
        return x @ (self.wu @ self.wvh).transpose() + self.bias[None]


class SubNetSVD(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 svd_components: int = 64, load_from: Optional[str] = None,
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
                model_layers.append(LinearSVD(in_features=hidden_features,
                                              out_features=hidden_features,
                                              svd_components=svd_components))
                if use_layernorm:
                    model_layers.append(nn.LayerNorm(hidden_features))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)

        if load_from:
            checkpoint_manager = torch_checkpoint_manager.CheckpointManager(load_from)
            latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint is None:
                raise ValueError("Load from checkpoint not available")
            subnet_state_dict = latest_checkpoint['model_state_dict']
            for i, layer in enumerate(self.model):
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
                    layer.weight.data.copy_(subnet_state_dict[f"model.{i}.weight"])
                    layer.bias.data.copy_(subnet_state_dict[f"model.{i}.bias"])
                elif isinstance(layer, LinearSVD):
                    wu, ws, wvh = torch.linalg.svd(subnet_state_dict[f"model.{i}.weight"])
                    layer.wu.data.copy_(wu[:, :svd_components])
                    layer.wvh.data.copy_((torch.diag(ws) @ wvh)[:svd_components, :])
                    layer.bias.data.copy_(subnet_state_dict[f"model.{i}.bias"])
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.svd_components = svd_components
        self.load_from = load_from
        self.layers = layers
        self.factors = [
            (0.5, 0.5),
            (1, 0.5),
            (1, 1)
        ] if factors is None else factors
        self.num_outputs = len(self.factors)

    def _forward_linear(self, layer: nn.Linear, inputs: torch.Tensor,
                        factors: Tuple[float, float]) -> torch.Tensor:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return inputs @ subweight.T + subbias[None]

    def _forward_linearsvd(self, layer: LinearSVD, inputs: torch.Tensor,
                           factors: Tuple[float, float]) -> torch.Tensor:
        weight = layer.wu @ layer.wvh
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

    def forward(self, inputs, factors=None, svd=False, **options) -> SubNetSVDOutputs:
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
                x = self._forward_linearsvd(self.model[3 * i], x, my_factor)
                x = self._forward_layernorm(self.model[3 * i + 1], x, my_factor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(self.model[-1], x, (1, factor[self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return SubNetSVDOutputs(outputs=model_outputs)

    def _numel_linear(self, layer: nn.Linear, factors: Tuple[float, float]) -> int:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return subweight.numel() + subbias.numel()

    def _numel_linearsvd(self, layer: LinearSVD, factors: Tuple[float, float]) -> int:
        raise NotImplementedError()
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
            total_params += self._numel_linearsvd(self.model[3 * i], my_factor)
            total_params += self._numel_layernorm(self.model[3 * i + 1], my_factor[0])
        total_params += self._numel_linear(self.model[-1], (1, factor[self.layers % 2]))
        return total_params

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto containing weights.

        Returns:
            The proto.
        """
        raise NotImplementedError("subnetsvd")
