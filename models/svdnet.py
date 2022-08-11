import dataclasses
from typing import Tuple, Sequence, Optional

import torch
from torch import nn
import numpy as np

from utils import torch_checkpoint_manager
from protos_compiled import model_pb2


@dataclasses.dataclass
class SVDNetOutputs:
    """Outputs for the SubNet.

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


class SVDNet(nn.Module):
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
                model_layers.append(nn.Linear(in_features=hidden_features,
                                              out_features=hidden_features))
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
        self.factors = [0.2, 0.4, 0.6, 0.8, 1.0] if factors is None else factors
        self.num_outputs = len(self.factors)

    def _forward_linearsvd(self, layer: LinearSVD, inputs: torch.Tensor,
                           factor: float) -> torch.Tensor:
        num_components = int(round(layer.wu.shape[1] * factor))
        return (inputs @ layer.wvh[:num_components, :].T) @ layer.wu[:, :num_components].T + layer.bias

    def _forward_randomized_matrix_multiply(self, layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        # Testing fast randomized matrix multiply
        # (B, 512) @ (512, 512) + (, 512)
        layer_weight = layer.weight.T
        num_samples = 512
        # select_probabilities = torch.ones(layer_size, device=inputs.device, dtype=inputs.dtype)
        inputs = torch.where(torch.isfinite(inputs), inputs, torch.zeros_like(inputs))
        # select_probabilities = torch.linalg.norm(layer_weight, dim=1) * torch.linalg.norm(inputs[:512, :], dim=0)
        select_probabilities = torch.linalg.norm(layer_weight, dim=1) + 1e-5
        assert torch.all(torch.isfinite(select_probabilities)), "select_probabilities has nan"
        # select_probabilities = torch.where(torch.isfinite(select_probabilities), select_probabilities,
        #                                    1e-5 + torch.zeros_like(select_probabilities))
        select_probabilities = select_probabilities / torch.sum(select_probabilities)
        inv_select_probabilities = 1 / select_probabilities
        subset_indices = torch.multinomial(select_probabilities, num_samples, replacement=False)
        multiplied_features = inputs[:, subset_indices] @ (
                inv_select_probabilities[subset_indices, None] * layer_weight[subset_indices, :]) / num_samples
        return multiplied_features + layer.bias[None]

    def _forward_svd_data(self, layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        # Testing fast randomized matrix multiply
        # (B, 512) @ (512, 512) + (, 512)
        dtype = inputs.dtype
        layer_weight = layer.weight.T
        svd_components = 512
        u, s, vh = torch.linalg.svd(inputs.float(), full_matrices=False)
        u, s, vh = u.type(dtype), s.type(dtype), vh.type(dtype)
        inputs_svd = (u[:, :svd_components] @ torch.diag(s[:svd_components]) @ vh[:svd_components, :])
        return inputs_svd @ layer_weight + layer.bias[None]

    def forward(self, inputs, factors=None) -> SVDNetOutputs:
        if factors is None:
            factors = self.factors
        model_outputs = []
        for factor in factors:
            x = self.model[0](inputs)
            x = self.model[1](x)
            x = self.model[2](x)
            for i in range(1, self.layers - 1):
                # x = self.model[3 * i](x)
                x = self._forward_svd_data(self.model[3 * i], x)
                x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)
            x = self.model[-1](x)
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return SVDNetOutputs(outputs=model_outputs)

    def _numel_linear(self, layer: nn.Linear) -> int:
        return layer.weight.numel() + layer.bias.numel()

    def _numel_linearsvd(self, layer: nn.Linear, factor: float) -> int:
        num_components = int(round(layer.wu.shape[1] * factor))
        return layer.wvh[:num_components, :].numel() + layer.wu[:, :num_components].numel() + layer.bias.numel()

    def _numel_layernorm(self, layer: nn.LayerNorm) -> int:
        if layer.elementwise_affine:
            return layer.weight.numel() + layer.bias.numel()
        return 0

    def num_params(self, lod: int):
        factor = lod if isinstance(lod, Tuple) else self.factors[lod]
        total_params = 0
        total_params += self._numel_linear(self.model[0])
        total_params += self._numel_layernorm(self.model[1])
        for i in range(1, self.layers - 1):
            total_params += self._numel_linearsvd(self.model[3 * i], factor)
            total_params += self._numel_layernorm(self.model[3 * i + 1])
        total_params += self._numel_linear(self.model[-1])
        return total_params

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto containing weights.

        Returns:
            The proto.
        """
        raise NotImplementedError("svdnet")
