"""Wrapper model which consists of multiple submodels."""
import dataclasses
from typing import Tuple

import torch
from torch import nn
import numpy as np

from models import mlp

from protos_compiled import model_pb2


def solve_quadratic(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Solution to quadratic formula."""
    return (-b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)


@dataclasses.dataclass
class MultiResnetOutputs:
    """Outputs for the multi model MLP.

    Attributes:
        model_outputs: The final outputs.
        selection_indices: Indices for which model to use.
        selection_logits: Outputs for the selection model.
        selection_probabilities: Probabilities for which model to use.
    """
    model_outputs: torch.Tensor
    selection_indices: torch.Tensor
    selection_logits: torch.Tensor
    selection_probabilities: torch.Tensor


class MultiResnetLinear(nn.Module):
    def __init__(self, num_models: int, in_features: int, out_features: int):
        super().__init__()
        sqrt_k = np.sqrt(1 / num_models)

        self.weights = nn.Parameter(
            2 * sqrt_k * torch.rand((num_models, out_features, in_features), dtype=torch.float32) - sqrt_k)
        self.biases = nn.Parameter(2 * sqrt_k * torch.rand((num_models, out_features), dtype=torch.float32) - sqrt_k)

    def forward(self, inputs: torch.Tensor, indices: torch.Tensor):
        weights = self.weights[indices]
        biases = self.biases[indices]
        return torch.matmul(weights, inputs[:, :, None])[:, :, 0] + biases


class MultiResnet(nn.Module):
    def __init__(self, num_models: int = 64, in_features: int = 6, out_features: int = 3,
                 layers: int = 5, hidden_features: int = 64,
                 selection_layers: int = 5, selection_hidden_features: int = 64,
                 lerp_value: float = 0.5, selection_mode: str = "angle", output_every: int = 2):
        """Initialize a multi mlp.

        Args:
            num_models: Number of submodels.
            in_features: Input features.
            out_features: Output features.
            layers: Layers per submodel.
            hidden_features: Features per submodel.
            selection_layers: Number of layers for selection network.
            selection_hidden_features: Number of hidden features for selection network.
            lerp_value: Value for lerp.
        """
        super().__init__()
        if num_models <= 0:
            raise ValueError(f"Num models is <= 0: {num_models}")
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.hidden_features = hidden_features
        self.selection_layers = selection_layers
        self.selection_hidden_features = selection_hidden_features
        self.selection_mode = selection_mode
        self.lerp_value = lerp_value
        self.output_every = output_every
        self.num_angle_sections = 8
        self.num_height_sections = 8
        self.max_height = 1.0
        self.min_height = -3.0
        self.radius = 1.0
        self.num_outputs = 3
        self.selection_model = mlp.MLP(in_features=in_features, out_features=num_models, layers=selection_layers,
                                       hidden_features=selection_hidden_features) if selection_mode == "mlp" else None
        model_layers = [MultiResnetLinear(
            num_models=num_models,
            in_features=in_features,
            out_features=hidden_features
        ), nn.ReLU()]
        for layer in range(layers - 2):
            model_layers.append(MultiResnetLinear(
                num_models=num_models,
                in_features=hidden_features,
                out_features=hidden_features
            ))
            model_layers.append(nn.ReLU())
        self.final_layer = MultiResnetLinear(
            num_models=num_models,
            in_features=hidden_features,
            out_features=out_features
        )
        self.model_layers = nn.ModuleList(model_layers)

    def freeze_selection_model(self):
        if self.selection_model is not None:
            for parameter in self.selection_model.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> MultiResnetOutputs:
        """Forward pass.

        Args:
            inputs: Inputs as (B, F) tensor.

        Returns:
            The model outputs as (B, F) tensor.
        """
        if self.selection_mode == "angle":
            angles = torch.atan2(inputs[:, 2], inputs[:, 0])
            angles = torch.fmod(angles + 2 * np.pi, 2 * np.pi) / (2 * np.pi) * self.num_models
            angles = torch.floor(angles)
            selection_indices = angles.long()
            selection_logits = torch.ones((angles.shape[0], self.num_models), dtype=inputs.dtype, device=inputs.device)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
        elif self.selection_mode == "mlp":
            selection_logits = self.selection_model(inputs)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
            selection_indices = torch.argmax(selection_probabilities, dim=1)
        elif self.selection_mode == "cylinder":
            num_angle_sections = self.num_angle_sections
            num_height_sections = self.num_height_sections
            max_height = self.max_height
            min_height = self.min_height
            radius = self.radius
            assert num_angle_sections * num_height_sections >= self.num_models, "Invalid"
            ray_direction = inputs[:, :3]
            ray_points = torch.cross(inputs[:, 3:6], inputs[:, :3])
            t = solve_quadratic(
                ray_direction[:, [0, 2]].pow(2).sum(1),
                (2 * ray_points[:, [0, 2]] * ray_direction[:, [0, 2]]).sum(-1),
                ray_points[:, [0, 2]].pow(2).sum(1) - radius * radius
            )
            intersection_points = ray_points + t[:, None] * ray_direction
            angles = torch.atan2(intersection_points[:, 2], intersection_points[:, 0])
            angles = torch.fmod(angles + 2 * np.pi, 2 * np.pi) / (2 * np.pi) * (self.num_models // num_angle_sections)
            angles = torch.floor(angles)
            heights = torch.clamp(
                num_height_sections * (-intersection_points[:, 1] - min_height) / (max_height - min_height),
                0, num_height_sections - 1)
            selection_indices = heights.long() * num_angle_sections + angles.long()
            selection_indices = torch.where(torch.all(torch.isfinite(intersection_points), dim=1), selection_indices, 0)
            selection_logits = torch.ones((angles.shape[0], self.num_models), dtype=inputs.dtype, device=inputs.device)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
        else:
            raise ValueError(f"Unknown model selection mode {self.selection_mode}")

        model_outputs = []
        x = inputs
        for i in range(self.layers - 1):
            x = self.model_layers[2 * i](x, selection_indices)
            x = self.model_layers[2 * i + 1](x)
            if (i + 1) % self.output_every == 0 and i != self.layers - 2:
                model_outputs.append(self.final_layer(x, selection_indices))
        model_outputs.append(self.final_layer(x, selection_indices))
        model_outputs = torch.stack(model_outputs, dim=1)
        return MultiResnetOutputs(
            model_outputs=model_outputs,
            selection_indices=selection_indices,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities
        )

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
        network.multiresnet.num_models = self.num_models
        network.multiresnet.in_features = self.in_features
        network.multiresnet.out_features = self.out_features
        network.multiresnet.layers = self.layers
        network.multiresnet.hidden_features = self.hidden_features
        network.multiresnet.selection_layers = self.selection_layers
        network.multiresnet.selection_hidden_features = self.selection_hidden_features
        network.multiresnet.selection_mode = self.selection_mode
        network.multiresnet.output_every = self.output_every
        return network


def compute_load_balance_loss(probabilities: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Load balanace loss

    Args:
        probabilities: Probabilities as (B, K) tensor.
        indices: Indices as (B,) tensor.

    Returns:
        Loss value.
    """
    batch_size, num_experts = probabilities.shape
    indices_hot = torch.zeros_like(probabilities).scatter_(1, indices[:, None], 1)
    density_1 = torch.mean(indices_hot, dim=0)
    density_1_proxy = torch.mean(probabilities, dim=0)
    loss = torch.mean(density_1_proxy * density_1) * (num_experts * num_experts)
    return loss
