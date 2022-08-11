"""This module contains the adaptive mlp model."""
import dataclasses
from typing import Optional

import torch
from torch import nn


class LinearBlock(nn.Module):
    """Linear-norm-relu block"""

    def __init__(self, in_features: int = 256, out_features: int = 256):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm_layer = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs: Input tensor.

        Returns:
            A tuple containing the outputs before norm and relu and the outputs after norm and relu.
        """
        x1 = self.linear(inputs)
        x2 = self.norm_layer(x1)
        x2 = self.activation(x2)
        return x1, x2


@dataclasses.dataclass
class AdaptiveMLPOutputs:
    """ Outputs for the AdaptiveMLP.

    Attributes:
        expected_output: Expected output.
        layer_outputs: Output for each layer.
        layer_stop_here_probabilities: Unconditional probability of stopping at each layer.
        expected_layers: Expected number of layers to pass through.
    """
    expected_output: torch.Tensor
    expected_layers: torch.Tensor
    layer_outputs: Optional[torch.Tensor] = None
    layer_stop_here_probabilities: Optional[torch.Tensor] = None


class AdaptiveMLP(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 6, hidden_features: int = 256):
        super().__init__()

        model_layers = [
            LinearBlock(in_features=in_features, out_features=hidden_features + out_features + 1)
        ]
        for i in range(layers - 2):
            model_layers.append(LinearBlock(in_features=hidden_features,
                                            out_features=hidden_features + out_features + 1))
        model_layers.append(
            nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model_layers = nn.ModuleList(model_layers)
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.num_outputs = layers

    def forward_with_early_stopping(self, inputs) -> AdaptiveMLPOutputs:
        """Forward pass without early stopping."""
        device = inputs.device
        x = inputs
        # Expected outputs.
        expected_output = torch.zeros((inputs.shape[0], self.out_features), device=device, dtype=inputs.dtype)
        # Expected number of layers to pass through.
        expected_layers = torch.zeros((inputs.shape[0]), device=device, dtype=inputs.dtype)
        all_indices = torch.arange(inputs.shape[0], device=device)
        remaining_indices = all_indices
        for i in range(len(self.model_layers) - 1):
            layer = self.model_layers[i]
            layer_output1, layer_output2 = layer(x)
            probability_stop_cond = torch.sigmoid(layer_output1[:, self.out_features])
            stop_here = torch.lt(torch.rand(remaining_indices.shape, device=device), probability_stop_cond)
            expected_output[remaining_indices[stop_here]] = layer_output1[stop_here, :self.out_features]
            expected_layers[remaining_indices[stop_here]] = i
            remaining_indices = remaining_indices[torch.logical_not(stop_here)]
            x = layer_output2[torch.logical_not(stop_here), (self.out_features + 1):]
        layer_output1 = self.model_layers[len(self.model_layers) - 1](x)
        expected_output[remaining_indices] = layer_output1[:, :self.out_features]
        expected_layers[remaining_indices] = len(self.model_layers) - 1
        return AdaptiveMLPOutputs(
            expected_output=expected_output,
            expected_layers=expected_layers,
        )

    def forward_without_early_stopping(self, inputs) -> AdaptiveMLPOutputs:
        """Forward pass without early stopping."""
        x = inputs
        # Expected outputs.
        expected_output = torch.zeros((inputs.shape[0], self.out_features), device=inputs.device, dtype=inputs.dtype)
        # Expected number of layers to pass through.
        expected_layers = torch.zeros((inputs.shape[0]), device=inputs.device, dtype=inputs.dtype)
        # Individual outputs.
        layer_outputs = []
        layer_stop_here_probabilities = []
        # Probability of having stopped already.
        probability_stopped = torch.zeros((inputs.shape[0]), device=inputs.device, dtype=inputs.dtype)
        for i in range(len(self.model_layers) - 1):
            layer = self.model_layers[i]
            layer_output1, layer_output2 = layer(x)
            probability_stop_cond = torch.sigmoid(layer_output1[:, self.out_features])
            probability_stop_here = (1 - probability_stopped) * probability_stop_cond
            probability_stopped += probability_stop_here
            expected_output += probability_stop_here[:, None] * layer_output1[:, :self.out_features]
            layer_outputs.append(layer_output1[:, :self.out_features])
            layer_stop_here_probabilities.append(probability_stop_here)
            expected_layers += i * probability_stop_here
            x = layer_output2[:, (self.out_features + 1):]
        layer_output1 = self.model_layers[len(self.model_layers) - 1](x)
        probability_stop_here = (1 - probability_stopped)
        probability_stopped += probability_stop_here
        expected_output += probability_stop_here[:, None] * layer_output1[:, :self.out_features]
        layer_outputs.append(layer_output1[:, :self.out_features])
        layer_stop_here_probabilities.append(probability_stop_here)
        expected_layers += (len(self.model_layers) - 1) * probability_stop_here
        return AdaptiveMLPOutputs(
            expected_output=expected_output,
            expected_layers=expected_layers,
            layer_outputs=torch.stack(layer_outputs, dim=1),
            layer_stop_here_probabilities=torch.stack(layer_stop_here_probabilities, dim=1),
        )

    def forward(self, inputs, early_stopping=False) -> AdaptiveMLPOutputs:
        """Forward pass.

        Args:
            inputs: Input tensor.
            early_stopping: Whether to stop the forward pass early.

        Returns:
            Object containing the expected output and the final output.
        """
        if early_stopping:
            return self.forward_with_early_stopping(inputs)
        else:
            return self.forward_without_early_stopping(inputs)
