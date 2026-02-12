import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """MLP feature extractor. Maps input to a hidden representation.

    Model-agnostic: does NOT produce mu/logvar or any model-specific outputs.
    The generative model (e.g., BetaVAE) adds its own projection heads.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    """MLP decoder. Maps a latent code to a reconstruction in data space."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
