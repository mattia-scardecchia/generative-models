import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar time input."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPVelocityField(nn.Module):
    """Time-conditioned MLP. Used for flow matching velocity fields and diffusion denoisers."""

    def __init__(self, data_dim: int, hidden_dims: list[int], time_embed_dim: int = 32, activation: str = "ReLU"):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        activation_cls = getattr(nn, activation)
        layers = []
        in_dim = data_dim + time_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        return self.net(torch.cat([x, t_embed], dim=-1))


class MLPEncoder(nn.Module):
    """MLP feature extractor. Maps input to a hidden representation.

    Model-agnostic: does NOT produce mu/logvar or any model-specific outputs.
    The generative model (e.g., BetaVAE) adds its own projection heads.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str = "ReLU"):
        super().__init__()
        activation_cls = getattr(nn, activation)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_cls())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    """MLP decoder. Maps a latent code to a reconstruction in data space."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, activation: str = "ReLU"):
        super().__init__()
        activation_cls = getattr(nn, activation)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
