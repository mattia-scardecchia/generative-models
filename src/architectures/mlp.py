import math

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int | None = None,
    activation: str = "ReLU",
) -> nn.Sequential:
    """Build MLP: input -> [hidden + activation]... -> optional output projection.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer widths.
        output_dim: If given, adds a final Linear(last_hidden, output_dim) without activation.
            If None, the output is the last hidden layer (with activation applied).
        activation: Name of an ``nn`` activation class.
    """
    activation_cls = getattr(nn, activation)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation_cls())
        in_dim = h_dim
    if output_dim is not None:
        layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Single residual block: x + Linear(expand) -> Activation -> Linear(contract)."""

    def __init__(self, dim: int, expansion: int = 4, activation: str = "ReLU"):
        super().__init__()
        activation_cls = getattr(nn, activation)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            activation_cls(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def build_resmlp(
    dim: int,
    n_blocks: int,
    expansion: int = 4,
    activation: str = "ReLU",
) -> nn.Sequential:
    """Build a sequence of residual MLP blocks (dimension-preserving).

    Each block: x -> Linear(dim, dim*expansion) -> Activation -> Linear(dim*expansion, dim) -> + x

    Args:
        dim: Hidden dimension (preserved through all blocks).
        n_blocks: Number of residual blocks.
        expansion: Inner expansion factor (default 4x).
        activation: Name of an ``nn`` activation class.
    """
    return nn.Sequential(*[ResidualBlock(dim, expansion, activation) for _ in range(n_blocks)])


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


ARCHITECTURE_TYPES = {"mlp", "resmlp"}


def build_backbone(
    architecture: dict, input_dim: int, output_dim: int | None = None,
) -> tuple[nn.Module, int]:
    """Build a backbone network from architecture config.

    Args:
        architecture: Dict with 'type' and architecture-specific kwargs.
            For 'mlp': hidden_dims, activation.
            For 'resmlp': hidden_dim, n_blocks, expansion, activation.
        input_dim: Input feature dimension.
        output_dim: If given, adds a final output projection. If None, outputs
            the last hidden dimension (feature extractor mode).

    Returns:
        (net, effective_output_dim) tuple.
    """
    arch_type = architecture["type"]

    if arch_type == "mlp":
        hidden_dims = list(architecture["hidden_dims"])
        activation = architecture.get("activation", "ReLU")
        net = build_mlp(input_dim, hidden_dims, output_dim=output_dim, activation=activation)
        eff_dim = output_dim if output_dim is not None else hidden_dims[-1]
    elif arch_type == "resmlp":
        hidden_dim = architecture["hidden_dim"]
        n_blocks = architecture["n_blocks"]
        expansion = architecture.get("expansion", 4)
        activation = architecture.get("activation", "ReLU")
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            build_resmlp(hidden_dim, n_blocks, expansion, activation),
        ]
        if output_dim is not None:
            layers.append(nn.Linear(hidden_dim, output_dim))
        net = nn.Sequential(*layers)
        eff_dim = output_dim if output_dim is not None else hidden_dim
    else:
        raise ValueError(f"Unknown architecture type '{arch_type}'. Choose from: {ARCHITECTURE_TYPES}")

    return net, eff_dim
