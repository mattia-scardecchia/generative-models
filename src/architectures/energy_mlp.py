import torch
import torch.nn as nn


class MLPEnergy(nn.Module):
    """MLP energy function. Maps input data to a scalar energy value.

    E(x): R^d -> R

    The density is defined as p(x) ∝ exp(-E(x)).
    Uses SiLU by default for smooth gradients (important for Langevin dynamics).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str = "SiLU"):
        super().__init__()
        layers = []
        in_dim = input_dim
        activation_cls = getattr(nn, activation)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns energy values, shape (batch,)."""
        return self.net(x).squeeze(-1)
