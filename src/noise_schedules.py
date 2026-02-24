"""Noise schedules for diffusion models.

Each schedule defines the marginal q(x_t | x_0) = N(α_t x_0, σ_t² I)
via (α_t, σ_t) as functions of continuous time t ∈ [0, 1].

Convention: t=0 is clean data, t=1 is maximum noise.
"""

import math
from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """Base class for noise schedules."""

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient α_t."""
        ...

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient σ_t."""
        ...

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t = α_t x_0 + σ_t ε."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.alpha(t) * x_0 + self.sigma(t) * noise


class VPCosineSchedule(NoiseSchedule):
    """Variance Preserving with cosine parameterization.

    α_t = cos(π/2 · t),  σ_t = sin(π/2 · t)
    Satisfies α_t² + σ_t² = 1.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi / 2 * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi / 2 * t)


class VPLinearSchedule(NoiseSchedule):
    """Variance Preserving with linear beta schedule.

    β(s) = β_min + s(β_max - β_min)
    log α_t = -½ ∫₀ᵗ β(s)ds = -½(β_min t + ½(β_max - β_min)t²)
    σ_t = √(1 - α_t²)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        return torch.exp(log_alpha)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alpha(t) ** 2)


class VESchedule(NoiseSchedule):
    """Variance Exploding schedule.

    α_t = 1,  σ_t = σ_min · (σ_max / σ_min)^t
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
