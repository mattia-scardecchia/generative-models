"""Role-specific network wrappers that delegate to build_backbone."""

import torch
import torch.nn as nn

from src.architectures.mlp import SinusoidalTimeEmbedding, build_backbone


class TimeConditionedNet(nn.Module):
    """Time-conditioned network for diffusion denoisers and flow matching velocity fields."""

    def __init__(self, data_dim: int, architecture: dict):
        super().__init__()
        arch = dict(architecture)
        time_embed_dim = arch.pop("time_embed_dim", 32)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.net, _ = build_backbone(arch, data_dim + time_embed_dim, output_dim=data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


class EncoderNet(nn.Module):
    """Feature extractor (no output projection). Exposes output_dim for downstream heads."""

    def __init__(self, input_dim: int, architecture: dict):
        super().__init__()
        self.net, self.output_dim = build_backbone(architecture, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderNet(nn.Module):
    """Decoder with explicit output projection to data space."""

    def __init__(self, input_dim: int, output_dim: int, architecture: dict):
        super().__init__()
        self.net, _ = build_backbone(architecture, input_dim, output_dim=output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EnergyNet(nn.Module):
    """Scalar energy function E(x): R^d -> R."""

    def __init__(self, input_dim: int, architecture: dict):
        super().__init__()
        self.net, _ = build_backbone(architecture, input_dim, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
