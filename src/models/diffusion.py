"""Diffusion model with configurable noise schedule and prediction parameterization.

Supports:
- Noise schedules: VP (cosine, linear beta), VE
- Prediction types: epsilon (noise), x0 (data), velocity
- Samplers: DDIM (deterministic), DDPM (stochastic), or interpolations via eta

References:
    Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    Song et al., "Denoising Diffusion Implicit Models" (2021)
    Salimans & Ho, "Progressive Distillation for Fast Sampling" (2022)
"""

import torch
import torch.nn as nn

from src.models.base import GenerativeModel
from src.noise_schedules import (
    NoiseSchedule,
    VPCosineSchedule,
    VPLinearSchedule,
    VESchedule,
)

SCHEDULE_REGISTRY: dict[str, type[NoiseSchedule]] = {
    "vp_cosine": VPCosineSchedule,
    "vp_linear": VPLinearSchedule,
    "ve": VESchedule,
}

PREDICTION_TYPES = {"epsilon", "x0", "velocity"}
SAMPLER_TYPES = {"ddim", "ddpm"}


class Diffusion(GenerativeModel):
    """Diffusion model with DDIM/DDPM sampling.

    The denoiser f_θ(x_t, t) predicts a target whose type depends on
    prediction_type. During sampling, all predictions are converted to
    (x̂₀, ε̂) for the update rule selected by `sampler` and `eta`.

    DDIM (eta=0): x_s = α_s x̂₀ + σ_s ε̂  (deterministic)
    DDPM (eta=1): adds stochastic noise scaled by the posterior variance

    Args:
        denoiser: Time-conditioned network, forward(x, t) → output.
        data_dim: Dimensionality of data space.
        noise_schedule: Schedule name ("vp_cosine", "vp_linear", "ve").
        prediction_type: What the network predicts ("epsilon", "x0", "velocity").
        sampler: Sampling algorithm ("ddim" or "ddpm").
        eta: Stochasticity. None → 0.0 for ddim, 1.0 for ddpm. Explicit value overrides.
        n_sampling_steps: Number of sampling steps.
        lr: Learning rate.
        schedule_kwargs: Extra kwargs for the noise schedule constructor.
        optimizer_config: Optional optimizer config dict.
        scheduler_config: Optional LR scheduler config dict.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        data_dim: int,
        noise_schedule: str = "vp_cosine",
        prediction_type: str = "epsilon",
        sampler: str = "ddim",
        eta: float | None = None,
        n_sampling_steps: int = 100,
        lr: float = 1e-3,
        schedule_kwargs: dict | None = None,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"])

        if noise_schedule not in SCHEDULE_REGISTRY:
            raise ValueError(
                f"Unknown noise schedule '{noise_schedule}'. "
                f"Choose from: {list(SCHEDULE_REGISTRY.keys())}"
            )
        if prediction_type not in PREDICTION_TYPES:
            raise ValueError(
                f"Unknown prediction type '{prediction_type}'. "
                f"Choose from: {PREDICTION_TYPES}"
            )
        if sampler not in SAMPLER_TYPES:
            raise ValueError(
                f"Unknown sampler '{sampler}'. "
                f"Choose from: {SAMPLER_TYPES}"
            )

        self.denoiser = denoiser
        self.data_dim = data_dim
        self.prediction_type = prediction_type
        self.n_sampling_steps = n_sampling_steps
        self.eta = eta if eta is not None else (0.0 if sampler == "ddim" else 1.0)

        schedule_cls = SCHEDULE_REGISTRY[noise_schedule]
        self.noise_schedule: NoiseSchedule = schedule_cls(**(schedule_kwargs or {}))

    def _get_target(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute training target based on prediction_type.

        Args:
            x_0: Clean data, (batch, data_dim).
            noise: Sampled ε, (batch, data_dim).
            t: Time values, (batch, 1).
        """
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "x0":
            return x_0
        else:  # velocity
            alpha_t = self.noise_schedule.alpha(t)
            sigma_t = self.noise_schedule.sigma(t)
            return alpha_t * noise - sigma_t * x_0

    def _predict_x0_and_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, network_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert network output to (x̂₀, ε̂) for any prediction type.

        This is the central abstraction: the DDIM sampler only calls this
        method, making it completely prediction-type agnostic.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        alpha_t = self.noise_schedule.alpha(t)
        sigma_t = self.noise_schedule.sigma(t)

        if self.prediction_type == "epsilon":
            eps_hat = network_output
            x0_hat = (x_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-8)
        elif self.prediction_type == "x0":
            x0_hat = network_output
            eps_hat = (x_t - alpha_t * x0_hat) / sigma_t.clamp(min=1e-8)
        else:  # velocity
            v_hat = network_output
            x0_hat = alpha_t * x_t - sigma_t * v_hat
            eps_hat = sigma_t * x_t + alpha_t * v_hat

        return x0_hat, eps_hat

    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Diffusion training loss: E_{t,x₀,ε} ||f_θ(x_t, t) - target||²."""
        batch_size = x.shape[0]
        t = torch.rand(batch_size, 1, device=x.device)
        noise = torch.randn_like(x)

        x_t = self.noise_schedule.q_sample(x, t, noise)
        target = self._get_target(x, noise, t)

        prediction = self.denoiser(x_t, t.squeeze(-1))
        return torch.mean((prediction - target) ** 2)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self, n_samples: int, return_trajectories: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from t=1 (noise) to t=0 (data).

        When eta=0 (DDIM), the update is deterministic:
            x_s = α_s x̂₀ + σ_s ε̂
        When eta>0 (DDPM-like), stochastic noise is injected via the
        posterior variance.
        """
        schedule = self.noise_schedule
        t_max = 1.0 - 1e-3
        ts = torch.linspace(t_max, 0.0, self.n_sampling_steps + 1, device=self.device)

        # Initialize at the correct noise level for the schedule
        sigma_max = schedule.sigma(torch.tensor(t_max, device=self.device))
        x = torch.randn(n_samples, self.data_dim, device=self.device) * sigma_max

        if return_trajectories:
            trajectory = [x.clone()]

        for i in range(self.n_sampling_steps):
            t_now = ts[i]
            t_next = ts[i + 1]

            t_batch = t_now.expand(n_samples)
            output = self.denoiser(x, t_batch)
            x0_hat, eps_hat = self._predict_x0_and_eps(x, t_batch, output)

            if t_next > 0:
                alpha_s = schedule.alpha(t_next.unsqueeze(0)).squeeze(0)
                sigma_s = schedule.sigma(t_next.unsqueeze(0)).squeeze(0)

                if self.eta > 0:
                    alpha_t = schedule.alpha(t_now.unsqueeze(0)).squeeze(0)
                    sigma_t = schedule.sigma(t_now.unsqueeze(0)).squeeze(0)
                    # Posterior variance: σ̃² = σ_s² · σ²_{t|s} / σ_t²
                    sigma_t_given_s_sq = (
                        sigma_t**2 - (alpha_t / alpha_s) ** 2 * sigma_s**2
                    ).clamp(min=0)
                    sigma_tilde = (
                        sigma_s * sigma_t_given_s_sq.sqrt() / sigma_t.clamp(min=1e-8)
                    )
                    coeff_eps = (
                        (sigma_s**2 - (self.eta * sigma_tilde) ** 2).clamp(min=0).sqrt()
                    )
                    z = torch.randn_like(x)
                    x = alpha_s * x0_hat + coeff_eps * eps_hat + self.eta * sigma_tilde * z
                else:
                    x = alpha_s * x0_hat + sigma_s * eps_hat
            else:
                x = x0_hat

            if return_trajectories:
                trajectory.append(x.clone())

        if return_trajectories:
            return x, torch.stack(trajectory)
        return x

    def evaluate(self, datamodule, train_data, train_labels, output_dir, cfg) -> None:
        from src.eval.trajectory import evaluate_trajectory_model
        evaluate_trajectory_model(self, datamodule, train_data, train_labels, output_dir, cfg)
