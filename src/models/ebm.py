import torch

from src.architectures import EnergyNet
from src.models.base import GenerativeModel


class EBM(GenerativeModel):
    """Energy-Based Model trained with Contrastive Divergence.

    Defines p(x) ∝ exp(-E(x)) where E is a learned energy function.
    Training uses Persistent Contrastive Divergence: the loss is the difference
    between the mean energy of data samples and the mean energy of negative
    samples obtained via Langevin MCMC, with a replay buffer for warm starts.

    Reference: Hinton, "Training Products of Experts by Minimizing Contrastive
    Divergence" (2002); Du & Mordatch, "Implicit Generation and Modeling with
    Energy-Based Models" (2019).
    """

    def __init__(
        self,
        data_dim: int,
        architecture: dict,
        lr: float = 1e-4,
        langevin_steps: int = 60,
        langevin_step_size: float = 0.01,
        langevin_noise_scale: float = 0.005,
        grad_clamp: float = 0.03,
        replay_buffer_size: int = 10000,
        replay_prob: float = 0.95,
        energy_reg_weight: float = 1.0,
        init_range: float = 2.0,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.energy_net = EnergyNet(data_dim, architecture)
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size
        self.langevin_noise_scale = langevin_noise_scale
        self.grad_clamp = grad_clamp
        self.replay_buffer_size = replay_buffer_size
        self.replay_prob = replay_prob
        self.energy_reg_weight = energy_reg_weight
        self.data_dim = data_dim
        self.init_range = init_range

        # Persistent replay buffer (not saved in checkpoints)
        self.register_buffer("_replay_buffer", torch.zeros(0, data_dim), persistent=False)

    def langevin_dynamics(
        self,
        x_init: torch.Tensor,
        n_steps: int | None = None,
        step_size: float | None = None,
        noise_scale: float | None = None,
    ) -> torch.Tensor:
        """Run Langevin MCMC to generate negative samples.

        x_{t+1} = x_t - step_size * ∇_x E(x_t) + noise_scale * sqrt(2 * step_size) * z_t

        where z_t ~ N(0, I). Gradients are clamped for stability.
        The chain is fully detached from the parameter graph (stop-gradient CD).
        """
        n_steps = n_steps if n_steps is not None else self.langevin_steps
        step_size = step_size if step_size is not None else self.langevin_step_size
        noise_scale = noise_scale if noise_scale is not None else self.langevin_noise_scale

        x = x_init.clone().detach()

        for _ in range(n_steps):
            x.requires_grad_(True)
            with torch.enable_grad():
                energy = self.energy_net(x).sum()
                grad = torch.autograd.grad(energy, x)[0]
            grad = grad.clamp(-self.grad_clamp, self.grad_clamp)

            x = x.detach()
            noise = torch.randn_like(x)
            x = x - step_size * grad + noise_scale * (2 * step_size) ** 0.5 * noise

        return x.detach()

    def _get_negative_samples(self, batch_size: int) -> torch.Tensor:
        """Initialize negative samples from replay buffer + fresh noise, then refine with Langevin."""
        n_replay = 0
        if self._replay_buffer.shape[0] > 0:
            n_replay = int(batch_size * self.replay_prob)

        n_fresh = batch_size - n_replay
        x_fresh = (torch.rand(n_fresh, self.data_dim, device=self.device) * 2 - 1) * self.init_range

        if n_replay > 0:
            indices = torch.randint(0, self._replay_buffer.shape[0], (n_replay,))
            x_replay = self._replay_buffer[indices].clone()
            x_init = torch.cat([x_replay, x_fresh], dim=0)
        else:
            x_init = x_fresh

        x_neg = self.langevin_dynamics(x_init)
        self._update_replay_buffer(x_neg)
        return x_neg

    def _update_replay_buffer(self, samples: torch.Tensor) -> None:
        """Add new samples to the replay buffer, evicting oldest if full."""
        new_buffer = torch.cat([self._replay_buffer, samples.detach()], dim=0)
        if new_buffer.shape[0] > self.replay_buffer_size:
            new_buffer = new_buffer[-self.replay_buffer_size :]
        self._replay_buffer = new_buffer

    def _compute_loss(self, x_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Contrastive Divergence loss with energy magnitude regularization.

        L = E_data[E(x)] - E_neg[E(x_neg)] + reg * (E[E(x)^2] + E[E(x_neg)^2])
        """
        energy_pos = self.energy_net(x_data)

        x_neg = self._get_negative_samples(x_data.shape[0])
        energy_neg = self.energy_net(x_neg)

        cd_loss = energy_pos.mean() - energy_neg.mean()
        reg_loss = (energy_pos**2).mean() + (energy_neg**2).mean()
        loss = cd_loss + self.energy_reg_weight * reg_loss

        return {
            "loss": loss,
            "cd_loss": cd_loss,
            "reg_loss": reg_loss,
            "energy_pos_mean": energy_pos.mean(),
            "energy_neg_mean": energy_neg.mean(),
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/cd_loss", losses["cd_loss"])
        self.log("train/reg_loss", losses["reg_loss"])
        self.log("train/energy_pos", losses["energy_pos_mean"])
        self.log("train/energy_neg", losses["energy_neg_mean"])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        energy_pos = self.energy_net(x)
        self.log("val/loss", energy_pos.mean(), prog_bar=True)
        self.log("val/energy_pos", energy_pos.mean())
        return energy_pos.mean()

    @torch.no_grad()
    def sample(self, n_samples: int, n_steps: int | None = None) -> torch.Tensor:
        """Generate samples via Langevin dynamics from uniform initialization."""
        n_steps = n_steps if n_steps is not None else self.langevin_steps * 3
        x_init = (torch.rand(n_samples, self.data_dim, device=self.device) * 2 - 1) * self.init_range
        return self.langevin_dynamics(x_init, n_steps=n_steps)

    def evaluate(self, datamodule, train_data, train_labels, output_dir, cfg) -> None:
        from src.eval.ebm import evaluate_ebm
        evaluate_ebm(self, datamodule, train_data, train_labels, output_dir, cfg)
