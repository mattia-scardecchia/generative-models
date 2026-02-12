import torch
import torch.nn as nn
import pytorch_lightning as pl


class BetaVAE(pl.LightningModule):
    """Beta-VAE with configurable encoder and decoder.

    The encoder is a model-agnostic feature extractor. This module adds its own
    fc_mu and fc_logvar projection heads on top of the encoder output.

    When beta > 1, encourages more disentangled latent representations.
    When beta = 1, equivalent to standard VAE.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        beta: float = 1.0,
        lr: float = 1e-3,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # VAE-specific projection heads on top of encoder features
        encoder_output_dim = encoder.output_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def _reconstruction_log_prob(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute log p(x|z) assuming unit variance Gaussian decoder.

        Returns per-sample log probabilities (summed over dimensions).
        """
        # For Gaussian with unit variance: log p(x|z) = -0.5 * ||x - x_recon||^2 + const
        # We omit the constant term as it cancels in relative comparisons
        return -0.5 * torch.sum((x - x_recon) ** 2, dim=list(range(1, x.dim())))

    def _log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) for standard normal prior.

        Returns per-sample log probabilities.
        """
        return -0.5 * torch.sum(z**2, dim=1)

    def _log_posterior(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute log q(z|x) for the variational posterior.

        Returns per-sample log probabilities.
        """
        var = torch.exp(logvar)
        return -0.5 * torch.sum(logvar + (z - mu) ** 2 / var, dim=1)

    def _compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_recon, mu, logvar = self(x)
        # Sum over features, mean over batch (consistent with KL)
        recon_loss = torch.mean(torch.sum((x_recon - x) ** 2, dim=1))
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + self.beta * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/recon_loss", losses["recon_loss"])
        self.log("train/kl_loss", losses["kl_loss"])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("val/loss", losses["loss"], prog_bar=True)
        self.log("val/recon_loss", losses["recon_loss"])
        self.log("val/kl_loss", losses["kl_loss"])
        return losses["loss"]

    def configure_optimizers(self):
        # Build optimizer
        if self.optimizer_config is not None:
            opt_type = self.optimizer_config.get("type", "Adam")
            opt_params = {k: v for k, v in self.optimizer_config.items() if k != "type"}
            opt_params.setdefault("lr", self.lr)
            optimizer_cls = getattr(torch.optim, opt_type)
            optimizer = optimizer_cls(self.parameters(), **opt_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Build scheduler if configured
        if self.scheduler_config is not None:
            sched_type = self.scheduler_config.get("type", "StepLR")
            sched_params = {k: v for k, v in self.scheduler_config.items() if k != "type"}
            scheduler_cls = getattr(torch.optim.lr_scheduler, sched_type)
            scheduler = scheduler_cls(optimizer, **sched_params)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decode(z)

    @torch.no_grad()
    def elbo(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Compute the Evidence Lower Bound (ELBO) for given samples.

        ELBO(x) = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

        Args:
            x: Input samples of shape (batch_size, ...).
            n_samples: Number of latent samples to use for Monte Carlo estimation.

        Returns:
            Per-sample ELBO values of shape (batch_size,).
        """
        mu, logvar = self.encode(x)

        elbo_samples = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)

            log_p_x_given_z = self._reconstruction_log_prob(x, x_recon)
            log_p_z = self._log_prior(z)
            log_q_z_given_x = self._log_posterior(z, mu, logvar)

            # ELBO = log p(x|z) + log p(z) - log q(z|x)
            elbo = log_p_x_given_z + log_p_z - log_q_z_given_x
            elbo_samples.append(elbo)

        return torch.stack(elbo_samples).mean(dim=0)

    @torch.no_grad()
    def log_likelihood(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Estimate log p(x) using importance sampling.

        Uses the variational posterior q(z|x) as the proposal distribution:
            log p(x) = log E_q(z|x)[p(x|z) * p(z) / q(z|x)]

        This is estimated using the log-sum-exp trick for numerical stability:
            log p(x) ≈ log(1/K) + logsumexp_k[log p(x|z_k) + log p(z_k) - log q(z_k|x)]

        where z_k ~ q(z|x).

        Args:
            x: Input samples of shape (batch_size, ...).
            n_samples: Number of importance samples (K).

        Returns:
            Per-sample log-likelihood estimates of shape (batch_size,).
        """
        mu, logvar = self.encode(x)

        log_weights = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)

            log_p_x_given_z = self._reconstruction_log_prob(x, x_recon)
            log_p_z = self._log_prior(z)
            log_q_z_given_x = self._log_posterior(z, mu, logvar)

            # Importance weight: p(x|z) * p(z) / q(z|x)
            log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
            log_weights.append(log_w)

        # Stack: (n_samples, batch_size)
        log_weights = torch.stack(log_weights, dim=0)

        # log p(x) ≈ logsumexp(log_weights) - log(n_samples)
        return torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(n_samples, device=x.device))
