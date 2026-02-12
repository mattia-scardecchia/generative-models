import math

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

        encoder_output_dim = encoder.output_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        self.decoder_log_var = nn.Parameter(torch.zeros(1))

    @property
    def decoder_var(self) -> torch.Tensor:
        return torch.exp(self.decoder_log_var)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample z from q(z|x) = N(μ, diag(σ²))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _log_prob_factorized_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Log probability of x under a Gaussian N(μ, diag(σ²)).

        Full formula:
            log N(x; μ, σ²) = -D/2 * log(2π) - 1/2 * Σᵢ [log(σᵢ²) + (xᵢ - μᵢ)² / σᵢ²]

        Args:
            x: Samples, shape (batch, dim).
            mu: Mean, shape (batch, dim) or broadcastable.
            logvar: Log variance log(σ²), shape (batch, dim) or broadcastable.

        Returns:
            Log probabilities, shape (batch,).
        """
        d = x.shape[1]
        const = -0.5 * d * math.log(2 * math.pi)
        return const - 0.5 * torch.sum(logvar + (x - mu) ** 2 / torch.exp(logvar), dim=1)

    def _log_prob_x_given_z(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Log likelihood log p(x|z) = N(x; f(z), σ²I), where:
          - f(z) is the decoder network output
          - σ² is a learned scalar shared across all dimensions

        Full formula:
            log p(x|z) = -D/2 * log(2π) - D/2 * log(σ²) - ||x - f(z)||² / (2σ²)

        Returns:
            Per-sample log probabilities, shape (batch,).
        """
        d = x.shape[1]
        logvar = self.decoder_log_var.expand(d)
        return self._log_prob_factorized_gaussian(x, x_recon, logvar)

    def _log_prob_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Log prior log p(z) under standard normal N(0, I).

        Full formula:
            log p(z) = -D/2 * log(2π) - 1/2 * ||z||²

        Returns:
            Per-sample log probabilities, shape (batch,).
        """
        d = z.shape[1]
        const = -0.5 * d * math.log(2 * math.pi)
        return const - 0.5 * torch.sum(z**2, dim=1)

    def _log_prob_variational_posterior(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Log posterior log q(z|x) under the encoder's variational distribution.

        The encoder outputs μ(x) and log(σ²(x)), defining q(z|x) = N(z; μ, diag(σ²)).

        Full formula:
            log q(z|x) = -D/2 * log(2π) - 1/2 * Σᵢ [log(σᵢ²) + (zᵢ - μᵢ)² / σᵢ²]

        Returns:
            Per-sample log probabilities, shape (batch,).
        """
        return self._log_prob_factorized_gaussian(z, mu, logvar)

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence KL(q(z|x) || p(z)) in closed form. q(z|x) = N(μ, diag(σ²)), p(z) = N(0, I).

        Derivation:
            KL(q||p) = E_q[log q(z|x) - log p(z)]
                     = E_q[-1/2 * Σᵢ(log σᵢ² + (zᵢ-μᵢ)²/σᵢ²) + 1/2 * Σᵢ zᵢ²]
                     = 1/2 * Σᵢ [-log σᵢ² - 1 + σᵢ² + μᵢ²]
                     = -1/2 * Σᵢ [1 + log(σᵢ²) - μᵢ² - σᵢ²]

        Returns:
            Per-sample KL divergence, shape (batch,).
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def _compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute VAE loss: -ELBO = -E[log p(x|z)] + β * KL(q(z|x) || p(z))"""
        x_recon, mu, logvar = self(x)
        recon_loss = -torch.mean(self._log_prob_x_given_z(x, x_recon))
        kl_loss = torch.mean(self._kl_divergence(mu, logvar))
        loss = recon_loss + self.beta * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss, "decoder_var": self.decoder_var}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/recon_loss", losses["recon_loss"])
        self.log("train/kl_loss", losses["kl_loss"])
        self.log("train/decoder_std", losses["decoder_var"].sqrt())
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Compute losses with gradients enabled to measure gradient magnitudes
        with torch.enable_grad():
            x_recon, mu, logvar = self(x)
            recon_loss = -torch.mean(self._log_prob_x_given_z(x, x_recon))
            kl_loss = torch.mean(self._kl_divergence(mu, logvar))

            # Compute gradient magnitude for reconstruction loss
            self.zero_grad()
            recon_loss.backward(retain_graph=True)
            recon_grad_norm = self._compute_grad_norm()

            # Compute gradient magnitude for KL loss
            self.zero_grad()
            kl_loss.backward()
            kl_grad_norm = self._compute_grad_norm()

            self.zero_grad()

        loss = recon_loss + self.beta * kl_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss)
        self.log("val/kl_loss", kl_loss)
        self.log("val/decoder_std", self.decoder_var.sqrt())
        self.log("val/recon_grad_norm", recon_grad_norm)
        self.log("val/kl_grad_norm", kl_grad_norm)

        return loss

    def _compute_grad_norm(self) -> torch.Tensor:
        """Compute the L2 norm of gradients across all parameters."""
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.pow(2).sum().item()
        return torch.tensor(total_norm_sq).sqrt()

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
                  ~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~
                   Monte Carlo estimate    closed form

        Args:
            x: Input samples of shape (batch_size, ...).
            n_samples: Number of latent samples for Monte Carlo estimate of E[log p(x|z)].

        Returns:
            Per-sample ELBO values of shape (batch_size,).
        """
        mu, logvar = self.encode(x)
        recon_samples = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            recon_samples.append(self._log_prob_x_given_z(x, x_recon))
        recon_term = torch.stack(recon_samples).mean(dim=0)
        kl_term = self._kl_divergence(mu, logvar)
        return recon_term - kl_term

    def importance_sampling_log_prob_estimate(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Estimate log p(x) using importance sampling. Uses the variational posterior q(z|x) as the proposal distribution.
            log p(x) = log E_q(z|x)[p(x|z) * p(z) / q(z|x)]
                     ≈ log(1/K * Σᵢ [p(x|z_k) * p(z_k) / q(z_k|x)]),
        where z_k ~ q(z|x).

        This is estimated using the log-sum-exp trick for numerical stability:
            log p(x) ≈ log(1/K) + logsumexp_k[log p(x|z_k) + log p(z_k) - log q(z_k|x)]

        Args:
            x: Input samples of shape (batch_size, ...).
            n_samples: Number of importance samples (K).

        Returns:
            Per-sample log-likelihood estimates of shape (batch_size,).
        """
        batch_size = x.shape[0]
        mu, logvar = self.encode(x)

        # Expand for K samples: (batch, latent_dim) -> (batch, K, latent_dim)
        mu_expanded = mu.unsqueeze(1).expand(-1, n_samples, -1)
        logvar_expanded = logvar.unsqueeze(1).expand(-1, n_samples, -1)

        # Sample all z's at once: (batch, K, latent_dim)
        std = torch.exp(0.5 * logvar_expanded)
        eps = torch.randn_like(std)
        z = mu_expanded + std * eps

        # Flatten for decoder: (batch * K, latent_dim)
        z_flat = z.view(batch_size * n_samples, -1)
        x_recon_flat = self.decode(z_flat)

        # Expand x for all K samples: (batch * K, ...)
        x_expanded = x.unsqueeze(1).expand(-1, n_samples, *x.shape[1:]).reshape(batch_size * n_samples, *x.shape[1:])

        # Compute log probs in batch
        log_prob_x_given_z = self._log_prob_x_given_z(x_expanded, x_recon_flat)  # (batch * K,)
        log_prob_prior = self._log_prob_prior(z_flat)  # (batch * K,)
        log_prob_q = self._log_prob_variational_posterior(
            z_flat,
            mu.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1),
            logvar.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1),
        )  # (batch * K,)

        # Importance weight: p(x|z) * p(z) / q(z|x)
        log_weights = log_prob_x_given_z + log_prob_prior - log_prob_q  # (batch * K,)

        # Reshape to (batch, K) and apply logsumexp
        log_weights = log_weights.view(batch_size, n_samples)

        # log p(x) ≈ logsumexp(log_weights) - log(n_samples)
        return torch.logsumexp(log_weights, dim=1) - torch.log(
            torch.tensor(n_samples, dtype=x.dtype, device=x.device)
        )


class IWAE(BetaVAE):
    """Importance Weighted Autoencoder.

    Optimizes a tighter bound than the standard ELBO using K importance samples:
        L_K(x) = E[log (1/K) Σ_k w_k]

    where w_k = p(x|z_k) * p(z_k) / q(z_k|x) and z_k ~ q(z|x).

    As K → ∞, L_K → log p(x). For K=1, this reduces to the standard VAE ELBO.

    Reference: Burda et al., "Importance Weighted Autoencoders" (2016)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        k: int = 5,
        lr: float = 1e-3,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            beta=1.0,
            lr=lr,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )
        self.k = k
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def _compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute IWAE loss: -L_K(x)"""
        iwae_bound = self.importance_sampling_log_prob_estimate(x, self.k)
        loss = -torch.mean(iwae_bound)
        return {"loss": loss, "iwae_bound": torch.mean(iwae_bound), "decoder_var": self.decoder_var}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/iwae_bound", losses["iwae_bound"])
        self.log("train/decoder_std", losses["decoder_var"].sqrt())
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("val/loss", losses["loss"], prog_bar=True)
        self.log("val/iwae_bound", losses["iwae_bound"])
        self.log("val/decoder_std", self.decoder_var.sqrt())
        return losses["loss"]
