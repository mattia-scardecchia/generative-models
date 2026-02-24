import math

import torch
import torch.nn as nn

from src.architectures import EncoderNet, DecoderNet
from src.models.base import GenerativeModel


VAR_MAX = 5.0  # Numerical stability


class BetaVAE(GenerativeModel):
    """Beta-VAE with configurable encoder and decoder.

    The encoder is a model-agnostic feature extractor. This module adds its own
    fc_mu and fc_logvar projection heads on top of the encoder output.

    When beta > 1, encourages more disentangled latent representations.
    When beta = 1, equivalent to standard VAE.
    """

    def __init__(
        self,
        data_dim: int,
        architecture: dict,
        latent_dim: int,
        beta: float = 1.0,
        decoder_var: float | str = 1.0,
        lr: float = 1e-3,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = EncoderNet(data_dim, architecture)
        self.decoder = DecoderNet(latent_dim, data_dim, architecture)
        self.latent_dim = latent_dim
        self.beta = beta
        self.data_dim = data_dim

        encoder_output_dim = self.encoder.output_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

        if decoder_var == "learned":
            self._fixed_decoder_var = None
            self.decoder_log_var = nn.Parameter(torch.zeros(1))
        else:
            self._fixed_decoder_var = float(decoder_var)
            self.register_buffer(
                "_decoder_var_tensor", torch.tensor([self._fixed_decoder_var])
            )

    @property
    def decoder_var(self) -> torch.Tensor:
        if self._fixed_decoder_var is not None:
            return self._decoder_var_tensor
        return torch.exp(torch.clamp(self.decoder_log_var, min=-VAR_MAX, max=VAR_MAX))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-VAR_MAX, max=VAR_MAX)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def _log_prob_factorized_gaussian(
        x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Log probability of x under a Gaussian N(μ, diag(σ²)).

        Full formula:
            log N(x; μ, σ²) = -D/2 * log(2π) - 1/2 * Σᵢ [log(σᵢ²) + (xᵢ - μᵢ)² / σᵢ²]

        Args:
            x: Samples, shape (batch, *).
            mu: Mean, shape (batch, *) or broadcastable.
            logvar: Log variance log(σ²), shape (batch, *) or broadcastable.

        Returns:
            Log probabilities, shape (batch,).
        """
        x = x.flatten(1)
        mu = mu.flatten(1)
        d = x.shape[1]
        const = -0.5 * d * math.log(2 * math.pi)
        return const - 0.5 * torch.sum(
            logvar + (x - mu) ** 2 / torch.exp(logvar), dim=1
        )

    def _log_prob_x_given_z(
        self, x: torch.Tensor, x_recon: torch.Tensor
    ) -> torch.Tensor:
        """Log likelihood log p(x|z) = N(x; f(z), σ²I), where:
          - f(z) is the decoder network output
          - σ² is a fixed or learned scalar shared across all dimensions

        Full formula:
            log p(x|z) = -D/2 * log(2π) - D/2 * log(σ²) - ||x - f(z)||² / (2σ²)

        Returns:
            Per-sample log probabilities, shape (batch,).
        """
        d = x[0].numel()
        if self._fixed_decoder_var is not None:
            logvar = torch.full(
                (d,), math.log(self._fixed_decoder_var), device=x.device
            )
        else:
            logvar = torch.clamp(
                self.decoder_log_var, min=-VAR_MAX, max=VAR_MAX
            ).expand(d)
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

    def _log_prob_variational_posterior(
        self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
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
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "decoder_var": self.decoder_var,
            "mu": mu,
            "logvar": logvar,
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/recon_loss", losses["recon_loss"])
        self.log("train/kl_loss", losses["kl_loss"])
        self.log("train/decoder_std", losses["decoder_var"].sqrt())
        self.log("train/posterior_mu_mean", losses["mu"].mean())
        self.log("train/posterior_var_mean", losses["logvar"].exp().mean())
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        losses = self._compute_loss(x)
        self.log("val/loss", losses["loss"], prog_bar=True)
        self.log("val/recon_loss", losses["recon_loss"])
        self.log("val/kl_loss", losses["kl_loss"])
        self.log("val/decoder_std", losses["decoder_var"].sqrt())
        self.log("val/posterior_mu_mean", losses["mu"].mean())
        self.log("val/posterior_var_mean", losses["logvar"].exp().mean())
        return losses["loss"]

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

    def importance_sampling_log_prob_estimate(
        self, x: torch.Tensor, n_samples: int = 100, batch_k: int = 100
    ) -> torch.Tensor:
        """Estimate log p(x) using importance sampling. Uses the variational posterior q(z|x) as the proposal distribution.
            log p(x) = log E_q(z|x)[p(x|z) * p(z) / q(z|x)]
                     ≈ log(1/K * Σᵢ [p(x|z_k) * p(z_k) / q(z_k|x)]),
        where z_k ~ q(z|x).

        This is estimated using the log-sum-exp trick for numerical stability:
            log p(x) ≈ log(1/K) + logsumexp_k[log p(x|z_k) + log p(z_k) - log q(z_k|x)]

        Args:
            x: Input samples of shape (batch_size, ...).
            n_samples: Number of importance samples (K).
            batch_k: Process importance samples in chunks of this size to reduce memory.

        Returns:
            Per-sample log-likelihood estimates of shape (batch_size,).
        """
        batch_size = x.shape[0]
        mu, logvar = self.encode(x)

        all_log_weights = []

        for k_start in range(0, n_samples, batch_k):
            k_end = min(k_start + batch_k, n_samples)
            k_chunk = k_end - k_start

            # Expand for this chunk: (batch, latent_dim) -> (batch, k_chunk, latent_dim)
            mu_expanded = mu.unsqueeze(1).expand(-1, k_chunk, -1)
            logvar_expanded = logvar.unsqueeze(1).expand(-1, k_chunk, -1)

            # Sample z's for this chunk: (batch, k_chunk, latent_dim)
            std = torch.exp(0.5 * logvar_expanded)
            eps = torch.randn_like(std)
            z = mu_expanded + std * eps

            # Flatten for decoder: (batch * k_chunk, latent_dim)
            z_flat = z.view(batch_size * k_chunk, -1)
            x_recon_flat = self.decode(z_flat)

            # Expand x for this chunk: (batch * k_chunk, ...)
            x_expanded = (
                x.unsqueeze(1)
                .expand(-1, k_chunk, *x.shape[1:])
                .reshape(batch_size * k_chunk, *x.shape[1:])
            )

            # Compute log probs in batch
            log_prob_x_given_z = self._log_prob_x_given_z(
                x_expanded, x_recon_flat
            )  # (batch * k_chunk,)
            log_prob_prior = self._log_prob_prior(z_flat)  # (batch * k_chunk,)
            log_prob_q = self._log_prob_variational_posterior(
                z_flat,
                mu.unsqueeze(1)
                .expand(-1, k_chunk, -1)
                .reshape(batch_size * k_chunk, -1),
                logvar.unsqueeze(1)
                .expand(-1, k_chunk, -1)
                .reshape(batch_size * k_chunk, -1),
            )  # (batch * k_chunk,)

            # Importance weight: p(x|z) * p(z) / q(z|x)
            log_weights = log_prob_x_given_z + log_prob_prior - log_prob_q
            log_weights = log_weights.view(batch_size, k_chunk)
            all_log_weights.append(log_weights)

        # Concatenate all chunks: (batch, n_samples)
        all_log_weights = torch.cat(all_log_weights, dim=1)

        # log p(x) ≈ logsumexp(log_weights) - log(n_samples)
        return torch.logsumexp(all_log_weights, dim=1) - torch.log(
            torch.tensor(n_samples, dtype=x.dtype, device=x.device)
        )

    def evaluate(self, datamodule, train_data, train_labels, output_dir, cfg) -> None:
        from src.eval.vae import evaluate_vae

        evaluate_vae(self, datamodule, train_data, train_labels, output_dir, cfg)


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
        data_dim: int,
        architecture: dict,
        latent_dim: int,
        k: int = 5,
        lr: float = 1e-3,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__(
            data_dim=data_dim,
            architecture=architecture,
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
        return {
            "loss": loss,
            "iwae_bound": torch.mean(iwae_bound),
            "decoder_var": self.decoder_var,
        }

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
