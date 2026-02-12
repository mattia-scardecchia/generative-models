import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BetaVAE(pl.LightningModule):
    """Beta-VAE with configurable encoder and decoder.

    The encoder is a model-agnostic feature extractor. This module adds its own
    fc_mu and fc_logvar projection heads on top of the encoder output.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        beta: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr

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

    def _compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decode(z)
