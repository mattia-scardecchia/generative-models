"""PyTorch Lightning callbacks for generative model training."""

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from src.eval.plots import plot_samples_panel


class SampleLoggerCallback(pl.Callback):
    """Log real-vs-generated sample plots to wandb during training.

    Works with any model that implements `sample(n_samples)`.
    """

    def __init__(self, every_n_epochs: int = 10, n_samples: int = 512):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if wandb.run is None:
            return

        datamodule = trainer.datamodule

        # Collect real training data
        real_data = []
        for x, _ in datamodule.train_dataloader():
            real_data.append(x)
            if sum(r.shape[0] for r in real_data) >= self.n_samples:
                break
        real_np = torch.cat(real_data, dim=0)[: self.n_samples].numpy()

        # Generate samples
        with torch.no_grad():
            samples = pl_module.sample(self.n_samples)
        samples_np = samples.cpu().numpy()

        # Plot and log
        matplotlib.use("Agg")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_samples_panel(fig, (axes[0], axes[1]), datamodule, real_np, samples_np)
        fig.suptitle(f"Epoch {trainer.current_epoch + 1}", fontsize=14)
        plt.tight_layout()
        wandb.log({"samples": wandb.Image(fig)})
        plt.close(fig)
