"""Base class for all generative models."""

import warnings

import torch
import pytorch_lightning as pl


class GenerativeModel(pl.LightningModule):
    """Base class for all generative models.

    Provides shared optimizer/scheduler configuration via self.hparams.
    Subclasses must call self.save_hyperparameters() in __init__ and include
    lr, optimizer_config, and scheduler_config in their constructor signatures.
    """

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 1e-3)
        optimizer_config = self.hparams.get("optimizer_config", None)
        scheduler_config = self.hparams.get("scheduler_config", None)

        if optimizer_config is not None:
            opt_type = optimizer_config.get("type", "Adam")
            opt_params = {k: v for k, v in optimizer_config.items() if k != "type"}
            opt_params.setdefault("lr", lr)
            optimizer_cls = getattr(torch.optim, opt_type)
            optimizer = optimizer_cls(self.parameters(), **opt_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if scheduler_config is not None:
            sched_type = scheduler_config.get("type", "StepLR")
            sched_params = {k: v for k, v in scheduler_config.items() if k != "type"}
            scheduler_cls = getattr(torch.optim.lr_scheduler, sched_type)
            scheduler = scheduler_cls(optimizer, **sched_params)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Generate samples from the model. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement sample()")

    def evaluate(self, datamodule, train_data, train_labels, output_dir, cfg) -> None:
        """Model-specific evaluation after training. Override in subclasses."""
        warnings.warn(f"{type(self).__name__} does not implement evaluate(). Skipping.")
