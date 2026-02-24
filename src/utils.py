from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer


def instantiate_loggers(logger_cfg: DictConfig | None) -> list:
    loggers = []
    if not logger_cfg or not isinstance(logger_cfg, DictConfig):
        return loggers
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list:
    callbacks = []
    if not callbacks_cfg or not isinstance(callbacks_cfg, DictConfig):
        return callbacks
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def train(cfg: DictConfig) -> float | None:
    """Run training from a resolved Hydra config.

    Returns the best validation loss (useful for hyperparameter optimization).
    """
    import torch

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    with open_dict(cfg):
        cfg.model.data_dim = datamodule.data_dim
    model = hydra.utils.instantiate(cfg.model)

    loggers = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers if loggers else False,
        callbacks=callbacks if callbacks else None,
    )

    datamodule.setup()
    x_all = torch.cat([x for x, _ in datamodule.train_dataloader()], dim=0)
    print(f"[DEBUG] Training data — mean: {x_all.mean():.4f}, std: {x_all.std():.4f}, "
          f"min: {x_all.min():.4f}, max: {x_all.max():.4f}")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    best_score = trainer.callback_metrics.get("val/loss")

    # Run evaluation after training
    evaluate(cfg, model=model, datamodule=datamodule)

    return float(best_score) if best_score is not None else None


def evaluate(cfg: DictConfig, model=None, datamodule=None) -> None:
    """Generate evaluation plots and metrics.

    Can be called standalone (with ckpt_path in cfg) or after training.
    Delegates to model.evaluate() for model-specific evaluation.
    """
    import torch
    import matplotlib

    matplotlib.use("Agg")

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # --- Setup: load model/data if needed ---
    if datamodule is None:
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()

    if model is None:
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when model is not passed directly")
        with open_dict(cfg):
            cfg.model.data_dim = datamodule.data_dim
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # --- Collect training data ---
    train_data, train_labels = [], []
    for x, y in datamodule.train_dataloader():
        train_data.append(x)
        train_labels.append(y)
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    model.evaluate(datamodule, train_data, train_labels, output_dir, cfg)
