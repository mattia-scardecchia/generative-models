"""Training entry point.

Usage:
    python src/train.py
    python src/train.py trainer=debug logger=none
    python src/train.py model.beta=4.0 model.lr=5e-4
    python src/train.py --multirun hydra/launcher=submitit_slurm model.beta=0.1,0.5,1.0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    loggers = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers if loggers else False,
        callbacks=callbacks if callbacks else None,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    best_score = trainer.callback_metrics.get("val/loss")
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
