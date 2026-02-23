import warnings

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
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

    # Run evaluation after training
    evaluate(cfg, model=model, datamodule=datamodule)

    return float(best_score) if best_score is not None else None


def evaluate(cfg: DictConfig, model=None, datamodule=None) -> None:
    """Generate evaluation plots and metrics.

    Can be called standalone (with ckpt_path in cfg) or after training.
    """
    import torch
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pathlib import Path

    from src.eval.metrics import compute_eval_metrics
    from src.eval.plots import plot_reconstructions, plot_latent_space, plot_convergence

    matplotlib.use("Agg")

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # --- Setup: load model/data if needed ---
    if model is None:
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when model is not passed directly")
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    if datamodule is None:
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()

    model.eval()
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # --- Collect training data ---
    train_data, train_labels = [], []
    for x, y in datamodule.train_dataloader():
        train_data.append(x)
        train_labels.append(y)
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    train_np = train_data.numpy()
    labels_np = train_labels.numpy()

    # --- Reconstruction plot ---
    with torch.no_grad():
        _, mu, logvar = model(train_data)
        z_sampled = model.reparameterize(mu, logvar)
        x_recon_mean = model.decode(z_sampled)
        decoder_std = model.decoder_var.sqrt()
        x_recon_sampled = x_recon_mean + decoder_std * torch.randn_like(x_recon_mean)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_reconstructions(
        fig, (axes[0], axes[1]), datamodule,
        train_np, x_recon_sampled.numpy(), labels_np,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstructions plot to {output_dir / 'reconstructions.png'}")

    # --- Latent space plot ---
    if model.latent_dim != 2:
        warnings.warn(
            f"Latent dimension is {model.latent_dim}, but latent space plots require 2D. "
            "Visualizing only the first two dimensions."
        )

    mu_np = mu.numpy()
    logvar_np = logvar.numpy()
    z_prior = torch.randn(1000, model.latent_dim).numpy()

    class_names = getattr(datamodule, "class_names",
                          [f"Class {i}" for i in range(int(labels_np.max()) + 1)])
    n_classes = len(class_names)
    n_datapoints_per_class = 5
    n_latent_samples_per_point = 100

    posterior_samples = []
    with torch.no_grad():
        for cls_idx in range(n_classes):
            cls_indices = torch.where(train_labels == cls_idx)[0]
            selected = cls_indices[torch.randperm(len(cls_indices))[:n_datapoints_per_class]]
            z_list, id_list = [], []
            for pt_id, idx in enumerate(selected):
                x_i = train_data[idx:idx+1]
                mu_i, logvar_i = model.encode(x_i)
                mu_i_exp = mu_i.expand(n_latent_samples_per_point, -1)
                logvar_i_exp = logvar_i.expand(n_latent_samples_per_point, -1)
                z = model.reparameterize(mu_i_exp, logvar_i_exp)
                z_list.append(z.numpy())
                id_list.append(np.full(n_latent_samples_per_point, pt_id))
            posterior_samples.append({
                "z": np.concatenate(z_list, axis=0),
                "point_ids": np.concatenate(id_list, axis=0),
            })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_latent_space(
        fig, (axes[0], axes[1], axes[2]),
        mu_np, logvar_np, labels_np, z_prior,
        posterior_samples, class_names, model.latent_dim,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "latent_space.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved latent space plot to {output_dir / 'latent_space.png'}")

    # --- Metrics ---
    n_eval_samples = cfg.get("n_eval_samples", 500)
    eval_data = train_data[:n_eval_samples]
    elbo_sample_counts = cfg.get("elbo_sample_counts", [1, 10, 100, 1000])
    ll_sample_counts = cfg.get("ll_sample_counts", [1, 10, 100, 1000])

    metrics = compute_eval_metrics(model, eval_data, elbo_sample_counts, ll_sample_counts)

    print("\nELBO estimates (Monte Carlo):")
    print("-" * 45)
    for k, val in zip(metrics.elbo_sample_counts, metrics.elbo_estimates):
        print(f"  K={k:5d}: {val:.4f}")

    print("\nLog-likelihood estimates (importance sampling):")
    print("-" * 45)
    for k, val in zip(metrics.ll_sample_counts, metrics.ll_estimates):
        print(f"  K={k:5d}: {val:.4f}")

    print("\nGap analysis:")
    print("-" * 45)
    print(f"  KL(q(z|x) || p(z)):        {metrics.kl_to_prior:.4f}  (encoder -> prior)")
    print(f"  KL(q(z|x) || p(z|x)):      {metrics.kl_to_posterior:.4f}  (encoder -> true posterior)")

    # --- Convergence plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_convergence(
        fig, (axes[0], axes[1]),
        metrics.elbo_sample_counts, metrics.elbo_estimates,
        metrics.ll_sample_counts, metrics.ll_estimates,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved convergence plot to {output_dir / 'convergence.png'}")
