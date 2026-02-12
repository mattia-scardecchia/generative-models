import hydra
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
    """Generate samples and save evaluation plots.

    Can be called standalone with a checkpoint path, or directly after training
    with a model and datamodule already in memory.
    """
    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    from pathlib import Path

    matplotlib.use("Agg")

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load from checkpoint if model not provided
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

    n_samples = cfg.get("n_samples", 1000)
    output_dir = Path.cwd()

    # Generate samples from prior
    samples = model.sample(n_samples).cpu().numpy()

    # Collect training data for comparison
    train_data, train_labels = [], []
    for x, y in datamodule.train_dataloader():
        train_data.append(x)
        train_labels.append(y)
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Compute reconstructions
    with torch.no_grad():
        x_recon, mu, logvar = model(train_data)

    train_np = train_data.numpy()
    labels_np = train_labels.numpy()
    recon_np = x_recon.numpy()

    # Plot: original data | generated samples | reconstructions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(train_np[:, 0], train_np[:, 1], c=labels_np, cmap="coolwarm", s=3, alpha=0.5)
    axes[0].set_title("Original Data")
    axes[0].set_aspect("equal")

    axes[1].scatter(samples[:, 0], samples[:, 1], c="steelblue", s=3, alpha=0.5)
    axes[1].set_title(f"Generated Samples (n={n_samples})")
    axes[1].set_aspect("equal")

    axes[2].scatter(recon_np[:, 0], recon_np[:, 1], c=labels_np, cmap="coolwarm", s=3, alpha=0.5)
    axes[2].set_title("Reconstructions")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plot_path = output_dir / "evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")

    # Latent space visualization (only when latent_dim == 2)
    mu_np = mu.numpy()
    if mu_np.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(mu_np[:, 0], mu_np[:, 1], c=labels_np, cmap="coolwarm", s=3, alpha=0.5)
        ax.set_title("Latent Space (mu)")
        ax.set_aspect("equal")
        latent_path = output_dir / "latent_space.png"
        plt.savefig(latent_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved latent space plot to {latent_path}")

    # Estimate ELBO and log-likelihood on a subset of data
    n_eval_samples = cfg.get("n_eval_samples", 500)
    eval_data = train_data[:n_eval_samples]

    with torch.no_grad():
        # ELBO estimation
        elbo_values = model.elbo(eval_data, n_samples=10)
        mean_elbo = elbo_values.mean().item()
        print(f"\nELBO (n={n_eval_samples} samples, 10 MC samples): {mean_elbo:.4f}")

        # Log-likelihood with increasing number of importance samples
        print("\nLog-likelihood estimates (importance sampling):")
        print("-" * 45)
        ll_sample_counts = cfg.get("ll_sample_counts", [10, 50, 100, 500, 1000])
        ll_estimates = []
        for k in ll_sample_counts:
            ll_values = model.log_likelihood(eval_data, n_samples=k)
            mean_ll = ll_values.mean().item()
            ll_estimates.append(mean_ll)
            print(f"  K={k:5d}: {mean_ll:.4f}")

    # Plot log-likelihood convergence
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ll_sample_counts, ll_estimates, marker="o", linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of importance samples (K)")
    ax.set_ylabel("Estimated log p(x)")
    ax.set_title("Log-Likelihood Convergence")
    ax.grid(True, alpha=0.3)
    ll_path = output_dir / "log_likelihood_convergence.png"
    plt.savefig(ll_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved log-likelihood convergence plot to {ll_path}")
