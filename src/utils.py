import warnings
import numpy as np
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
    output_dir = Path(HydraConfig.get().runtime.output_dir)

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

    # Plot: original data | reconstructions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Original data
    axes[0].scatter(train_np[:, 0], train_np[:, 1], c=labels_np, cmap="coolwarm", s=3, alpha=0.5)
    axes[0].set_title("Original Data")
    axes[0].set_aspect("equal")

    # Panel 2: Reconstructions (sample z from q(z|x), decode, sample from p(x|z))
    with torch.no_grad():
        z_sampled = model.reparameterize(mu, logvar)  # sample from variational posterior
        x_recon_mean = model.decode(z_sampled)  # decode to get mean
        # Sample from decoder distribution p(x|z) = N(f(z), σ²I)
        decoder_std = model.decoder_var.sqrt()
        x_recon_sampled = x_recon_mean + decoder_std * torch.randn_like(x_recon_mean)
    recon_sampled_np = x_recon_sampled.numpy()
    axes[1].scatter(recon_sampled_np[:, 0], recon_sampled_np[:, 1], c=labels_np, cmap="coolwarm", s=3, alpha=0.5)
    axes[1].set_title("Reconstructions (with decoder variance)")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plot_path = output_dir / "reconstructions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstructions plot to {plot_path}")

    # Latent space visualization
    latent_dim = model.latent_dim
    if latent_dim != 2:
        warnings.warn(
            f"Latent dimension is {latent_dim}, but latent space plots require 2D. "
            "Visualizing only the first two dimensions."
        )
    latent_title_suffix = " (dims 0-1)" if latent_dim != 2 else ""

    mu_np = mu.numpy()
    logvar_np = logvar.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Latent means (mu) with color = sum of variance, different markers for moons
    variance_sum = logvar_np[:, :2].sum(axis=1)  # sum of variance across first 2 dims
    moon_0_mask = labels_np == 0
    moon_1_mask = labels_np == 1

    # Subsample for clearer visualization
    n_plot = 100
    moon_0_idx = np.where(moon_0_mask)[0]
    moon_1_idx = np.where(moon_1_mask)[0]
    np.random.shuffle(moon_0_idx)
    np.random.shuffle(moon_1_idx)
    plot_idx_0 = moon_0_idx[:n_plot]
    plot_idx_1 = moon_1_idx[:n_plot]

    # Use consistent color scale for both markers
    vmin, vmax = variance_sum.min(), variance_sum.max()

    sc0 = axes[0].scatter(
        mu_np[plot_idx_0, 0], mu_np[plot_idx_0, 1],
        c=variance_sum[plot_idx_0], cmap="viridis", s=20, alpha=0.7,
        marker="o", vmin=vmin, vmax=vmax, label="Moon 0"
    )
    axes[0].scatter(
        mu_np[plot_idx_1, 0], mu_np[plot_idx_1, 1],
        c=variance_sum[plot_idx_1], cmap="viridis", s=30, alpha=0.7,
        marker="+", linewidths=1.5, vmin=vmin, vmax=vmax, label="Moon 1"
    )
    axes[0].set_title(f"Latent Means (μ){latent_title_suffix}")
    # Set axis limits with some padding to prevent over-shrinking
    mu_range = max(mu_np[:, 0].max() - mu_np[:, 0].min(), mu_np[:, 1].max() - mu_np[:, 1].min(), 2.0)
    mu_center = [(mu_np[:, 0].max() + mu_np[:, 0].min()) / 2, (mu_np[:, 1].max() + mu_np[:, 1].min()) / 2]
    axes[0].set_xlim(mu_center[0] - mu_range * 0.6, mu_center[0] + mu_range * 0.6)
    axes[0].set_ylim(mu_center[1] - mu_range * 0.6, mu_center[1] + mu_range * 0.6)
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", markerscale=2)
    plt.colorbar(sc0, ax=axes[0], label="Σ log σ²")

    # Panel 2: Samples from prior in latent space
    n_prior_samples = 1000
    z_prior = torch.randn(n_prior_samples, latent_dim)
    z_prior_np = z_prior.numpy()
    axes[1].scatter(z_prior_np[:, 0], z_prior_np[:, 1], c="steelblue", s=3, alpha=0.5)
    axes[1].set_title(f"Prior Samples{latent_title_suffix}")
    axes[1].set_aspect("equal")

    # Panel 3: Samples from variational posteriors with different markers for moons
    n_datapoints_per_moon = 5
    n_latent_samples_per_point = 100

    # Select datapoints from each moon
    moon_0_indices = torch.where(train_labels == 0)[0]
    moon_1_indices = torch.where(train_labels == 1)[0]
    selected_0 = moon_0_indices[torch.randperm(len(moon_0_indices))[:n_datapoints_per_moon]]
    selected_1 = moon_1_indices[torch.randperm(len(moon_1_indices))[:n_datapoints_per_moon]]

    colors_0 = plt.cm.Blues(torch.linspace(0.4, 0.9, n_datapoints_per_moon).numpy())
    colors_1 = plt.cm.Reds(torch.linspace(0.4, 0.9, n_datapoints_per_moon).numpy())

    with torch.no_grad():
        # Moon 0 - circles
        for i, idx in enumerate(selected_0):
            x_i = train_data[idx:idx+1]
            mu_i, logvar_i = model.encode(x_i)
            mu_i_expanded = mu_i.expand(n_latent_samples_per_point, -1)
            logvar_i_expanded = logvar_i.expand(n_latent_samples_per_point, -1)
            z_samples = model.reparameterize(mu_i_expanded, logvar_i_expanded)
            z_samples_np = z_samples.numpy()
            axes[2].scatter(z_samples_np[:, 0], z_samples_np[:, 1], c=[colors_0[i]], s=15, alpha=0.7, marker="o")

        # Moon 1 - plus signs
        for i, idx in enumerate(selected_1):
            x_i = train_data[idx:idx+1]
            mu_i, logvar_i = model.encode(x_i)
            mu_i_expanded = mu_i.expand(n_latent_samples_per_point, -1)
            logvar_i_expanded = logvar_i.expand(n_latent_samples_per_point, -1)
            z_samples = model.reparameterize(mu_i_expanded, logvar_i_expanded)
            z_samples_np = z_samples.numpy()
            axes[2].scatter(z_samples_np[:, 0], z_samples_np[:, 1], c=[colors_1[i]], s=25, alpha=0.7, marker="+", linewidths=1.2)

    axes[2].set_title(f"Posterior Samples{latent_title_suffix}")
    axes[2].set_aspect("equal")
    # Add legend for moon markers
    axes[2].scatter([], [], c="steelblue", marker="o", s=40, label="Moon 0")
    axes[2].scatter([], [], c="indianred", marker="+", s=60, linewidths=1.5, label="Moon 1")
    axes[2].legend(loc="upper right", markerscale=1.5)

    plt.tight_layout()
    latent_path = output_dir / "latent_space.png"
    plt.savefig(latent_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved latent space plot to {latent_path}")

    # Estimate ELBO and log-likelihood on a subset of data
    n_eval_samples = cfg.get("n_eval_samples", 500)
    eval_data = train_data[:n_eval_samples]

    with torch.no_grad():
        # ELBO with increasing number of MC samples
        print("\nELBO estimates (Monte Carlo):")
        print("-" * 45)
        elbo_sample_counts = cfg.get("elbo_sample_counts", [1, 10, 100, 1000, 10000])
        elbo_estimates = []
        for k in elbo_sample_counts:
            elbo_values = model.elbo(eval_data, n_samples=k)
            mean_elbo = elbo_values.mean().item()
            elbo_estimates.append(mean_elbo)
            print(f"  K={k:5d}: {mean_elbo:.4f}")

        # Log-likelihood with increasing number of importance samples
        print("\nLog-likelihood estimates (importance sampling):")
        print("-" * 45)
        ll_sample_counts = cfg.get("ll_sample_counts", [1, 10, 100, 1000, 10000])
        ll_estimates = []
        for k in ll_sample_counts:
            with torch.no_grad():
                ll_values = model.importance_sampling_log_prob_estimate(eval_data, n_samples=k)
            mean_ll = ll_values.mean().item()
            ll_estimates.append(mean_ll)
            print(f"  K={k:5d}: {mean_ll:.4f}")

        # KL divergences and gap analysis
        eval_mu, eval_logvar = model.encode(eval_data)
        kl_to_prior = model._kl_divergence(eval_mu, eval_logvar).mean().item()
        kl_to_posterior = ll_estimates[-1] - elbo_estimates[-1]  # log p(x) - ELBO = KL(q||p_posterior)

        print("\nGap analysis:")
        print("-" * 45)
        print(f"  KL(q(z|x) || p(z)):        {kl_to_prior:.4f}  (encoder → prior)")
        print(f"  KL(q(z|x) || p(z|x)):      {kl_to_posterior:.4f}  (encoder → true posterior)")

    # Plot ELBO and log-likelihood convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(elbo_sample_counts, elbo_estimates, marker="o", linewidth=2, markersize=6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of MC samples (K)")
    axes[0].set_ylabel("Estimated ELBO")
    axes[0].set_title("ELBO Convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ll_sample_counts, ll_estimates, marker="o", linewidth=2, markersize=6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of importance samples (K)")
    axes[1].set_ylabel("Estimated log p(x)")
    axes[1].set_title("Log-Likelihood Convergence")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    convergence_path = output_dir / "convergence.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved convergence plot to {convergence_path}")
