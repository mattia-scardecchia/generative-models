"""Evaluation for VAE-family models (BetaVAE, IWAE)."""

import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from omegaconf import DictConfig

from src.eval.metrics import compute_eval_metrics, log_sample_metrics
from src.eval.plots import plot_reconstructions, plot_latent_space, plot_convergence


def evaluate_vae(
    model, datamodule, train_data: torch.Tensor, train_labels: torch.Tensor,
    output_dir: Path, cfg: DictConfig,
) -> None:
    train_np = train_data.numpy()
    labels_np = train_labels.numpy()

    # --- Reconstruction plot ---
    eval_batch_size = cfg.get("eval_batch_size", 256)
    with torch.no_grad():
        mu_list, logvar_list, x_recon_list = [], [], []
        for i in range(0, len(train_data), eval_batch_size):
            batch = train_data[i : i + eval_batch_size]
            _, mu_batch, logvar_batch = model(batch)
            z_batch = model.reparameterize(mu_batch, logvar_batch)
            x_recon_batch = model.decode(z_batch)
            mu_list.append(mu_batch)
            logvar_list.append(logvar_batch)
            x_recon_list.append(x_recon_batch)
        mu = torch.cat(mu_list, dim=0)
        logvar = torch.cat(logvar_list, dim=0)
        x_recon_mean = torch.cat(x_recon_list, dim=0)
        decoder_std = model.decoder_var.sqrt()
        x_recon_sampled = x_recon_mean + decoder_std * torch.randn_like(x_recon_mean)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_reconstructions(
        fig, (axes[0], axes[1]), datamodule,
        train_np, x_recon_sampled.numpy(), labels_np,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructions.png", dpi=150, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"eval/reconstructions": wandb.Image(fig)})
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
    if wandb.run is not None:
        wandb.log({"eval/latent_space": wandb.Image(fig)})
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
    if wandb.run is not None:
        wandb.log({"eval/convergence": wandb.Image(fig)})
        wandb.log({
            "eval/kl_to_prior": metrics.kl_to_prior,
            "eval/kl_to_posterior": metrics.kl_to_posterior,
            "eval/elbo_K1": metrics.elbo_estimates[0],
            "eval/ll_K1000": metrics.ll_estimates[-1],
        })
    plt.close()
    print(f"\nSaved convergence plot to {output_dir / 'convergence.png'}")

    # --- Sample quality metrics ---
    n_gen_samples = cfg.get("evaluate", {}).get("n_samples", 1000)
    with torch.no_grad():
        generated = model.sample(n_gen_samples)
    log_sample_metrics(train_data[:n_gen_samples], generated)
