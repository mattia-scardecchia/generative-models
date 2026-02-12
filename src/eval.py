"""Evaluation and sample generation script.

Usage:
    python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
    python src/eval.py ckpt_path=/path/to/checkpoint.ckpt n_samples=2000 logger=none
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided. Usage: python src/eval.py ckpt_path=<path>")

    # Instantiate data and model, then load trained weights
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    n_samples = cfg.get("n_samples", 1000)
    output_dir = Path.cwd()  # Hydra sets cwd to output dir

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


if __name__ == "__main__":
    main()
