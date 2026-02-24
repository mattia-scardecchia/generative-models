"""Evaluation for Energy-Based Models."""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import wandb
from omegaconf import DictConfig

from src.eval.ebm_plots import plot_energy_landscape, plot_ebm_samples
from src.eval.metrics import log_sample_metrics
from src.eval.plots import plot_samples_panel


def evaluate_ebm(
    model, datamodule, train_data: torch.Tensor, train_labels: torch.Tensor,
    output_dir: Path, cfg: DictConfig,
) -> None:
    train_np = train_data.numpy()
    labels_np = train_labels.numpy()
    data_2d = datamodule.project_to_viz(train_np)

    # --- Energy landscape + generated samples ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_energy_landscape(axes[0], model, datamodule, data_2d)

    n_gen_samples = cfg.get("n_gen_samples", 1000)
    gen_steps = cfg.get("gen_langevin_steps", None)
    with torch.no_grad():
        generated_tensor = model.sample(n_gen_samples, n_steps=gen_steps).cpu()
    generated = generated_tensor.numpy()
    gen_2d = datamodule.project_to_viz(generated)

    plot_ebm_samples(axes[1], datamodule, data_2d, gen_2d, labels_np)

    plt.tight_layout()
    plt.savefig(output_dir / "ebm_evaluation.png", dpi=200, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"eval/ebm_evaluation": wandb.Image(fig)})
    plt.close()
    print(f"Saved EBM evaluation plot to {output_dir / 'ebm_evaluation.png'}")

    # --- Samples plot (real vs generated) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_samples_panel(fig, (axes[0], axes[1]), datamodule, train_np, generated, labels_np)
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=150, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"eval/samples": wandb.Image(fig)})
    plt.close()

    # --- Energy statistics ---
    with torch.no_grad():
        energy_train = model.energy_net(train_data).numpy()
    print(f"\nEnergy statistics on training data:")
    print("-" * 45)
    print(f"  Mean:   {energy_train.mean():.4f}")
    print(f"  Std:    {energy_train.std():.4f}")
    print(f"  Min:    {energy_train.min():.4f}")
    print(f"  Max:    {energy_train.max():.4f}")

    if wandb.run is not None:
        wandb.log({
            "eval/energy_mean": energy_train.mean(),
            "eval/energy_std": energy_train.std(),
            "eval/energy_min": energy_train.min(),
            "eval/energy_max": energy_train.max(),
        })

    # --- Sample quality metrics ---
    log_sample_metrics(train_data[:n_gen_samples], generated_tensor)
