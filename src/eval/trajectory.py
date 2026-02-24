"""Evaluation for trajectory-based models (FlowMatching, Diffusion)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from src.eval.plots import plot_samples_panel, plot_snapshots, plot_trajectories


def evaluate_trajectory_model(
    model, datamodule, train_data: torch.Tensor, train_labels: torch.Tensor,
    output_dir: Path, cfg: DictConfig,
) -> None:
    train_np = train_data.numpy()
    labels_np = train_labels.numpy()
    n_eval = len(train_data)

    # --- Generate samples with trajectories ---
    samples, trajectories = model.sample(n_eval, return_trajectories=True)
    samples_np = samples.numpy()
    traj_np = trajectories.numpy()  # (steps+1, n, data_dim)

    # --- Samples plot (real vs generated) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_samples_panel(fig, (axes[0], axes[1]), datamodule, train_np, samples_np, labels_np)
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples plot to {output_dir / 'samples.png'}")

    # --- Snapshots plot ---
    n_timesteps = 6
    fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4))
    plot_snapshots(fig, list(axes), datamodule, traj_np, n_timesteps=n_timesteps)
    plt.tight_layout()
    plt.savefig(output_dir / "snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshots plot to {output_dir / 'snapshots.png'}")

    # --- Trajectory plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_trajectories(fig, ax, datamodule, traj_np, n_trajectories=50)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectories plot to {output_dir / 'trajectories.png'}")
