"""Evaluation for trajectory-based models (FlowMatching, Diffusion)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from src.eval.plots import plot_samples_panel, plot_snapshots, plot_trajectories


def evaluate_trajectory_model(
    model, datamodule, train_data: torch.Tensor, train_labels: torch.Tensor,
    output_dir: Path, cfg: DictConfig,
    trajectories: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a trajectory-based model (diffusion, flow matching).

    Returns the trajectories tensor for use by callers (e.g. combined plots).
    """
    train_np = train_data.numpy()
    labels_np = train_labels.numpy()
    n_eval = len(train_data)

    # --- Generate samples with trajectories (unless pre-computed) ---
    if trajectories is None:
        samples, trajectories = model.sample(n_eval, return_trajectories=True)
    else:
        samples = trajectories[-1]
    samples_np = samples.numpy()
    traj_np = trajectories.numpy()  # (steps+1, n, data_dim)

    # --- Samples plot (real vs generated) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_samples_panel(fig, (axes[0], axes[1]), datamodule, train_np, samples_np, labels_np)
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Trajectory plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_trajectories(fig, ax, datamodule, traj_np, n_trajectories=50)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()

    return trajectories
