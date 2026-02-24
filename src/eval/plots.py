import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_reconstructions(
    fig: Figure,
    axes: tuple[Axes, Axes],
    datamodule,
    original: np.ndarray,
    reconstructed: np.ndarray,
    labels: np.ndarray | None = None,
) -> None:
    """Plot original data vs reconstructions (2 panels).

    Delegates to datamodule.plot_samples() for rendering, so the plot adapts
    to the dataset's dimensionality (2D scatter, image grid, etc.).
    """
    datamodule.plot_samples(axes[0], original, labels)
    axes[0].set_title("Original Data")

    datamodule.plot_samples(axes[1], reconstructed, labels)
    axes[1].set_title("Reconstructions (with decoder variance)")


def plot_latent_space(
    fig: Figure,
    axes: tuple[Axes, Axes, Axes],
    mu: np.ndarray,
    logvar: np.ndarray,
    labels: np.ndarray,
    z_prior: np.ndarray,
    posterior_samples: list[dict],
    class_names: list[str],
    latent_dim: int,
    n_plot: int = 100,
) -> None:
    """Plot latent space visualization (3 panels: means, prior, posteriors).

    Args:
        posterior_samples: List of dicts (one per class), each with:
            "z": ndarray (n_points * n_samples_per_point, D)
            "point_ids": ndarray mapping each sample to its originating datapoint
    """
    title_suffix = " (dims 0-1)" if latent_dim != 2 else ""
    n_classes = len(class_names)
    markers = ["o", "+", "^", "s", "D", "x"][:n_classes]
    marker_sizes = [20, 30, 20, 20, 20, 25][:n_classes]
    colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples,
                 plt.cm.Oranges, plt.cm.Greys][:n_classes]

    # --- Panel 1: Latent means colored by variance sum, marker per class ---
    variance_sum = logvar[:, :2].sum(axis=1)
    vmin, vmax = variance_sum.min(), variance_sum.max()

    sc = None
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        idx = np.where(mask)[0]
        np.random.shuffle(idx)
        idx = idx[:n_plot]

        extra = {}
        if markers[cls_idx] in ("+", "x"):
            extra["linewidths"] = 1.5

        sc = axes[0].scatter(
            mu[idx, 0], mu[idx, 1],
            c=variance_sum[idx], cmap="viridis",
            s=marker_sizes[cls_idx], alpha=0.7,
            marker=markers[cls_idx], vmin=vmin, vmax=vmax,
            label=class_names[cls_idx], **extra,
        )

    axes[0].set_title(f"Latent Means (\u03bc){title_suffix}")
    mu_range = max(mu[:, 0].max() - mu[:, 0].min(), mu[:, 1].max() - mu[:, 1].min(), 2.0)
    mu_center = [
        (mu[:, 0].max() + mu[:, 0].min()) / 2,
        (mu[:, 1].max() + mu[:, 1].min()) / 2,
    ]
    axes[0].set_xlim(mu_center[0] - mu_range * 0.6, mu_center[0] + mu_range * 0.6)
    axes[0].set_ylim(mu_center[1] - mu_range * 0.6, mu_center[1] + mu_range * 0.6)
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", markerscale=2)
    if sc is not None:
        plt.colorbar(sc, ax=axes[0], label="\u03a3 log \u03c3\u00b2")

    # --- Panel 2: Prior samples ---
    axes[1].scatter(z_prior[:, 0], z_prior[:, 1], c="steelblue", s=3, alpha=0.5)
    axes[1].set_title(f"Prior Samples{title_suffix}")
    axes[1].set_aspect("equal")

    # --- Panel 3: Posterior samples, colored per datapoint, marker per class ---
    for cls_idx, samples_dict in enumerate(posterior_samples):
        z = samples_dict["z"]
        point_ids = samples_dict["point_ids"]
        n_points = int(point_ids.max()) + 1
        colors = colormaps[cls_idx](np.linspace(0.4, 0.9, n_points))

        extra = {}
        s = 15
        if markers[cls_idx] in ("+", "x"):
            extra["linewidths"] = 1.2
            s = 25

        for pt in range(n_points):
            pt_mask = point_ids == pt
            axes[2].scatter(
                z[pt_mask, 0], z[pt_mask, 1],
                c=[colors[pt]], s=s, alpha=0.7,
                marker=markers[cls_idx], **extra,
            )

    for cls_idx in range(n_classes):
        extra = {}
        if markers[cls_idx] in ("+", "x"):
            extra["linewidths"] = 1.5
        axes[2].scatter(
            [], [], c="gray", marker=markers[cls_idx],
            s=40, label=class_names[cls_idx], **extra,
        )
    axes[2].set_title(f"Posterior Samples{title_suffix}")
    axes[2].set_aspect("equal")
    axes[2].legend(loc="upper right", markerscale=1.5)


def plot_samples_panel(
    fig: Figure,
    axes: tuple[Axes, Axes],
    datamodule,
    real_data: np.ndarray,
    generated_data: np.ndarray,
    labels: np.ndarray | None = None,
) -> None:
    """Plot real data vs generated samples (2 panels)."""
    datamodule.plot_samples(axes[0], real_data, labels)
    axes[0].set_title("Training Data")

    datamodule.plot_samples(axes[1], generated_data)
    axes[1].set_title("Generated Samples")


def plot_snapshots(
    fig: Figure,
    axes: list[Axes],
    datamodule,
    trajectories: np.ndarray,
    n_timesteps: int = 6,
) -> None:
    """Plot point cloud snapshots at evenly-spaced time steps.

    Args:
        trajectories: Array of shape (total_steps, n_samples, data_dim).
        n_timesteps: Number of snapshots to show (including t=0 and t=1).
    """
    total_steps = trajectories.shape[0]
    step_indices = np.linspace(0, total_steps - 1, n_timesteps, dtype=int)

    for ax, step_idx in zip(axes, step_indices):
        t_val = step_idx / (total_steps - 1)
        snapshot = trajectories[step_idx]
        datamodule.plot_samples(ax, snapshot)
        ax.set_title(f"t = {t_val:.2f}")


def plot_trajectories(
    fig: Figure,
    ax: Axes,
    datamodule,
    trajectories: np.ndarray,
    n_trajectories: int = 50,
) -> None:
    """Plot ODE trajectories from noise to data as time-colored lines.

    Args:
        trajectories: Array of shape (total_steps, n_samples, data_dim).
        n_trajectories: Number of random trajectories to plot.
    """
    total_steps, n_samples, _ = trajectories.shape

    # Project all steps to 2D for visualization
    projected = np.stack([datamodule.project_to_viz(trajectories[s]) for s in range(total_steps)])

    indices = np.random.choice(n_samples, size=min(n_trajectories, n_samples), replace=False)
    t_vals = np.linspace(0, 1, total_steps)
    cmap = plt.cm.viridis

    for idx in indices:
        path = projected[:, idx, :]  # (total_steps, 2)
        for s in range(total_steps - 1):
            ax.plot(
                path[s:s+2, 0], path[s:s+2, 1],
                color=cmap(t_vals[s]), linewidth=0.5, alpha=0.6,
            )

    # Mark start (noise) and end (data)
    ax.scatter(projected[0, indices, 0], projected[0, indices, 1], c="blue", s=8, zorder=5, label="t=0 (noise)")
    ax.scatter(projected[-1, indices, 0], projected[-1, indices, 1], c="red", s=8, zorder=5, label="t=1 (data)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("ODE Trajectories")

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="t")


def plot_schedule(
    fig: Figure,
    axes: tuple[Axes, Axes],
    noise_schedule,
) -> None:
    """Plot noise schedule diagnostics (2 panels: coefficients and log SNR)."""
    t = torch.linspace(0, 1, 200)
    alpha = noise_schedule.alpha(t).numpy()
    sigma = noise_schedule.sigma(t).numpy()
    t_np = t.numpy()

    # Panel 1: alpha_t and sigma_t vs t
    axes[0].plot(t_np, alpha, label=r"$\alpha_t$", linewidth=2)
    axes[0].plot(t_np, sigma, label=r"$\sigma_t$", linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Noise Schedule")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: log SNR vs t
    snr = alpha ** 2 / np.clip(sigma ** 2, 1e-10, None)
    log_snr = np.log10(snr)
    axes[1].plot(t_np, log_snr, linewidth=2, color="tab:purple")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1, label="SNR = 1")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\log_{10}$ SNR")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


def plot_forward_process(
    fig: Figure,
    axes: list[Axes],
    datamodule,
    data: np.ndarray,
    noise_schedule,
    n_timesteps: int = 6,
) -> None:
    """Plot training data noised at increasing noise levels."""
    t_vals = np.linspace(0, 1, n_timesteps)
    x_0 = torch.tensor(data, dtype=torch.float32)
    noise = torch.randn_like(x_0)

    for ax, t_val in zip(axes, t_vals):
        t = torch.full((x_0.shape[0], 1), t_val)
        x_t = noise_schedule.q_sample(x_0, t, noise).numpy()
        datamodule.plot_samples(ax, x_t)
        ax.set_title(f"t = {t_val:.1f}")


def plot_convergence(
    fig: Figure,
    axes: tuple[Axes, Axes],
    elbo_sample_counts: list[int],
    elbo_estimates: list[float],
    ll_sample_counts: list[int],
    ll_estimates: list[float],
) -> None:
    """Plot ELBO and log-likelihood convergence curves (2 panels)."""
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
