import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_energy_landscape(
    ax: Axes,
    model,
    datamodule,
    data_2d: np.ndarray,
    grid_resolution: int = 200,
    padding: float = 0.5,
) -> None:
    """Plot 2D energy landscape as filled contour with data points overlaid.

    Plots -E(x) (proportional to log-density) so high regions = high probability.
    For ambient_dim > 2, grid points are lifted back to ambient space via the
    embedding matrix before evaluating energy.
    """
    x_min, x_max = data_2d[:, 0].min() - padding, data_2d[:, 0].max() + padding
    y_min, y_max = data_2d[:, 1].min() - padding, data_2d[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # If data lives in ambient space, lift grid points back
    if hasattr(datamodule, "embedding_matrix") and datamodule.embedding_matrix is not None:
        E = datamodule.embedding_matrix.numpy()  # (2, ambient_dim)
        grid_ambient = grid_2d @ E
        grid_tensor = torch.tensor(grid_ambient, dtype=torch.float32)
    else:
        grid_tensor = torch.tensor(grid_2d, dtype=torch.float32)

    with torch.no_grad():
        energies = model.energy_net(grid_tensor.to(model.device)).cpu().numpy()

    energy_grid = energies.reshape(xx.shape)

    contour = ax.contourf(xx, yy, -energy_grid, levels=50, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="-E(x)")
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c="white", s=1, alpha=0.3, edgecolors="none")
    ax.set_title("Energy Landscape (-E(x) ∝ log p(x))")
    ax.set_aspect("equal")


def plot_ebm_samples(
    ax: Axes,
    datamodule,
    data_2d: np.ndarray,
    generated_2d: np.ndarray,
    labels: np.ndarray | None = None,
) -> None:
    """Plot training data vs generated samples on the same axes."""
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c="steelblue", s=3, alpha=0.3, label="Training data")
    ax.scatter(generated_2d[:, 0], generated_2d[:, 1], c="red", s=3, alpha=0.5, label="Generated")
    ax.set_title("Generated vs Training Samples")
    ax.set_aspect("equal")
    ax.legend(markerscale=3)
