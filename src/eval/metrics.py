from typing import Callable

import torch
import wandb
from dataclasses import dataclass


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_projections: int = 256) -> float:
    """Sliced Wasserstein Distance between two point clouds.

    Approximates Wasserstein distance by averaging 1D Wasserstein distances
    over random projections. O(n log n) per projection.

    Args:
        x: First point cloud (n, d)
        y: Second point cloud (m, d)
        n_projections: Number of random 1D projections

    Returns:
        Sliced Wasserstein distance (scalar)
    """
    d = x.shape[1]
    # Random unit vectors for projections
    projections = torch.randn(n_projections, d, device=x.device, dtype=x.dtype)
    projections = projections / projections.norm(dim=1, keepdim=True)

    # Project both point clouds
    x_proj = x @ projections.T  # (n, n_proj)
    y_proj = y @ projections.T  # (m, n_proj)

    # Sort along sample dimension
    x_sorted = x_proj.sort(dim=0).values
    y_sorted = y_proj.sort(dim=0).values

    # If different sizes, interpolate to match
    if x_sorted.shape[0] != y_sorted.shape[0]:
        n = max(x_sorted.shape[0], y_sorted.shape[0])
        x_sorted = torch.nn.functional.interpolate(
            x_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze(0).T
        y_sorted = torch.nn.functional.interpolate(
            y_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze(0).T

    # 1D Wasserstein is just L1 distance of sorted values
    return (x_sorted - y_sorted).abs().mean().item()


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    Uses the median heuristic for bandwidth selection if not provided.

    Args:
        x: First point cloud (n, d)
        y: Second point cloud (m, d)
        bandwidth: RBF bandwidth. If None, uses median of pairwise squared distances.

    Returns:
        MMD value (scalar)
    """
    if bandwidth is None:
        z = torch.cat([x, y], dim=0)
        dists_sq = torch.cdist(z, z).pow(2)
        mask = torch.triu(torch.ones_like(dists_sq, dtype=torch.bool), diagonal=1)
        bandwidth = dists_sq[mask].median().item()
        if bandwidth == 0:
            bandwidth = 1.0

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * bandwidth))

    n, m = x.shape[0], y.shape[0]
    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)

    # Unbiased estimator (exclude diagonal for within-set terms)
    mmd_sq = (
        (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
        + (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return max(mmd_sq.item(), 0.0) ** 0.5


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Energy distance between two point clouds.

    E(X, Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    Args:
        x: First point cloud (n, d)
        y: Second point cloud (m, d)

    Returns:
        Energy distance (scalar)
    """
    n, m = x.shape[0], y.shape[0]

    d_xy = torch.cdist(x, y).mean()

    d_xx = torch.cdist(x, x)
    d_xx = (d_xx.sum() - d_xx.diagonal().sum()) / (n * (n - 1))

    d_yy = torch.cdist(y, y)
    d_yy = (d_yy.sum() - d_yy.diagonal().sum()) / (m * (m - 1))

    e_dist = 2 * d_xy - d_xx - d_yy
    return max(e_dist.item(), 0.0) ** 0.5


def coverage(x_real: torch.Tensor, x_gen: torch.Tensor, k: int = 5) -> float:
    """Fraction of real samples with a generated neighbor within k-NN threshold.

    For each real sample, the threshold is its k-th nearest neighbor distance
    within the real set. A sample is "covered" if any generated sample falls
    within this radius.

    Args:
        x_real: Real data (n, d)
        x_gen: Generated data (m, d)
        k: Number of neighbors for threshold

    Returns:
        Coverage in [0, 1]
    """
    # k-NN distances within real data
    d_rr = torch.cdist(x_real, x_real)
    d_rr.fill_diagonal_(float("inf"))
    thresholds = d_rr.topk(k, largest=False).values[:, -1]  # (n,)

    # Minimum distance from each real sample to any generated sample
    d_rg = torch.cdist(x_real, x_gen)
    min_dists = d_rg.min(dim=1).values  # (n,)

    return (min_dists <= thresholds).float().mean().item()


def one_nn_accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    """1-Nearest Neighbor leave-one-out accuracy as a two-sample test.

    Combines x (label=0) and y (label=1), finds each point's nearest
    neighbor, and computes classification accuracy. 50% means the
    distributions are indistinguishable.

    Args:
        x: First point cloud (n, d)
        y: Second point cloud (m, d)

    Returns:
        1-NN accuracy in [0, 1]. Closer to 0.5 is better.
    """
    n, m = x.shape[0], y.shape[0]
    z = torch.cat([x, y], dim=0)
    labels = torch.cat([torch.zeros(n), torch.ones(m)]).to(x.device)

    dists = torch.cdist(z, z)
    dists.fill_diagonal_(float("inf"))

    nn_labels = labels[dists.argmin(dim=1)]
    return (nn_labels == labels).float().mean().item()


@torch.no_grad()
def compute_sample_metrics(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """Compute all sample quality metrics between real and generated data.

    This is the main entry point for evaluating any generative model's
    sample quality. All metrics operate on point clouds in the model's
    native space.

    Args:
        real: Real data (n, d)
        generated: Generated data (m, d)

    Returns:
        Dict with keys: SWD, MMD, Energy, COV, 1-NNA
    """
    return {
        "SWD": sliced_wasserstein(real, generated),
        "MMD": mmd_rbf(real, generated),
        "Energy": energy_distance(real, generated),
        "COV": coverage(real, generated),
        "1-NNA": one_nn_accuracy(real, generated),
    }


def log_sample_metrics(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """Compute sample quality metrics, print them, and log to wandb."""
    metrics = compute_sample_metrics(real, generated)

    print("\nSample quality metrics:")
    print("-" * 45)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    if wandb.run is not None:
        for name, val in metrics.items():
            wandb.log({f"eval/{name}": val})

    return metrics


# --- VAE-specific metrics ---


@dataclass
class EvalMetrics:
    elbo_sample_counts: list[int]
    elbo_estimates: list[float]
    ll_sample_counts: list[int]
    ll_estimates: list[float]
    kl_to_prior: float
    kl_to_posterior: float


def _compute_metric_curve(
    model_fn: Callable[[torch.Tensor, int], torch.Tensor],
    data: torch.Tensor,
    sample_counts: list[int],
    batch_size: int = 64,
) -> list[float]:
    """Compute a per-sample metric for increasing numbers of MC/IS samples.

    Args:
        model_fn: Callable (batch, n_samples) -> per-sample values of shape (batch_size,).
        data: Evaluation data.
        sample_counts: List of K values to evaluate.
        batch_size: Data batch size for memory-efficient evaluation.
    """
    estimates = []
    with torch.no_grad():
        for k in sample_counts:
            total = 0.0
            n = 0
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                values = model_fn(batch, k)
                total += values.sum().item()
                n += len(batch)
            estimates.append(total / n)
    return estimates


def compute_eval_metrics(
    model,
    data: torch.Tensor,
    elbo_sample_counts: list[int],
    ll_sample_counts: list[int],
) -> EvalMetrics:
    """Compute all evaluation metrics for a VAE-family model.

    The model must expose: elbo(), importance_sampling_log_prob_estimate(),
    encode(), and _kl_divergence().
    """
    elbo_estimates = _compute_metric_curve(
        lambda batch, k: model.elbo(batch, n_samples=k),
        data, elbo_sample_counts,
    )
    ll_estimates = _compute_metric_curve(
        lambda batch, k: model.importance_sampling_log_prob_estimate(batch, n_samples=k),
        data, ll_sample_counts,
    )

    with torch.no_grad():
        mu, logvar = model.encode(data)
        kl_to_prior = model._kl_divergence(mu, logvar).mean().item()

    kl_to_posterior = ll_estimates[-1] - elbo_estimates[-1]

    return EvalMetrics(
        elbo_sample_counts=elbo_sample_counts,
        elbo_estimates=elbo_estimates,
        ll_sample_counts=ll_sample_counts,
        ll_estimates=ll_estimates,
        kl_to_prior=kl_to_prior,
        kl_to_posterior=kl_to_posterior,
    )
