import torch
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


@dataclass
class EvalMetrics:
    elbo_sample_counts: list[int]
    elbo_estimates: list[float]
    ll_sample_counts: list[int]
    ll_estimates: list[float]
    kl_to_prior: float
    kl_to_posterior: float


def compute_elbo_curve(model, data: torch.Tensor, sample_counts: list[int]) -> list[float]:
    """Compute ELBO estimates for increasing numbers of MC samples."""
    estimates = []
    with torch.no_grad():
        for k in sample_counts:
            elbo_values = model.elbo(data, n_samples=k)
            estimates.append(elbo_values.mean().item())
    return estimates


def compute_ll_curve(model, data: torch.Tensor, sample_counts: list[int]) -> list[float]:
    """Compute log-likelihood estimates for increasing numbers of importance samples."""
    estimates = []
    with torch.no_grad():
        for k in sample_counts:
            ll_values = model.importance_sampling_log_prob_estimate(data, n_samples=k)
            estimates.append(ll_values.mean().item())
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
    elbo_estimates = compute_elbo_curve(model, data, elbo_sample_counts)
    ll_estimates = compute_ll_curve(model, data, ll_sample_counts)

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
