import torch
from dataclasses import dataclass


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
