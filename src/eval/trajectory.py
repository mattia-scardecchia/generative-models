"""Evaluation for trajectory-based models (FlowMatching, Diffusion)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from omegaconf import DictConfig

from src.eval.metrics import sliced_wasserstein, compute_sample_metrics
from src.eval.plots import plot_samples_panel, plot_snapshots, plot_trajectories


def evaluate_trajectory_model(
    model, datamodule, train_data: torch.Tensor, train_labels: torch.Tensor,
    output_dir: Path, cfg: DictConfig,
    trajectories: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Evaluate a trajectory-based model (diffusion, flow matching).

    Returns (trajectories, swd) tuple.
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

    # --- Compute Sliced Wasserstein Distance ---
    swd = sliced_wasserstein(train_data, samples)
    print(f"Sliced Wasserstein Distance: {swd:.4f}")
    if wandb.run is not None:
        wandb.log({"eval/swd": swd})

    # --- Samples plot (real vs generated) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_samples_panel(fig, (axes[0], axes[1]), datamodule, train_np, samples_np, labels_np)
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=150, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"eval/samples": wandb.Image(fig)})
    plt.close()

    # --- Trajectory plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_trajectories(fig, ax, datamodule, traj_np, n_trajectories=50)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"eval/trajectories": wandb.Image(fig)})
    plt.close()

    return trajectories, swd


def format_latex_table(results: dict[int, dict[str, float]]) -> str:
    """Format step sweep results as a booktabs-style LaTeX table.

    Best values per column are bolded (lowest for distances, highest for
    coverage, closest to 0.5 for 1-NNA).
    """
    metric_names = list(next(iter(results.values())).keys())
    step_counts = sorted(results.keys())

    # Determine best value per metric
    best = {}
    for name in metric_names:
        values = [results[s][name] for s in step_counts]
        if name == "COV":
            best[name] = max(values)
        elif name == "1-NNA":
            best[name] = min(values, key=lambda v: abs(v - 0.5))
        else:  # SWD, MMD, Energy — lower is better
            best[name] = min(values)

    # Column headers with arrows indicating direction
    arrows = {"SWD": r"$\downarrow$", "MMD": r"$\downarrow$", "Energy": r"$\downarrow$",
              "COV": r"$\uparrow$", "1-NNA": r"$\rightarrow .5$"}
    headers = [f"{name} {arrows.get(name, '')}" for name in metric_names]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample quality vs.\ number of sampling steps}",
        r"\begin{tabular}{r " + "c" * len(metric_names) + "}",
        r"\toprule",
        "Steps & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for s in step_counts:
        cells = []
        for name in metric_names:
            val = results[s][name]
            fmt = f"{val:.4f}" if name not in ("COV", "1-NNA") else f"{val:.3f}"
            is_best = (name == "1-NNA" and abs(val - 0.5) == abs(best[name] - 0.5)) or \
                      (name != "1-NNA" and val == best[name])
            cells.append(rf"\textbf{{{fmt}}}" if is_best else fmt)
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    preamble = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
    ]
    postamble = [r"\end{document}"]

    return "\n".join(preamble + [""] + lines + [""] + postamble)


def evaluate_step_sweep(
    model,
    train_data: torch.Tensor,
    output_dir: Path,
    step_counts: list[int] | None = None,
    n_samples: int | None = None,
) -> dict[int, dict[str, float]]:
    """Evaluate sample quality across different numbers of sampling steps.

    For each step count, generates samples and computes all metrics.
    Saves results as a LaTeX table to output_dir/step_sweep.tex.

    Args:
        model: A generative model with sample(n, n_steps=...).
        train_data: Reference data in normalized model space.
        output_dir: Directory to save the .tex file.
        step_counts: Step counts to evaluate. Defaults to [5, 10, 20, 50, 100, 200].
        n_samples: Number of samples to generate per step count.
                   Defaults to len(train_data).

    Returns:
        Dict mapping step count to metric dict.
    """
    if step_counts is None:
        step_counts = [5, 10, 20, 50, 100, 200]

    n_eval = n_samples if n_samples is not None else len(train_data)
    ref_data = train_data[:n_eval]
    results: dict[int, dict[str, float]] = {}

    print("\nStep sweep evaluation:")
    for n_steps in step_counts:
        samples = model.sample(n_eval, n_steps=n_steps)
        if isinstance(samples, tuple):
            samples = samples[0]
        metrics = compute_sample_metrics(ref_data, samples)
        results[n_steps] = metrics
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(f"  Steps={n_steps:>4d}:  {' '.join(parts)}")

    table = format_latex_table(results)
    tex_path = output_dir / "step_sweep.tex"
    tex_path.write_text(table)
    print(f"\nLaTeX table saved to {tex_path}")

    # Log to wandb
    if wandb.run is not None:
        metric_names = list(next(iter(results.values())).keys())
        wandb_table = wandb.Table(columns=["steps"] + metric_names)
        for n_steps in sorted(results.keys()):
            row = [n_steps] + [results[n_steps][m] for m in metric_names]
            wandb_table.add_data(*row)
        wandb.log({"eval/step_sweep": wandb_table})

    return results
