import torch

from src.architectures import TimeConditionedNet
from src.models.base import GenerativeModel


class FlowMatching(GenerativeModel):
    """Flow matching with optimal transport conditional paths.

    Learns a velocity field v_θ(x_t, t) that transports samples from a
    standard Gaussian prior (t=0) to the data distribution (t=1) along
    straight-line (OT) conditional paths.

    Training loss:
        E_{t, x_0, x_1} ||v_θ(x_t, t) - (x_1 - x_0)||²
    where x_t = (1-t)*x_0 + t*x_1, x_0 ~ N(0,I), x_1 = data.

    Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023)
    """

    def __init__(
        self,
        data_dim: int,
        architecture: dict,
        time_embed_dim: int = 32,
        sigma_min: float = 0.0,
        n_sampling_steps: int = 100,
        lr: float = 1e-3,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.velocity_field = TimeConditionedNet(data_dim, architecture, time_embed_dim)
        self.data_dim = data_dim
        self.sigma_min = sigma_min
        self.n_sampling_steps = n_sampling_steps

    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        t = torch.rand(batch_size, 1, device=x.device)
        x_0 = torch.randn_like(x)
        x_t = (1 - t) * x_0 + t * x
        u_t = x - x_0
        v = self.velocity_field(x_t, t.squeeze(-1))
        return torch.mean((v - u_t) ** 2)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, n_samples: int, return_trajectories: bool = False, n_steps: int | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        n_steps = n_steps if n_steps is not None else self.n_sampling_steps
        x = torch.randn(n_samples, self.data_dim, device=self.device)
        ts = torch.linspace(0, 1, n_steps + 1, device=self.device)
        dt = ts[1] - ts[0]

        if return_trajectories:
            trajectory = [x.clone()]

        for t in ts[:-1]:
            t_batch = t.expand(n_samples)
            x = x + dt * self.velocity_field(x, t_batch)
            if return_trajectories:
                trajectory.append(x.clone())

        if return_trajectories:
            return x, torch.stack(trajectory)  # (n, d), (steps+1, n, d)
        return x

    def evaluate(self, datamodule, train_data, train_labels, output_dir, cfg) -> dict:
        from src.eval.trajectory import evaluate_trajectory_model, evaluate_step_sweep

        eval_cfg = cfg.get("evaluate", {})
        _, swd = evaluate_trajectory_model(self, datamodule, train_data, train_labels, output_dir, cfg)

        # --- Step sweep ---
        step_counts = eval_cfg.get("step_counts", None)
        n_eval_samples = eval_cfg.get("n_samples", None)
        sweep_results = evaluate_step_sweep(
            self, train_data, output_dir, step_counts=step_counts, n_samples=n_eval_samples,
        )

        print(f"Saved plots to {output_dir}")
        return {"swd": swd, "step_sweep": sweep_results}
