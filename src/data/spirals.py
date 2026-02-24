import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class SpiralDataModule(pl.LightningDataModule):

    class_names = ["Spiral"]

    def __init__(
        self,
        n_samples: int = 10000,
        noise: float = 0.1,
        n_rotations: float = 3.0,
        val_fraction: float = 0.2,
        batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
        ambient_dim: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.noise = noise
        self.n_rotations = n_rotations
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.ambient_dim = ambient_dim
        self.embedding_matrix: torch.Tensor | None = None

    @property
    def data_dim(self) -> int:
        if self.ambient_dim is not None and self.ambient_dim > 2:
            return self.ambient_dim
        return 2

    def setup(self, stage: str | None = None):
        if hasattr(self, "train_dataset"):
            return

        rng = np.random.default_rng(self.seed)
        t = np.sqrt(rng.uniform(0, 1, self.n_samples)) * self.n_rotations * 2 * np.pi
        x = t * np.cos(t) + rng.normal(0, self.noise, self.n_samples)
        y = t * np.sin(t) + rng.normal(0, self.noise, self.n_samples)

        X = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)
        labels = torch.zeros(self.n_samples, dtype=torch.long)

        if self.ambient_dim is not None and self.ambient_dim > 2:
            gen = torch.Generator().manual_seed(self.seed)
            Q, _ = torch.linalg.qr(torch.randn(self.ambient_dim, self.ambient_dim, generator=gen))
            self.embedding_matrix = Q[:2, :]  # (2, ambient_dim)
            X = X @ self.embedding_matrix  # (n, 2) @ (2, ambient_dim) -> (n, ambient_dim)

        # Standardize after embedding so the model sees unit-variance data
        self.data_mean = X.mean(dim=0)
        self.data_std = X.std()
        X = (X - self.data_mean) / self.data_std

        dataset = TensorDataset(X, labels)

        val_size = int(len(dataset) * self.val_fraction)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def project_to_viz(self, data: np.ndarray) -> np.ndarray:
        """Denormalize and project back to 2D for visualization."""
        data = data * self.data_std.numpy() + self.data_mean.numpy()
        if self.embedding_matrix is not None:
            E = self.embedding_matrix.numpy()
            data = data @ E.T  # (n, ambient_dim) @ (ambient_dim, 2) -> (n, 2)
        return data

    def plot_samples(self, ax, data: np.ndarray, labels: np.ndarray | None = None, **kwargs) -> None:
        """Plot data points as a 2D scatter."""
        data = self.project_to_viz(data)
        defaults = {"s": 3, "alpha": 0.5, "cmap": "coolwarm"}
        defaults.update(kwargs)
        ax.scatter(data[:, 0], data[:, 1], c=labels, **defaults)
        ax.set_aspect("equal")
