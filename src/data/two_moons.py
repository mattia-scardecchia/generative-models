import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons


class TwoMoonsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_samples: int = 10000,
        noise: float = 0.05,
        val_fraction: float = 0.2,
        batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.noise = noise
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str | None = None):
        if hasattr(self, "train_dataset"):
            return

        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.seed)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)

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
