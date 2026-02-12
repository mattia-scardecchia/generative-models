"""Usage:
    python scripts/train.py
    python scripts/train.py trainer=debug logger=none
    python scripts/train.py model.beta=4.0 data.noise=0.1
    python scripts/train.py --multirun hydra/launcher=submitit_slurm model.beta=0.1,0.5,1.0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import train


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    print(OmegaConf.to_yaml(cfg))
    return train(cfg)


if __name__ == "__main__":
    main()
