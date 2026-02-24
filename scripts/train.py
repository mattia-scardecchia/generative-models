"""Usage:
    python scripts/train.py
    python scripts/train.py -cn diffusion
    python scripts/train.py -cn flow_matching architecture=residual_mlp
    python scripts/train.py -cn ebm trainer=debug logger=none
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import train


@hydra.main(version_base="1.3", config_path="../configs", config_name="diffusion")
def main(cfg: DictConfig) -> float | None:
    print(OmegaConf.to_yaml(cfg))
    return train(cfg)


if __name__ == "__main__":
    main()
