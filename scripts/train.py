"""Usage:
    python scripts/train.py --config-name=config/vae
    python scripts/train.py --config-name=config/diffusion data=spiral
    python scripts/train.py --config-name=config/flow_matching architecture=residual_mlp
    python scripts/train.py --config-name=config/ebm trainer=debug logger=none
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import train


@hydra.main(version_base="1.3", config_path="../configs", config_name="config/vae")
def main(cfg: DictConfig) -> float | None:
    print(OmegaConf.to_yaml(cfg))
    return train(cfg)


if __name__ == "__main__":
    main()
