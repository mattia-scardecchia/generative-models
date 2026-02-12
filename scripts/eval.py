"""Usage:
    python scripts/eval.py ckpt_path=/path/to/checkpoint.ckpt
    python scripts/eval.py ckpt_path=/path/to/checkpoint.ckpt n_samples=2000 logger=none
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import evaluate


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    evaluate(cfg)


if __name__ == "__main__":
    main()
