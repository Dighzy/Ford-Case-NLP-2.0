import hydra
from omegaconf import DictConfig

from utils import check_and_create_env

@hydra.main(config_path="../../configs", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    check_and_create_env(cfg)

if __name__ == '__main__':
    main()