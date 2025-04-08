import hydra
from omegaconf import DictConfig

from utils import check_and_create_env, load_json, convert_json_to_temp_yaml

@hydra.main(config_path="../../configs", config_name="temp.yaml", version_base=None)
def main(cfg: DictConfig):
    check_and_create_env(cfg)

if __name__ == '__main__':
    json_file = load_json('./configs/main.json')
    convert_json_to_temp_yaml(json_file)

    main()