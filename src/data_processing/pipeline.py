import os
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_data import get_processed_data

@hydra.main(config_path="../../configs", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = to_absolute_path(cfg.main.processed_data_path)
    if os.path.exists(file_path):
        df_final = pd.read_csv(file_path)
        print("\n➡️  Preprocessed Data Already exists\n")

    else:
        df_final = pd.read_csv(to_absolute_path(cfg.main.raw_data_path))
        #df_final = df_final.head(cfg.preprocessing.data_limit).copy()

        print("\n➡️  Preprocessed Data dont exists, starting the preprocessing:\n")
        df_final = get_processed_data(cfg, df_final, is_training=True)
