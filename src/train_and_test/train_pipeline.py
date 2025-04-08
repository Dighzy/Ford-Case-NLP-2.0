import os
import json
import ast
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.process_data import get_processed_data

from train_and_test.train import get_model_tuner  
from utils_scripts import utils

@hydra.main(config_path="../../configs", config_name="temp.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = to_absolute_path(cfg.main.processed_data_path)
    if os.path.exists(file_path):
        df_final = pd.read_csv(file_path)
        print("\n➡️  Preprocessed Data Already exists, starting the training process:\n")
    else:
        df_final = pd.read_csv(to_absolute_path(cfg.main.raw_data_path))
        df_final = df_final.head(cfg.preprocessing.data_limit).copy()

        print("\n➡️  Preprocessed Data dont exists, starting the preprocessing:\n")
        df_final = get_processed_data(cfg, df_final, is_training=cfg.train.is_training)
    
    embeddings = np.array(df_final["summary_embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    components_binary = np.array(df_final["components_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    problems_type_binary = np.array(df_final["problem_type_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    problems_cause_binary = np.array(df_final["cause_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())

    X_emb_train, X_emb_val, y_components_train, y_components_val, y_type_train, y_type_val, y_cause_train, y_cause_val = train_test_split(
        embeddings, components_binary, problems_type_binary, problems_cause_binary, test_size=cfg.train.test_size, random_state=42
    )

    tuning_dict = OmegaConf.to_container(cfg.tuning, resolve=True) 
    tuner = get_model_tuner(tuning_dict, X_emb_train, n_components_classes=y_components_train.shape[1], n_type_classes=y_type_train.shape[1], n_cause_classes=y_cause_train.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.train.early_stopping_patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=to_absolute_path(cfg.train.best_model_path), monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    tuner.search(
        {"embedding_input": X_emb_train},
        {"components_output": y_components_train, "type_output": y_type_train, "cause_output": y_cause_train},
        validation_data=(
            {"embedding_input": X_emb_val},
            {"components_output": y_components_val, "type_output": y_type_val, "cause_output": y_cause_val}
        ),
        epochs=int(cfg.tuning.epochs),
        batch_size=int(cfg.tuning.batch_size),
        callbacks=[early_stopping]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps_dict = {"Best tuner hyperparameters": best_hps.values}
    hps_file = to_absolute_path(cfg.train.hps_file)
    with open(hps_file, 'w') as f:
        json.dump(best_hps_dict, f, indent=4)
    print(f"Best hyperparameters saved to {hps_file}")
    print(best_hps_dict)

    model = tuner.hypermodel.build(best_hps)
    
    history = model.fit(
        {"embedding_input": X_emb_train},
        {"components_output": y_components_train, "type_output": y_type_train, "cause_output": y_cause_train},
        validation_data=(
            {"embedding_input": X_emb_val},
            {"components_output": y_components_val, "type_output": y_type_val, "cause_output": y_cause_val}
        ),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    best_epoch = np.argmin(history.history['val_loss'])
    history_file = to_absolute_path(cfg.train.history_file)
    history_dict = {key: np.array(value).tolist() for key, value in history.history.items()}
    with open(history_file, "w") as file:
        json.dump(history_dict, file)


    metrics_dict = {"Training and Validation Metrics": {metric: float(values[best_epoch]) for metric, values in history.history.items()}}
    metrics_file = to_absolute_path(cfg.train.metrics_file)
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    json_file = utils.load_json('./configs/main.json')
    utils.convert_json_to_temp_yaml(json_file)

    main()
