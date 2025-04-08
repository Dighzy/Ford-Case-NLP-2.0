import json
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.data_processing.scripts import CategoryTransformer, EmbeddingExtractor, TextProcessor
from utils_scripts import utils


@hydra.main(config_path="../../configs", config_name="temp.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = to_absolute_path(cfg.main.processed_data_path)
    if os.path.exists(file_path):
        df_final = pd.read_csv(file_path)
        print("\n➡️  Preprocessed Data Already exists\n")

    else:
        df_final = pd.read_csv(to_absolute_path(cfg.main.raw_data_path))
        df_final = df_final.head(cfg.preprocessing.data_limit).copy()
        print(df_final)

        print("\n➡️  Preprocessed Data dont exists, starting the preprocessing:\n")
        df_final = get_processed_data(cfg, df_final, is_training=cfg.preprocessing.is_training)

def get_processed_data(cfg: DictConfig, df_final: pd.DataFrame, is_training: bool =False) -> pd.DataFrame:
    """
    ## Pipeline to process all data.

    ### Parameters:
    - **df_final (pd.DataFrame)**: The data to process.
    - **is_training (bool)**: Flag to determine if it's training.
    - **path (str)**: Path to safe the dataframe into a csv file.
    """
    # Getting classes
    text_processor = TextProcessor()
    embedding_extractor = EmbeddingExtractor(cfg)
    category_tranformer =  CategoryTransformer(cfg)

    # Text preprocessing function
    df_final['processed_summary'] = df_final[cfg.preprocessing.summary_column].apply(text_processor.preprocess_text)
    # Get embeddings
    df_final['summary_embedding'] = df_final[cfg.preprocessing.summary_column].apply(embedding_extractor.get_model_embedding)


    if is_training:
        df_final[['problem_type', 'problem_cause']] = df_final[cfg.preprocessing.summary_column].apply(lambda x: category_tranformer.extract_pieces_and_problems(x) if isinstance(x, str) else ('undefined', 'undefined', 'undefined')).apply(pd.Series)
        df_final = category_tranformer.transform_categories(df_final, is_training=is_training)

        # Saving my final data to use in the model
        df_save = df_final.copy()
        columns_save = [cfg.preprocessing.summary_column, 'summary_embedding', 'components_binary', 'problem_type_binary', 'cause_binary']
        for coluna in columns_save:
            df_save[coluna] = df_save[coluna].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, (np.ndarray, list)) else x)
        df_save[columns_save].to_csv(cfg.main.processed_data_path, index=False)

    return df_final

if __name__ == '__main__':
    json_file = utils.load_json('./configs/main.json')
    utils.convert_json_to_temp_yaml(json_file)

    main()
