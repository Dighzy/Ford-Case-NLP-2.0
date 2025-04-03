import re
import nltk
import torch
import json

import pandas as pd
import numpy as np
from typing import Tuple
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import RobertaTokenizer, RobertaModel

class TextProcessor:
    def __init__(self) -> None:
        """
        # Load the lemmatizer and set the stop_words
        """
        # Downloading the resources if necessary
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """
        - Apply preprocess into the text, such as removing  special characters, punctuation and stop words. 
        - Apply Tokenization and Lemmatization.

        ### Parameters
        - **text (str)**: The text column in the data frame.
        """
        # Remove special characters and punctuation
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)   
        text = text.lower()
        
        # Tokenization: Split the text into individual words
        words = word_tokenize(text)
        
        # Remove stopwords (common words like 'the', 'and', etc.)
        words = [word for word in words if word not in self.stop_words]
        
        # Lemmatization: Convert words to their base or root form (e.g., 'running' to 'run')
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

class FeatureExtractor:
    def __init__(self) -> None:
        """
        # Load the pre-trained RoBERTa model and tokenizer
        """
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large')

    def get_model_embedding(self, text: str) -> any:
        """
        Converts text into a BERT embedding

        ### Parameters
        - **text (str)**: The text column in the data frame.
        """
        if pd.isna(text):
            return torch.zeros(512)  
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
class CategoryTransformer:
    def transform_categories(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Transform the 'pieces', 'type_of_problem', and 'cause' columns into binary categories.

        ### Parameters
        - **df (pd.DataFrame)**: DataFrame.
        - **is_training (bool)**: Flag to determine if it's training (to save the categories).
        """
        
        # Initialize MultiLabelBinarizer for each category column
        mlb_components = MultiLabelBinarizer()
        mlb_type = MultiLabelBinarizer()
        mlb_cause = MultiLabelBinarizer()

        # Tranform the components class into a list
        df['components'] = df['components'].apply(lambda x: x.split(",") if pd.notna(x) else [])
        components_exploded = df['components'].explode().value_counts()

        # Replacing rare labels 
        components_to_replace = components_exploded[components_exploded < 200].index
        df['components'] = df['components'].apply(
            lambda x: ['UNKNOWN OR OTHER' if component in components_to_replace else component for component in x]
        )

        # Transform the columns into binary categories
        y_components = mlb_components.fit_transform(list(df['components']))
        y_type = mlb_type.fit_transform(list(df['problem_type']))
        y_cause = mlb_cause.fit_transform(list(df['problem_cause']))

        # Add the binary columns to the DataFrame
        df['components_binary'] = list(y_components)
        df['problem_type_binary'] = list(y_type)
        df['cause_binary'] = list(y_cause)

        if is_training:
            # Save the classes to JSON files for later reconstruction
            categories = {
                'components': mlb_components.classes_.tolist(),
                'type_problem': mlb_type.classes_.tolist(),
                'cause': mlb_cause.classes_.tolist()
            }

            # Save the categories in JSON files
            with open("models/params/categories/components_classes.json", "w") as file:
                json.dump(categories['components'], file)
            with open("models/params/categories/type_classes.json", "w") as file:
                json.dump(categories['type_problem'], file)
            with open("models/params/categories/cause_classes.json", "w") as file:
                json.dump(categories['cause'], file)

        return df
    
    def extract_pieces_and_problems(self, text: str) -> Tuple[str, str]:
        """
        Extracts all pieces and problem types from the given text.
        
        ### Parameters:
        - **text (str)**: The summary text.
        
        """
        # List of keywords that indicate types of problems or failures
        with open("models/params/categories/type_problems.json", "r") as file:
                    problem_mapping = json.load(file)

        with open("models/params/categories/cause_problems.json", "r") as file:
                    cause_mapping = json.load(file)

        # Clean text to lower case for easier matching
        text = text.lower()

        # Detect all matching problems
        detected_problems = [general_problem for general_problem, variations in problem_mapping.items()
                            if any(var in text for var in variations)]
        problem_result = list(set(detected_problems)) if detected_problems else ["undefined"]

        # Detect all matching causes
        detected_causes = [general_cause for general_cause, variations in cause_mapping.items()
                            if any(var in text for var in variations)]
        cause_result = list(set(detected_causes)) if detected_causes else ["undefined"]

        return problem_result, cause_result

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
    feature_extractor = FeatureExtractor()
    category_tranformer =  CategoryTransformer()

    # Text preprocessing function
    df_final['processed_summary'] = df_final['summary'].apply(text_processor.preprocess_text)
    # Get embeddings
    df_final['summary_embedding'] = df_final['summary'].apply(feature_extractor.get_model_embedding)


    if is_training:
        df_final[['problem_type', 'problem_cause']] = df_final['summary'].apply(lambda x: category_tranformer.extract_pieces_and_problems(x) if isinstance(x, str) else ('undefined', 'undefined', 'undefined')).apply(pd.Series)
        df_final = category_tranformer.transform_categories(df_final, is_training=is_training)

        # Saving my final data to use in the model
        df_save = df_final.copy()
        columns_save = ['summary', 'summary_embedding', 'components_binary', 'problem_type_binary', 'cause_binary']
        for coluna in columns_save:
            df_save[coluna] = df_save[coluna].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, (np.ndarray, list)) else x)
        df_save[columns_save].to_csv(cfg.main.processed_data_path, index=False)

    return df_final
