import os
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple, Any
from omegaconf import DictConfig

class CategoryTransformer:
    def __init__(self, config: DictConfig):
        """
        Initialize the CategoryTransformer with Hydra configuration.

        ### Parameters:
        - **config (DictConfig)**: Configuration object from Hydra.
        """
        self.replace_threshold = config.category_transformer.replace_threshold
        self.categories_path = config.category_transformer.categories_path
        self.problem_mapping_file = config.category_transformer.problem_mapping_file
        self.cause_mapping_file = config.category_transformer.cause_mapping_file

        # Ensure the directory exists
        os.makedirs(self.categories_path, exist_ok=True)

        # Initialize MultiLabelBinarizers
        self.mlb_components = MultiLabelBinarizer()
        self.mlb_type = MultiLabelBinarizer()
        self.mlb_cause = MultiLabelBinarizer()

        # Load category mappings
        self.problem_mapping = self._load_json(self.problem_mapping_file)
        self.cause_mapping = self._load_json(self.cause_mapping_file)
    
    def _load_json(self, filename: str) -> Any:
        """Load JSON file and return as a dictionary."""
        try:
            with open(f"{self.categories_path}{filename}", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def _save_json(self, data: Any, filename: str) -> None:
        """Save dictionary to a JSON file."""
        with open(f"{self.categories_path}{filename}", "w") as file:
            json.dump(data, file)

    def transform_categories(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        
        """
        Transform the 'pieces', 'type_of_problem', and 'cause' columns into binary categories.

        ### Parameters
        - **df (pd.DataFrame)**: DataFrame.
        - **is_training (bool)**: Flag to determine if it's training (to save the categories).
        """
        
        # Transform 'components' into a list
        df['components'] = df['components'].apply(lambda x: x.split(",") if pd.notna(x) else [])
        components_exploded = df['components'].explode().value_counts()

        # Identify components to replace
        components_to_replace = components_exploded[components_exploded < self.replace_threshold].index.tolist()
        df['components'] = df['components'].apply(
            lambda x: ['UNKNOWN OR OTHER' if component in components_to_replace else component for component in x]
        )

        # Transform categories into binary representation
        y_components = self.mlb_components.fit_transform(df['components'].tolist())
        y_type = self.mlb_type.fit_transform(df['problem_type'].tolist())
        y_cause = self.mlb_cause.fit_transform(df['problem_cause'].tolist())

        # Add binary columns to DataFrame
        df['components_binary'] = list(y_components)
        df['problem_type_binary'] = list(y_type)
        df['cause_binary'] = list(y_cause)

        if is_training:
            # Save category mappings
            categories = {
                'components': self.mlb_components.classes_.tolist(),
                'type_problem': self.mlb_type.classes_.tolist(),
                'cause': self.mlb_cause.classes_.tolist()
            }
            self._save_json(categories['components'], "components_classes.json")
            self._save_json(categories['type_problem'], "type_classes.json")
            self._save_json(categories['cause'], "cause_classes.json")

        return df
    
    def extract_pieces_and_problems(self, text: str) -> Tuple[str, str]:
        """
        Extracts all pieces and problem types from the given text.
        
        ### Parameters:
        - **text (str)**: The summary text.
        
        """
        # Clean text to lower case for easier matching
        text = text.lower()

       # Detect matching problems
        detected_problems = [
            general_problem for general_problem, variations in self.problem_mapping.items()
            if any(var in text for var in variations)
        ]
        problem_result = list(set(detected_problems)) if detected_problems else ["undefined"]

        # Detect matching causes
        detected_causes = [
            general_cause for general_cause, variations in self.cause_mapping.items()
            if any(var in text for var in variations)
        ]
        cause_result = list(set(detected_causes)) if detected_causes else ["undefined"]

        return problem_result, cause_result