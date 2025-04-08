import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from omegaconf import DictConfig


class EmbeddingExtractor:
    def __init__(self, cfg: DictConfig) -> None:
        """
        # Load the pre-trained RoBERTa model and tokenizer
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.embedding.name)
        self.model = RobertaModel.from_pretrained(cfg.embedding.name)
        self.max_length = cfg.embedding.max_length
        self.padding = cfg.embedding.padding
        self.truncaton = cfg.embedding.truncation

    def get_model_embedding(self, text: str) -> any:
        """
        Converts text into a BERT embedding

        ### Parameters
        - **text (str)**: The text column in the data frame.
        """
        if pd.isna(text):
            return torch.zeros(self.max_length)

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=self.padding, truncation=self.truncaton, max_length=self.max_length
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()