import pandas as pd
import numpy as np
import tensorflow as tf
import re
import nltk
import torch
import joblib
import json
import os
import ast

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


# Baixar recursos necessários
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function


class TextProcessor:
    def __init__(self):
        """
        Load the lemmatizer and set the stop_words
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Apply preprocess into the text, such as removing  special characters, punctuation and stop words. 
        Apply Tokenization and Lemmatization.

        ## Parameters
        - text (str): The text column in the data frame.
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
    def __init__(self):
        """
        Load the pre-trained BERT model and tokenizer
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_model_embedding(self, text):
        """
        Converts text into a BERT embedding

        ## Parameters
        - text (str): The text column in the data frame.
        """
        if pd.isna(text):
            return torch.zeros(512)  
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


    def get_sentiment_and_count(self, df):
        """
        Get sentiment in the text and count the number of words and characters in the text.

        ## Parameters
        - df (pd.DataFrame): DataFrame.
        """
        # Calculating the word count and character count for the processed summary
        df["word_count"] = df["processed_summary"].apply(lambda x: len(x.split())) 
        df["char_count"] = df["processed_summary"].apply(len)  

        # Calculating the sentiment polarity for each processed summary
        df['sentiment'] = df['processed_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
        # 1.0 → Positive, 0.0 → Neutral, -1.0 → Negative

        return df
    
    def get_categories(self, df):
        """
        Transform the 'components_list' column into general categories.

        ## Parameters
        - df (pd.DataFrame): DataFrame.
        """
        
        # Mapping the categories problemns 
        with open("models/params/categories.json", "r") as file:
            category_map = json.load(file)
        
        df["components_list"] = df["components"].apply(lambda x: x.split(",") if pd.notna(x) else [])
        
        df["general_category"] = df["components_list"].apply(
            lambda x: list(set(category_map.get(i, "Other Problems") for i in x))
        )

        return df
    
    def transform_categories(self, df):
        """
        Transform the 'general_category' column into binary categories.

        ## Parameters
        - df (pd.DataFrame): DataFrame.
        """
        
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df['general_category'])
        df['category_binary'] = list(y)

        # Save the classes to a JSON file for later reconstruction
        categories = mlb.classes_.tolist()

        with open("models/params/category_classes.json", "w") as file:
            json.dump(categories, file)

        return df
        

class TopicsProcessor:  
    def __init__(self):
        """
        Initializes the TopicsProcessor class.
        """
        self.topic_model = None
        self.tfidf = None

    def fit_tfidf(self, df):
        """
        Train the TF-IDF with the train data.
        
        ## Parameters
        - df (pd.DataFrame): DataFrame 
        """
        self.tfidf = TfidfVectorizer()
        tfidf_matrix = self.tfidf.fit_transform(df["processed_summary"])
        
        joblib.dump(self.tfidf, "models/tfidf_vectorizer.pkl")
        return tfidf_matrix

    def fit_topic_model(self, tfidf_matrix):
        """
        Train the topic model
        """
        self.topic_model = LatentDirichletAllocation(n_components=2)
        topic_features = self.topic_model.fit_transform(tfidf_matrix)
        
        joblib.dump(self.topic_model, "models/lda_model.pkl")
        return topic_features

    def transform_for_prediction(self, df):
        """
        Loads the pre-trained topic model.
        
        ## Parameters
        - df (pd.DataFrame): DataFrame 
        """
        self.topic_model = joblib.load("models/lda_model.pkl")
        self.tfidf = joblib.load("models/tfidf_vectorizer.pkl")

        tfidf_matrix = self.tfidf.transform(df["processed_summary"])
        topic_features = self.topic_model.transform(tfidf_matrix) 
        return topic_features

    def get_topics(self, df, is_training=False):
        """
        Get the topics

        ## Parameters
        - is_training (bool) : If True, fits the topic model with training data. Otherwise, 
                              loads the pre-trained topic model for making predictions.
        """

        if is_training:
            tfidf_matrix = self.fit_tfidf(df)
            topic_features = self.fit_topic_model(tfidf_matrix)
        else:
            topic_features = self.transform_for_prediction(df)

        topic_df = pd.DataFrame(topic_features, columns=[f"topic_{i+1}" for i in range(topic_features.shape[1])])
        df = pd.concat([df, topic_df], axis=1)

        return df


class DataProcessor:
    def __init__(self, is_training: bool = False) -> None:
        """
        Initializes the DataProcessor class.

        ## Parameters
        - is_training (bool) : If True, fits the scaler with training data. Otherwise, 
                              loads the pre-trained scaler for making predictions.
        """
        if is_training:
            self.scaler = StandardScaler()
        else:
            scaler_params_file = "models/params/scaler_params.json"
            if not os.path.exists(scaler_params_file):
                raise FileNotFoundError(f"Scaler parameters file not found: {scaler_params_file}")

            with open(scaler_params_file, "r") as file:
                scaler_params = json.load(file)

            if "mean" not in scaler_params or "scale" not in scaler_params:
                raise ValueError("Missing 'mean' or 'scale' in scaler parameters file")

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_params["mean"])
            self.scaler.scale_ = np.array(scaler_params["scale"])

    def process_train_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes the training data and fits the scaler.

        ## Parameters
        - df (pd.DataFrame): DataFrame 
        """
        #embeddings = np.array(df["summary_embedding"].tolist())
        df['model_embedding'] = np.array(df["model_embedding"].tolist())

        additional_features = df[['model_embedding', 'word_count', 'char_count', 'sentiment', 'topic_1', 'topic_2']]
        
        additional_features_scaled = self.scaler.fit_transform(additional_features)

        # Save the scaler parameters for later use in prediction
        self.save_scaler_params()

        return additional_features_scaled

    def save_scaler_params(self) -> None:
        """
        Saves the scaler parameters (mean and scale) to a JSON file for future use.
        """
        scaler_params = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist()
        }
        with open("models/params/scaler_params.json", "w") as file:
            json.dump(scaler_params, file)

    def process_predict_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes the prediction data and uses the pre-trained scaler.

        ## Parameters
        - df (pd.DataFrame): DataFrame.
        """
        #embeddings = np.array(df["summary_embedding"].tolist())
        df["model_embedding"] = df["model_embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['model_embedding'] = np.array(df["model_embedding"].tolist())

        additional_features = df[['model_embedding', 'word_count', 'char_count', 'sentiment', 'topic_1', 'topic_2']]
        
        additional_features_scaled = self.scaler.transform(additional_features)
        print(additional_features_scaled)

        return df["summary_embedding"], additional_features_scaled


if __name__ == "__main__":
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    df_final = pd.read_csv('data/raw/full_data_2020_2025_FORD.csv')
    df_final = df_final.head(5).copy()

    # Getting classes
    text_processor = TextProcessor()
    feature_extractor = FeatureExtractor()
    topic_processor = TopicsProcessor()
    data_processor = DataProcessor()


    # Getting model types
    models = df_final['Model'].unique().tolist()
    model_data = {
        "models":models
    }
    #with open('models/params/models_params.json', 'w') as f:
        #json.dump(model_data, f)
    
    # Text preprocessing function
    df_final['processed_summary'] = df_final['summary'].apply(text_processor.preprocess_text)

    df_final = feature_extractor.get_categories(df_final)
    df_final = feature_extractor.transform_categories(df_final)

    print(df_final[['category_binary','general_category','components_list']].head())
    

    # Get embeddings
    df_final['summary_embedding'] = df_final['summary'].apply(feature_extractor.get_model_embedding)
    df_final['model_embedding'] = df_final['Model'].apply(feature_extractor.get_model_embedding)

    print("embeddings:", df_final['summary_embedding'])

    # Get sentiment and topics
    df_final = feature_extractor.get_sentiment_and_count(df_final)
    df_final = topic_processor.get_topics(df_final)

    # Processing and normalazing the embedding and additional_features
    embeddings, additional_features = data_processor.process_predict_data(df_final)

    # Verify the shapes of the data
    print("embeddings shape:", embeddings.shape) 
    print("additional_features shape:", additional_features.shape) 

    print("embeddings:", embeddings)
    print("additional_features:", additional_features)
