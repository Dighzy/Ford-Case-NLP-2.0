import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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