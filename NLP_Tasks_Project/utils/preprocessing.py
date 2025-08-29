"""
Text preprocessing utilities for NLP tasks
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import contractions
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks
    """
    
    def __init__(self, 
                 remove_html=True,
                 expand_contractions=True,
                 to_lowercase=True,
                 remove_punctuation=True,
                 remove_numbers=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 min_length=2):
        """
        Initialize the preprocessor with various options
        
        Args:
            remove_html (bool): Remove HTML tags
            expand_contractions (bool): Expand contractions (can't -> cannot)
            to_lowercase (bool): Convert to lowercase
            remove_punctuation (bool): Remove punctuation
            remove_numbers (bool): Remove numbers
            remove_stopwords (bool): Remove stopwords
            lemmatize (bool): Lemmatize words
            min_length (int): Minimum word length to keep
        """
        self.remove_html = remove_html
        self.expand_contractions = expand_contractions
        self.to_lowercase = to_lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_length = min_length
        
        # Initialize tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_html(self, text):
        """Remove HTML tags from text"""
        if not isinstance(text, str):
            return str(text)
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        return text
    
    def expand_contractions_func(self, text):
        """Expand contractions in text"""
        if not isinstance(text, str):
            return str(text)
        return contractions.fix(text)
    
    def remove_special_chars(self, text):
        """Remove special characters and extra whitespace"""
        if not isinstance(text, str):
            return str(text)
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_nums(self, text):
        """Remove numbers from text"""
        if not isinstance(text, str):
            return str(text)
        return re.sub(r'\d+', '', text)
    
    def preprocess_text(self, text):
        """
        Apply all preprocessing steps to a single text
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags
        if self.remove_html:
            text = self.clean_html(text)
        
        # Expand contractions
        if self.expand_contractions:
            text = self.expand_contractions_func(text)
        
        # Convert to lowercase
        if self.to_lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = self.remove_nums(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_special_chars(text)
        else:
            # Still remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= self.min_length]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column, new_column_name=None):
        """
        Preprocess text in a pandas DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text
            new_column_name (str): Name for the new preprocessed column
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text column
        """
        if new_column_name is None:
            new_column_name = f"{text_column}_cleaned"
        
        df = df.copy()
        df[new_column_name] = df[text_column].apply(self.preprocess_text)
        
        return df


def get_sentiment_polarity(text):
    """
    Get sentiment polarity using TextBlob
    
    Args:
        text (str): Input text
        
    Returns:
        float: Polarity score (-1 to 1)
    """
    if not isinstance(text, str):
        text = str(text)
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_word_frequencies(texts, top_n=20):
    """
    Get word frequencies from a list of texts
    
    Args:
        texts (list): List of text strings
        top_n (int): Number of top words to return
        
    Returns:
        pd.DataFrame: DataFrame with words and their frequencies
    """
    from collections import Counter
    
    # Combine all texts and split into words
    all_words = []
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Get top N words
    top_words = word_freq.most_common(top_n)
    
    return pd.DataFrame(top_words, columns=['word', 'frequency'])


def split_train_val_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Features
        y: Labels
        train_size (float): Proportion for training set
        val_size (float): Proportion for validation set
        test_size (float): Proportion for test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
