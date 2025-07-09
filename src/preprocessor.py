"""
Data preprocessing utilities for retail sentiment analysis
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple
import string
import warnings
warnings.filterwarnings('ignore')

class RetailTextPreprocessor:
    """Comprehensive text preprocessing for retail review analysis"""
    
    def __init__(self, download_nltk: bool = True):
        """
        Initialize preprocessor with NLTK resources
        
        Args:
            download_nltk: Whether to download required NLTK data
        """
        if download_nltk:
            self._download_nltk_data()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add retail-specific stop words
        self.stop_words.update([
            'product', 'item', 'buy', 'purchase', 'order', 'bought', 'get', 'got',
            'would', 'could', 'should', 'really', 'good', 'bad', 'nice', 'great',
            'amazon', 'seller', 'delivery', 'shipping', 'price', 'money'
        ])
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = [
            'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 
            'vader_lexicon', 'omw-1.4'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")
                
        # Additional fallback downloads
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"SSL workaround failed: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short and very long texts
        if len(text) < 10 or len(text) > 5000:
            return ""
        
        return text
    
    def remove_punctuation(self, text: str, keep_periods: bool = False) -> str:
        """
        Remove punctuation from text
        
        Args:
            text: Input text
            keep_periods: Whether to keep periods for sentence structure
            
        Returns:
            str: Text without punctuation
        """
        if keep_periods:
            punct = string.punctuation.replace('.', '')
        else:
            punct = string.punctuation
        
        return text.translate(str.maketrans('', '', punct))
    
    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character lemmatizer accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using WordNet
        
        Args:
            text: Input text
            
        Returns:
            str: Lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatized = []
        
        for token in tokens:
            if token.isalpha() and len(token) > 2:
                pos = self.get_wordnet_pos(token)
                lemmatized.append(self.lemmatizer.lemmatize(token, pos))
        
        return ' '.join(lemmatized)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            
        Returns:
            str: Text without stopwords
        """
        tokens = word_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in self.stop_words and len(token) > 2
        ]
        return ' '.join(filtered_tokens)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common contractions
        
        Args:
            text: Input text
            
        Returns:
            str: Text with expanded contractions
        """
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def correct_spelling(self, text: str) -> str:
        """
        Basic spell correction using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            str: Spell-corrected text
        """
        try:
            blob = TextBlob(text)
            return str(blob.correct())
        except:
            return text
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       lemmatize: bool = True,
                       remove_punct: bool = True,
                       expand_contractions: bool = True,
                       correct_spelling: bool = False) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            remove_punct: Whether to remove punctuation
            expand_contractions: Whether to expand contractions
            correct_spelling: Whether to correct spelling
            
        Returns:
            str: Preprocessed text
        """
        # Basic cleaning
        text = self.clean_text(text)
        if not text:
            return ""
        
        # Expand contractions
        if expand_contractions:
            text = self.expand_contractions(text)
        
        # Spell correction (optional, can be slow)
        if correct_spelling:
            text = self.correct_spelling(text)
        
        # Remove punctuation
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'review_text',
                           **kwargs) -> pd.DataFrame:
        """
        Preprocess text column in DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to preprocess
            **kwargs: Arguments for preprocess_text
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        df_copy = df.copy()
        
        # Apply preprocessing
        df_copy[f'{text_column}_clean'] = df_copy[text_column].apply(
            lambda x: self.preprocess_text(x, **kwargs)
        )
        
        # Remove empty reviews after preprocessing
        df_copy = df_copy[df_copy[f'{text_column}_clean'].str.len() > 0]
        
        # Add text statistics
        df_copy[f'{text_column}_word_count'] = df_copy[f'{text_column}_clean'].apply(
            lambda x: len(x.split())
        )
        df_copy[f'{text_column}_char_count'] = df_copy[f'{text_column}_clean'].apply(len)
        
        return df_copy
    
    def extract_features(self, texts: List[str], 
                        max_features: int = 5000,
                        ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, List[str]]:
        """
        Extract TF-IDF features from texts
        
        Args:
            texts: List of preprocessed texts
            max_features: Maximum number of features
            ngram_range: Range of n-grams
            
        Returns:
            Tuple[np.ndarray, List[str]]: Feature matrix and feature names
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8
        )
        
        features = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        return features.toarray(), feature_names
    
    def get_text_statistics(self, df: pd.DataFrame, 
                          text_column: str = 'review_text_clean') -> pd.DataFrame:
        """
        Generate text statistics for analysis
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            pd.DataFrame: Statistics summary
        """
        stats = {
            'total_reviews': len(df),
            'avg_word_count': df[f'{text_column}_word_count'].mean(),
            'avg_char_count': df[f'{text_column}_char_count'].mean(),
            'max_word_count': df[f'{text_column}_word_count'].max(),
            'min_word_count': df[f'{text_column}_word_count'].min(),
            'empty_reviews': (df[text_column].str.len() == 0).sum()
        }
        
        return pd.DataFrame([stats])