"""
Simplified text preprocessor with better error handling
"""
import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class SimpleTextPreprocessor:
    """Simplified text preprocessing that doesn't rely heavily on NLTK"""
    
    def __init__(self):
        """Initialize with basic stopwords list"""
        # Basic English stopwords (no NLTK required)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
            'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
            'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
            'call', 'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take',
            'only', 'little', 'work', 'know', 'place', 'year', 'live', 'me',
            'back', 'give', 'most', 'very', 'after', 'thing', 'our', 'just', 'name',
            'good', 'sentence', 'man', 'think', 'say', 'great', 'where', 'help',
            'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old',
            'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show',
            'also', 'around', 'form', 'three', 'small', 'set', 'put', 'end',
            'why', 'again', 'turn', 'here', 'off', 'went', 'see', 'need',
            'should', 'home', 'about', 'while', 'below', 'country', 'plant',
            'last', 'school', 'father', 'keep', 'tree', 'never', 'start',
            'city', 'earth', 'eye', 'light', 'thought', 'head', 'under', 'story',
            'saw', 'left', 'don', 'few', 'while', 'along', 'might', 'close',
            'something', 'seem', 'next', 'hard', 'open', 'example', 'begin',
            'life', 'always', 'those', 'both', 'paper', 'together', 'got',
            'group', 'often', 'run', 'important', 'until', 'children', 'side',
            'feet', 'car', 'mile', 'night', 'walk', 'white', 'sea', 'began',
            'grow', 'took', 'river', 'four', 'carry', 'state', 'once', 'book',
            'hear', 'stop', 'without', 'second', 'later', 'miss', 'idea',
            'enough', 'eat', 'face', 'watch', 'far', 'indian', 'really',
            'almost', 'let', 'above', 'girl', 'sometimes', 'mountain', 'cut',
            'young', 'talk', 'soon', 'list', 'song', 'being', 'leave', 'family',
            'it', 'i', 'you', 'we', 'us', 'can', 'not', 'all', 'use', 'your',
            'there', 'one', 'when', 'have', 'an', 'each', 'which', 'their',
            'said', 'do', 'has', 'its', 'had', 'two', 'more', 'her', 'like',
            'other', 'after', 'first', 'been', 'many', 'who', 'oil', 'its',
            'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made',
            'may', 'part'
        }
        
        # Add retail-specific stop words
        self.stop_words.update({
            'product', 'item', 'buy', 'purchase', 'order', 'bought', 'get', 'got',
            'would', 'could', 'should', 'really', 'good', 'bad', 'nice', 'great',
            'amazon', 'seller', 'delivery', 'shipping', 'price', 'money'
        })
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
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
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
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
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization without NLTK"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        tokens = self.simple_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in self.stop_words and len(token) > 2
        ]
        return ' '.join(filtered_tokens)
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       remove_punct: bool = True,
                       expand_contractions: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        # Basic cleaning
        text = self.clean_text(text)
        if not text:
            return ""
        
        # Expand contractions
        if expand_contractions:
            text = self.expand_contractions(text)
        
        # Remove punctuation
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'review_text',
                           **kwargs) -> pd.DataFrame:
        """Preprocess text column in DataFrame"""
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
    
    def get_text_statistics(self, df: pd.DataFrame, 
                          text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Generate text statistics for analysis"""
        stats = {
            'total_reviews': len(df),
            'avg_word_count': df[f'{text_column}_word_count'].mean(),
            'avg_char_count': df[f'{text_column}_char_count'].mean(),
            'max_word_count': df[f'{text_column}_word_count'].max(),
            'min_word_count': df[f'{text_column}_word_count'].min(),
            'empty_reviews': (df[text_column].str.len() == 0).sum()
        }
        
        return pd.DataFrame([stats])