"""
Completely NLTK-free text preprocessor for retail sentiment analysis
"""
import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class NLTKFreePreprocessor:
    """Text preprocessing that works without any NLTK dependencies"""
    
    def __init__(self):
        """Initialize with basic stopwords and settings"""
        # Comprehensive English stopwords list (no NLTK required)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
            'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
            'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        }
        
        # Add retail-specific stop words
        self.stop_words.update({
            'product', 'item', 'buy', 'purchase', 'order', 'bought', 'get', 'got',
            'would', 'could', 'should', 'really', 'good', 'bad', 'nice', 'great',
            'amazon', 'seller', 'delivery', 'shipping', 'price', 'money', 'one',
            'two', 'three', 'also', 'like', 'make', 'made', 'go', 'come', 'came',
            'say', 'said', 'know', 'think', 'see', 'look', 'way', 'take', 'want',
            'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try',
            'leave', 'call', 'back', 'new', 'first', 'last', 'long', 'little',
            'own', 'right', 'big', 'high', 'small', 'large', 'next', 'early',
            'young', 'important', 'different', 'following', 'public', 'able'
        })
        
        # Common contractions
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning without external dependencies"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short and very long texts
        if len(text) < 5 or len(text) > 5000:
            return ""
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions using predefined dictionary"""
        text_lower = text.lower()
        for contraction, expansion in self.contractions.items():
            text_lower = text_lower.replace(contraction, expansion)
        return text_lower
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization using regex"""
        # Split on whitespace and extract word-like tokens
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return tokens
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords using simple tokenization"""
        tokens = self.simple_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(filtered_tokens)
    
    def simple_lemmatize(self, text: str) -> str:
        """Simple suffix stripping for basic lemmatization"""
        tokens = self.simple_tokenize(text)
        lemmatized = []
        
        for token in tokens:
            # Simple suffix removal rules
            if token.endswith('ing') and len(token) > 6:
                token = token[:-3]
            elif token.endswith('ed') and len(token) > 5:
                token = token[:-2]
            elif token.endswith('er') and len(token) > 5:
                token = token[:-2]
            elif token.endswith('est') and len(token) > 6:
                token = token[:-3]
            elif token.endswith('ly') and len(token) > 5:
                token = token[:-2]
            elif token.endswith('tion') and len(token) > 7:
                token = token[:-4]
            elif token.endswith('ness') and len(token) > 7:
                token = token[:-4]
            elif token.endswith('ment') and len(token) > 7:
                token = token[:-4]
            elif token.endswith('ful') and len(token) > 6:
                token = token[:-3]
            elif token.endswith('less') and len(token) > 7:
                token = token[:-4]
            
            if len(token) > 2:
                lemmatized.append(token)
        
        return ' '.join(lemmatized)
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       lemmatize: bool = True,
                       expand_contractions: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        # Basic cleaning
        text = self.clean_text(text)
        if not text:
            return ""
        
        # Expand contractions
        if expand_contractions:
            text = self.expand_contractions(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Simple lemmatization
        if lemmatize:
            text = self.simple_lemmatize(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'review_text',
                           **kwargs) -> pd.DataFrame:
        """Preprocess text column in DataFrame"""
        df_copy = df.copy()
        
        print(f"Preprocessing {len(df_copy)} reviews...")
        
        # Apply preprocessing
        df_copy[f'{text_column}_clean'] = df_copy[text_column].apply(
            lambda x: self.preprocess_text(x, **kwargs)
        )
        
        # Remove empty reviews after preprocessing
        original_count = len(df_copy)
        df_copy = df_copy[df_copy[f'{text_column}_clean'].str.len() > 0]
        removed_count = original_count - len(df_copy)
        
        if removed_count > 0:
            print(f"Removed {removed_count} empty reviews after preprocessing")
        
        # Add text statistics
        df_copy[f'{text_column}_word_count'] = df_copy[f'{text_column}_clean'].apply(
            lambda x: len(x.split())
        )
        df_copy[f'{text_column}_char_count'] = df_copy[f'{text_column}_clean'].apply(len)
        
        print(f"Preprocessing complete! {len(df_copy)} reviews ready for analysis.")
        
        return df_copy
    
    def get_text_statistics(self, df: pd.DataFrame, 
                          text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Generate text statistics for analysis"""
        word_counts = df[f'{text_column}_word_count']
        char_counts = df[f'{text_column}_char_count']
        
        stats = {
            'total_reviews': len(df),
            'avg_word_count': word_counts.mean(),
            'median_word_count': word_counts.median(),
            'avg_char_count': char_counts.mean(),
            'max_word_count': word_counts.max(),
            'min_word_count': word_counts.min(),
            'std_word_count': word_counts.std(),
            'empty_reviews': (df[text_column].str.len() == 0).sum()
        }
        
        return pd.DataFrame([stats])
    
    def extract_basic_features(self, texts: List[str], max_features: int = 1000) -> tuple:
        """Extract simple TF-IDF-like features without sklearn initially"""
        from collections import Counter
        import math
        
        # Get all unique words
        all_words = []
        doc_word_counts = []
        
        for text in texts:
            words = self.simple_tokenize(text)
            all_words.extend(words)
            doc_word_counts.append(Counter(words))
        
        # Get most common words
        word_counts = Counter(all_words)
        vocab = [word for word, count in word_counts.most_common(max_features)]
        
        # Create simple TF matrix
        tf_matrix = []
        for doc_words in doc_word_counts:
            tf_vector = []
            total_words = sum(doc_words.values())
            for word in vocab:
                tf = doc_words.get(word, 0) / total_words if total_words > 0 else 0
                tf_vector.append(tf)
            tf_matrix.append(tf_vector)
        
        return np.array(tf_matrix), vocab