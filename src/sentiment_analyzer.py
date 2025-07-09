"""
Sentiment analysis implementation for retail reviews
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

class RetailSentimentAnalyzer:
    """Comprehensive sentiment analysis for retail reviews"""
    
    def __init__(self, model_type: str = 'textblob'):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: Type of model ('textblob', 'vader', 'ml', 'transformer')
        """
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.transformer_pipeline = None
        
        if model_type == 'transformer':
            self._load_transformer_model()
    
    def _load_transformer_model(self):
        """Load pre-trained transformer model for sentiment analysis"""
        try:
            # Use a pre-trained model optimized for customer reviews
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            print("Falling back to TextBlob")
            self.model_type = 'textblob'
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment labels
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        try:
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            
            # Determine sentiment based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'confidence': abs(compound)
            }
        except Exception as e:
            print(f"VADER not available: {e}")
            return self.textblob_sentiment(text)
    
    def transformer_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        if self.transformer_pipeline is None:
            return self.textblob_sentiment(text)
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.transformer_pipeline(text)[0]
            
            # Map transformer labels to our format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'POSITIVE': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'raw_label': result['label'],
                'raw_score': result['score']
            }
        except Exception as e:
            print(f"Transformer sentiment failed: {e}")
            return self.textblob_sentiment(text)
    
    def train_ml_model(self, df: pd.DataFrame, 
                      text_column: str = 'review_text_clean',
                      target_column: str = 'sentiment',
                      model_type: str = 'logistic') -> Dict[str, float]:
        """
        Train machine learning model for sentiment analysis
        
        Args:
            df: Training DataFrame
            text_column: Column with preprocessed text
            target_column: Column with sentiment labels
            model_type: Type of ML model ('logistic', 'rf', 'svm', 'nb')
            
        Returns:
            Dict with training metrics
        """
        # Prepare features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose model
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'rf': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'nb': MultinomialNB()
        }
        
        self.model = models[model_type]
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def ml_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using trained ML model
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        if self.model is None or self.vectorizer is None:
            return self.textblob_sentiment(text)
        
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Predict sentiment
            sentiment = self.model.predict(X)[0]
            confidence = max(self.model.predict_proba(X)[0])
            
            return {
                'sentiment': sentiment,
                'confidence': confidence
            }
        except Exception as e:
            print(f"ML sentiment failed: {e}")
            return self.textblob_sentiment(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using selected method
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment analysis results
        """
        if self.model_type == 'textblob':
            return self.textblob_sentiment(text)
        elif self.model_type == 'vader':
            return self.vader_sentiment(text)
        elif self.model_type == 'transformer':
            return self.transformer_sentiment(text)
        elif self.model_type == 'ml':
            return self.ml_sentiment(text)
        else:
            return self.textblob_sentiment(text)
    
    def batch_analyze(self, texts: List[str], 
                     batch_size: int = 100) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            batch_size: Size of processing batches
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = 'review_text_clean') -> pd.DataFrame:
        """
        Analyze sentiment for entire DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column with text to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        df_copy = df.copy()
        
        # Analyze sentiment
        results = self.batch_analyze(df_copy[text_column].tolist())
        
        # Add results to DataFrame
        for key in results[0].keys():
            df_copy[f'sentiment_{key}'] = [result[key] for result in results]
        
        return df_copy
    
    def get_sentiment_distribution(self, df: pd.DataFrame, 
                                 sentiment_column: str = 'sentiment_sentiment') -> pd.DataFrame:
        """
        Get sentiment distribution statistics
        
        Args:
            df: DataFrame with sentiment analysis results
            sentiment_column: Column with sentiment labels
            
        Returns:
            DataFrame with distribution statistics
        """
        distribution = df[sentiment_column].value_counts(normalize=True)
        
        return pd.DataFrame({
            'sentiment': distribution.index,
            'count': df[sentiment_column].value_counts(),
            'percentage': distribution.values * 100
        })
    
    def aspect_based_sentiment(self, df: pd.DataFrame, 
                             aspects: List[str],
                             text_column: str = 'review_text_clean') -> pd.DataFrame:
        """
        Perform aspect-based sentiment analysis
        
        Args:
            df: Input DataFrame
            aspects: List of aspects to analyze
            text_column: Column with text
            
        Returns:
            DataFrame with aspect-based sentiment results
        """
        results = []
        
        for _, row in df.iterrows():
            text = row[text_column]
            
            for aspect in aspects:
                if aspect.lower() in text.lower():
                    # Extract sentences containing the aspect
                    sentences = text.split('.')
                    aspect_sentences = [
                        s for s in sentences 
                        if aspect.lower() in s.lower()
                    ]
                    
                    if aspect_sentences:
                        # Analyze sentiment of aspect-related sentences
                        aspect_text = '. '.join(aspect_sentences)
                        sentiment_result = self.analyze_sentiment(aspect_text)
                        
                        results.append({
                            'review_id': row.name,
                            'aspect': aspect,
                            'aspect_text': aspect_text,
                            'sentiment': sentiment_result['sentiment'],
                            'confidence': sentiment_result.get('confidence', 0)
                        })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath: str):
        """Save trained model and vectorizer"""
        if self.model is not None and self.vectorizer is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'model_type': self.model_type
                }, f)
    
    def load_model(self, filepath: str):
        """Load trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.model_type = data['model_type']