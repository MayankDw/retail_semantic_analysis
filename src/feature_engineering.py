"""
Advanced feature engineering for enhanced sentiment analysis
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for sentiment analysis"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.feature_names = []
    
    def extract_linguistic_features(self, df: pd.DataFrame, text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Extract linguistic features from text"""
        df_copy = df.copy()
        
        print("Extracting linguistic features...")
        
        # Basic text statistics
        df_copy['text_length'] = df_copy[text_column].str.len()
        df_copy['word_count'] = df_copy[text_column].str.split().str.len()
        df_copy['sentence_count'] = df_copy[text_column].str.count(r'[.!?]+') + 1
        df_copy['avg_word_length'] = df_copy[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Punctuation features
        df_copy['exclamation_count'] = df_copy[text_column].str.count('!')
        df_copy['question_count'] = df_copy[text_column].str.count(r'\?')
        df_copy['capital_ratio'] = df_copy[text_column].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Readability features
        df_copy['avg_sentence_length'] = df_copy['word_count'] / df_copy['sentence_count']
        df_copy['lexical_diversity'] = df_copy[text_column].apply(
            lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
        )
        
        # Sentiment-related patterns
        df_copy['positive_word_count'] = df_copy[text_column].apply(self._count_positive_words)
        df_copy['negative_word_count'] = df_copy[text_column].apply(self._count_negative_words)
        df_copy['negation_count'] = df_copy[text_column].apply(self._count_negations)
        df_copy['intensifier_count'] = df_copy[text_column].apply(self._count_intensifiers)
        
        # Product-specific features
        df_copy['price_mentions'] = df_copy[text_column].str.count(r'\$|price|cost|expensive|cheap|budget')
        df_copy['quality_mentions'] = df_copy[text_column].str.count(r'quality|durable|sturdy|flimsy|cheap')
        df_copy['service_mentions'] = df_copy[text_column].str.count(r'service|support|help|customer|delivery|shipping')
        df_copy['comparison_mentions'] = df_copy[text_column].str.count(r'better|worse|best|worst|than|compared')
        
        print("Linguistic features extracted!")
        return df_copy
    
    def _count_positive_words(self, text: str) -> int:
        """Count positive words in text"""
        positive_words = {
            'excellent', 'amazing', 'fantastic', 'wonderful', 'perfect', 'outstanding',
            'superb', 'brilliant', 'awesome', 'great', 'good', 'love', 'like',
            'best', 'better', 'nice', 'beautiful', 'recommend', 'satisfied',
            'happy', 'pleased', 'quality', 'fast', 'easy', 'helpful'
        }
        return sum(1 for word in text.lower().split() if word in positive_words)
    
    def _count_negative_words(self, text: str) -> int:
        """Count negative words in text"""
        negative_words = {
            'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst',
            'bad', 'poor', 'useless', 'broken', 'defective', 'disappointed',
            'frustrating', 'annoying', 'expensive', 'waste', 'difficult',
            'uncomfortable', 'damaged', 'wrong', 'regret', 'sorry'
        }
        return sum(1 for word in text.lower().split() if word in negative_words)
    
    def _count_negations(self, text: str) -> int:
        """Count negation words in text"""
        negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt',
            'couldnt', 'cannot', 'cant', 'mustnt'
        }
        return sum(1 for word in text.lower().split() if word in negation_words)
    
    def _count_intensifiers(self, text: str) -> int:
        """Count intensifier words in text"""
        intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'totally',
            'completely', 'perfectly', 'definitely', 'certainly', 'really',
            'quite', 'rather', 'pretty', 'fairly', 'somewhat', 'slightly'
        }
        return sum(1 for word in text.lower().split() if word in intensifiers)
    
    def create_tfidf_features(self, df: pd.DataFrame, text_column: str = 'review_text_clean', 
                             max_features: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create TF-IDF features"""
        print(f"Creating TF-IDF features with {max_features} features...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words='english',
            lowercase=True,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column])
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return df, tfidf_matrix
    
    def create_topic_features(self, df: pd.DataFrame, tfidf_matrix: np.ndarray, 
                            n_topics: int = 10) -> pd.DataFrame:
        """Create topic modeling features"""
        print(f"Creating topic features with {n_topics} topics...")
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='batch'
        )
        
        # Get topic distributions
        topic_distributions = self.lda_model.fit_transform(tfidf_matrix)
        
        # Add topic features to dataframe
        df_copy = df.copy()
        for i in range(n_topics):
            df_copy[f'topic_{i}_weight'] = topic_distributions[:, i]
        
        # Add dominant topic
        df_copy['dominant_topic'] = np.argmax(topic_distributions, axis=1)
        df_copy['dominant_topic_weight'] = np.max(topic_distributions, axis=1)
        
        print("Topic features created!")
        return df_copy
    
    def create_sentiment_features(self, df: pd.DataFrame, text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Create sentiment-specific features"""
        print("Creating sentiment-specific features...")
        
        df_copy = df.copy()
        
        # Sentiment word ratios
        df_copy['positive_ratio'] = df_copy['positive_word_count'] / df_copy['word_count']
        df_copy['negative_ratio'] = df_copy['negative_word_count'] / df_copy['word_count']
        df_copy['sentiment_word_ratio'] = (df_copy['positive_word_count'] + df_copy['negative_word_count']) / df_copy['word_count']
        
        # Context features
        df_copy['has_recommendation'] = df_copy[text_column].str.contains(r'recommend|suggest', case=False, na=False).astype(int)
        df_copy['has_comparison'] = df_copy[text_column].str.contains(r'better|worse|than|compared', case=False, na=False).astype(int)
        df_copy['has_emotion'] = df_copy[text_column].str.contains(r'love|hate|excited|disappointed|thrilled|frustrated', case=False, na=False).astype(int)
        
        # Product experience features
        df_copy['mentions_purchase'] = df_copy[text_column].str.contains(r'bought|purchase|buy|order', case=False, na=False).astype(int)
        df_copy['mentions_usage'] = df_copy[text_column].str.contains(r'use|used|using|work|works', case=False, na=False).astype(int)
        df_copy['mentions_problem'] = df_copy[text_column].str.contains(r'problem|issue|trouble|fault|error', case=False, na=False).astype(int)
        
        # Temporal features
        df_copy['mentions_time'] = df_copy[text_column].str.contains(r'day|week|month|year|time|long|short|quick|fast|slow', case=False, na=False).astype(int)
        
        print("Sentiment-specific features created!")
        return df_copy
    
    def create_engineered_features(self, df: pd.DataFrame, text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Create all engineered features"""
        print("Creating comprehensive engineered features...")
        
        # Extract linguistic features
        df_features = self.extract_linguistic_features(df, text_column)
        
        # Create TF-IDF features
        df_features, tfidf_matrix = self.create_tfidf_features(df_features, text_column)
        
        # Create topic features
        df_features = self.create_topic_features(df_features, tfidf_matrix)
        
        # Create sentiment features
        df_features = self.create_sentiment_features(df_features, text_column)
        
        # Create interaction features
        df_features = self._create_interaction_features(df_features)
        
        print("All features created successfully!")
        return df_features
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        print("Creating interaction features...")
        
        df_copy = df.copy()
        
        # Sentiment-length interactions
        df_copy['positive_per_word'] = df_copy['positive_word_count'] / (df_copy['word_count'] + 1)
        df_copy['negative_per_word'] = df_copy['negative_word_count'] / (df_copy['word_count'] + 1)
        df_copy['sentiment_density'] = (df_copy['positive_word_count'] + df_copy['negative_word_count']) / (df_copy['word_count'] + 1)
        
        # Punctuation-sentiment interactions
        df_copy['exclamation_sentiment'] = df_copy['exclamation_count'] * df_copy['positive_ratio']
        df_copy['question_negative'] = df_copy['question_count'] * df_copy['negative_ratio']
        
        # Topic-sentiment interactions
        if 'dominant_topic_weight' in df_copy.columns:
            df_copy['topic_sentiment_strength'] = df_copy['dominant_topic_weight'] * df_copy['sentiment_density']
        
        # Length-quality interactions
        df_copy['length_quality_score'] = df_copy['word_count'] * df_copy['lexical_diversity']
        
        print("Interaction features created!")
        return df_copy
    
    def get_feature_importance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of feature importance"""
        feature_columns = [col for col in df.columns if any(x in col for x in [
            'text_', 'word_', 'sentence_', 'avg_', 'exclamation_', 'question_', 
            'capital_', 'lexical_', 'positive_', 'negative_', 'negation_', 
            'intensifier_', 'price_', 'quality_', 'service_', 'comparison_',
            'topic_', 'dominant_', 'has_', 'mentions_', '_per_word', '_density',
            '_strength', '_score'
        ])]
        
        feature_stats = []
        for feature in feature_columns:
            if feature in df.columns:
                stats = {
                    'feature': feature,
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'non_zero_count': (df[feature] != 0).sum(),
                    'non_zero_pct': (df[feature] != 0).mean() * 100
                }
                feature_stats.append(stats)
        
        return pd.DataFrame(feature_stats).sort_values('non_zero_pct', ascending=False)
    
    def save_feature_names(self, filepath: str):
        """Save feature names for later use"""
        if self.feature_names is not None:
            feature_df = pd.DataFrame({
                'feature_name': self.feature_names,
                'feature_index': range(len(self.feature_names))
            })
            feature_df.to_csv(filepath, index=False)
            print(f"Feature names saved to {filepath}")
    
    def get_top_tfidf_features_by_sentiment(self, df: pd.DataFrame, tfidf_matrix: np.ndarray, 
                                          sentiment_column: str = 'enhanced_sentiment',
                                          top_n: int = 20) -> Dict[str, List[str]]:
        """Get top TF-IDF features by sentiment"""
        if self.tfidf_vectorizer is None:
            print("TF-IDF vectorizer not fitted yet!")
            return {}
        
        results = {}
        
        for sentiment in df[sentiment_column].unique():
            sentiment_mask = df[sentiment_column] == sentiment
            sentiment_tfidf = tfidf_matrix[sentiment_mask]
            
            # Calculate mean TF-IDF scores for this sentiment
            mean_scores = np.array(sentiment_tfidf.mean(axis=0)).flatten()
            
            # Get top features
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_scores = [mean_scores[i] for i in top_indices]
            
            results[sentiment] = list(zip(top_features, top_scores))
        
        return results