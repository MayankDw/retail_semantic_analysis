"""
NLTK-free sentiment analyzer using TextBlob only
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class NLTKFreeSentimentAnalyzer:
    """Sentiment analyzer that works without NLTK tokenization"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.model_type = 'textblob'
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob (no NLTK tokenization)"""
        try:
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
        except Exception as e:
            print(f"TextBlob sentiment analysis failed for text: {text[:50]}... Error: {e}")
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'confidence': 0.0
            }
    
    def simple_rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Simple rule-based sentiment analysis as fallback"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
            'perfect', 'love', 'like', 'best', 'outstanding', 'superb', 'brilliant',
            'impressive', 'satisfied', 'happy', 'pleased', 'recommend', 'quality',
            'fast', 'quick', 'beautiful', 'comfortable', 'easy', 'helpful', 'nice',
            'pretty', 'solid', 'sturdy', 'reliable', 'durable', 'efficient', 'smooth'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst',
            'useless', 'broken', 'defective', 'poor', 'cheap', 'flimsy', 'disappointed',
            'frustrating', 'annoying', 'slow', 'expensive', 'overpriced', 'waste',
            'difficult', 'hard', 'complicated', 'uncomfortable', 'ugly', 'damaged',
            'wrong', 'misleading', 'fake', 'fraud', 'scam', 'regret', 'sorry'
        }
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = 'neutral'
            polarity = 0.0
            confidence = 0.0
        elif positive_count > negative_count:
            sentiment = 'positive'
            polarity = (positive_count - negative_count) / len(words)
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = 'negative'
            polarity = -(negative_count - positive_count) / len(words)
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = 'neutral'
            polarity = 0.0
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': 0.5,  # Default subjectivity
            'confidence': min(confidence, 1.0)
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with fallback methods"""
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'confidence': 0.0
            }
        
        # Try TextBlob first
        try:
            return self.textblob_sentiment(text)
        except Exception as e:
            print(f"TextBlob failed, using rule-based fallback: {e}")
            return self.simple_rule_based_sentiment(text)
    
    def batch_analyze(self, texts: List[str], batch_size: int = 100) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)
            
            if i % (batch_size * 5) == 0:
                print(f"Processed {min(i + len(batch), len(texts))}/{len(texts)} texts")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame"""
        df_copy = df.copy()
        
        print(f"Analyzing sentiment for {len(df_copy)} reviews...")
        
        # Analyze sentiment
        results = self.batch_analyze(df_copy[text_column].tolist())
        
        # Add results to DataFrame
        for key in results[0].keys():
            df_copy[f'sentiment_{key}'] = [result[key] for result in results]
        
        print("Sentiment analysis completed!")
        return df_copy
    
    def get_sentiment_distribution(self, df: pd.DataFrame, 
                                 sentiment_column: str = 'sentiment_sentiment') -> pd.DataFrame:
        """Get sentiment distribution statistics"""
        distribution = df[sentiment_column].value_counts(normalize=True)
        
        return pd.DataFrame({
            'sentiment': distribution.index,
            'count': df[sentiment_column].value_counts(),
            'percentage': distribution.values * 100
        })
    
    def aspect_based_sentiment(self, df: pd.DataFrame, 
                             aspects: List[str],
                             text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Simple aspect-based sentiment analysis"""
        results = []
        
        for _, row in df.iterrows():
            text = row[text_column]
            
            for aspect in aspects:
                if aspect.lower() in text.lower():
                    # Extract sentences containing the aspect (simple split on periods)
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
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