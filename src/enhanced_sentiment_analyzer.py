"""
Enhanced sentiment analyzer with multiple approaches for improved accuracy
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import List, Dict, Tuple
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer combining multiple approaches"""
    
    def __init__(self):
        self.model_type = 'enhanced_ensemble'
        self._setup_sentiment_lexicons()
        self._setup_negation_words()
        self._setup_intensifiers()
    
    def _setup_sentiment_lexicons(self):
        """Setup comprehensive sentiment word lists"""
        self.positive_words = {
            # Core positive
            'excellent', 'amazing', 'fantastic', 'wonderful', 'perfect', 'outstanding',
            'superb', 'brilliant', 'awesome', 'great', 'good', 'love', 'like',
            'best', 'better', 'nice', 'beautiful', 'impressive', 'satisfied',
            'happy', 'pleased', 'recommend', 'quality', 'fast', 'quick',
            'easy', 'helpful', 'comfortable', 'reliable', 'durable', 'efficient',
            'smooth', 'solid', 'sturdy', 'worth', 'value', 'bargain',
            
            # Product-specific positive
            'works', 'working', 'perfect', 'exactly', 'fits', 'arrived',
            'delivery', 'shipping', 'packed', 'condition', 'expected',
            'recommended', 'buying', 'purchase', 'ordered', 'received',
            'install', 'setup', 'assembly', 'instructions', 'manual',
            
            # Emotional positive
            'thrilled', 'delighted', 'ecstatic', 'excited', 'grateful',
            'thankful', 'blessed', 'lucky', 'fortunate', 'confident',
            'optimistic', 'hopeful', 'cheerful', 'joyful', 'content',
            
            # Intensity positive
            'extremely', 'incredibly', 'absolutely', 'totally', 'completely',
            'perfectly', 'definitely', 'certainly', 'surely', 'truly',
            'really', 'very', 'quite', 'rather', 'pretty', 'fairly'
        }
        
        self.negative_words = {
            # Core negative
            'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst',
            'bad', 'poor', 'useless', 'broken', 'defective', 'cheap',
            'flimsy', 'disappointed', 'frustrating', 'annoying', 'slow',
            'expensive', 'overpriced', 'waste', 'difficult', 'hard',
            'complicated', 'uncomfortable', 'ugly', 'damaged', 'wrong',
            'misleading', 'fake', 'fraud', 'scam', 'regret', 'sorry',
            
            # Product-specific negative
            'failed', 'failure', 'problem', 'issue', 'trouble', 'fault',
            'faulty', 'defect', 'return', 'returned', 'refund', 'exchange',
            'complaint', 'complain', 'dissatisfied', 'unhappy', 'upset',
            'angry', 'mad', 'furious', 'outraged', 'appalled', 'shocked',
            
            # Quality negative
            'flawed', 'inferior', 'substandard', 'worthless', 'pathetic',
            'ridiculous', 'absurd', 'unacceptable', 'inadequate', 'insufficient',
            'lacking', 'missing', 'incomplete', 'partial', 'limited',
            
            # Experience negative
            'nightmare', 'disaster', 'catastrophe', 'mess', 'chaos',
            'confusion', 'headache', 'struggle', 'battle', 'fight',
            'stuck', 'trapped', 'lost', 'confused', 'bewildered'
        }
        
        self.neutral_words = {
            'okay', 'ok', 'fine', 'decent', 'average', 'normal', 'standard',
            'typical', 'regular', 'ordinary', 'common', 'usual', 'expected',
            'acceptable', 'reasonable', 'fair', 'moderate', 'medium',
            'neutral', 'balanced', 'mixed', 'varied', 'different'
        }
    
    def _setup_negation_words(self):
        """Setup negation handling"""
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'nowhere', 'hardly', 'scarcely', 'barely', 'dont', 'doesnt',
            'didnt', 'wont', 'wouldnt', 'shouldnt', 'couldnt', 'cannot',
            'cant', 'mustnt', 'neednt', 'dared', 'need', 'ought'
        }
    
    def _setup_intensifiers(self):
        """Setup intensity modifiers"""
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
            'totally': 1.7, 'completely': 1.8, 'perfectly': 1.6, 'definitely': 1.4,
            'certainly': 1.4, 'surely': 1.4, 'truly': 1.3, 'really': 1.3,
            'quite': 1.2, 'rather': 1.2, 'pretty': 1.1, 'fairly': 1.1,
            'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5, 'hardly': 0.4,
            'scarcely': 0.4, 'almost': 0.9, 'nearly': 0.9, 'just': 0.8
        }
    
    def enhanced_lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """Enhanced lexicon-based sentiment with negation and intensity handling"""
        if not text or pd.isna(text):
            return {'sentiment': 'neutral', 'polarity': 0.0, 'confidence': 0.0}
        
        words = text.lower().split()
        sentiment_score = 0.0
        sentiment_count = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for negation in previous 3 words
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.negation_words:
                    negated = True
                    break
            
            # Check for intensifiers in previous 2 words
            intensity = 1.0
            for j in range(max(0, i-2), i):
                if words[j] in self.intensifiers:
                    intensity = self.intensifiers[words[j]]
                    break
            
            # Calculate sentiment for current word
            word_sentiment = 0.0
            if word in self.positive_words:
                word_sentiment = 1.0
            elif word in self.negative_words:
                word_sentiment = -1.0
            elif word in self.neutral_words:
                word_sentiment = 0.0
            
            # Apply negation
            if negated and word_sentiment != 0:
                word_sentiment = -word_sentiment
            
            # Apply intensity
            word_sentiment *= intensity
            
            if word_sentiment != 0:
                sentiment_score += word_sentiment
                sentiment_count += 1
            
            i += 1
        
        # Calculate final sentiment
        if sentiment_count == 0:
            return {'sentiment': 'neutral', 'polarity': 0.0, 'confidence': 0.0}
        
        avg_sentiment = sentiment_score / sentiment_count
        confidence = min(sentiment_count / len(words), 1.0)
        
        # Determine sentiment label
        if avg_sentiment > 0.2:
            sentiment = 'positive'
        elif avg_sentiment < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': avg_sentiment,
            'confidence': confidence
        }
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Enhanced TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Enhanced thresholds
            if polarity > 0.05:
                sentiment = 'positive'
            elif polarity < -0.05:
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
            return {'sentiment': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0, 'confidence': 0.0}
    
    def pattern_based_sentiment(self, text: str) -> Dict[str, float]:
        """Pattern-based sentiment analysis for specific constructions"""
        text_lower = text.lower()
        
        # Positive patterns
        positive_patterns = [
            r'\b(love|loved|loving)\b',
            r'\b(perfect|perfectly)\b',
            r'\b(excellent|amazing|fantastic)\b',
            r'\b(recommend|recommended)\b',
            r'\b(worth|worthy)\b',
            r'\b(satisfied|satisfaction)\b',
            r'\b(happy|pleased)\b',
            r'\b(great|good|nice)\s+(quality|product|item|purchase)\b',
            r'\b(fast|quick|rapid)\s+(delivery|shipping|service)\b',
            r'\b(easy|simple)\s+(to|installation|setup)\b',
            r'\b(works|working)\s+(perfectly|great|well|fine)\b'
        ]
        
        # Negative patterns
        negative_patterns = [
            r'\b(waste|wasted)\s+(money|time|purchase)\b',
            r'\b(dont|do not|never)\s+(buy|purchase|recommend)\b',
            r'\b(returned|returning|return)\b',
            r'\b(broken|defective|faulty)\b',
            r'\b(disappointed|frustrating|annoying)\b',
            r'\b(terrible|awful|horrible)\b',
            r'\b(poor|bad|worst)\s+(quality|product|service)\b',
            r'\b(difficult|hard|complicated)\s+(to|installation|setup)\b',
            r'\b(does not|doesnt|did not|didnt)\s+(work|function)\b'
        ]
        
        positive_matches = sum(1 for pattern in positive_patterns if re.search(pattern, text_lower))
        negative_matches = sum(1 for pattern in negative_patterns if re.search(pattern, text_lower))
        
        if positive_matches > negative_matches:
            sentiment = 'positive'
            polarity = 0.5 + (positive_matches * 0.2)
        elif negative_matches > positive_matches:
            sentiment = 'negative'
            polarity = -0.5 - (negative_matches * 0.2)
        else:
            sentiment = 'neutral'
            polarity = 0.0
        
        confidence = (positive_matches + negative_matches) / 10.0
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'confidence': min(confidence, 1.0)
        }
    
    def ensemble_sentiment(self, text: str) -> Dict[str, float]:
        """Ensemble method combining multiple approaches"""
        if not text or pd.isna(text):
            return {'sentiment': 'neutral', 'polarity': 0.0, 'confidence': 0.0}
        
        # Get predictions from all methods
        lexicon_result = self.enhanced_lexicon_sentiment(text)
        textblob_result = self.textblob_sentiment(text)
        pattern_result = self.pattern_based_sentiment(text)
        
        # Weighted ensemble
        weights = {
            'lexicon': 0.4,
            'textblob': 0.35,
            'pattern': 0.25
        }
        
        # Combine polarities
        combined_polarity = (
            lexicon_result['polarity'] * weights['lexicon'] +
            textblob_result['polarity'] * weights['textblob'] +
            pattern_result['polarity'] * weights['pattern']
        )
        
        # Combine confidences
        combined_confidence = (
            lexicon_result['confidence'] * weights['lexicon'] +
            textblob_result['confidence'] * weights['textblob'] +
            pattern_result['confidence'] * weights['pattern']
        )
        
        # Determine final sentiment with adjusted thresholds
        if combined_polarity > 0.1:
            sentiment = 'positive'
        elif combined_polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Voting mechanism as backup
        sentiments = [lexicon_result['sentiment'], textblob_result['sentiment'], pattern_result['sentiment']]
        sentiment_counts = Counter(sentiments)
        
        # If ensemble is uncertain, use voting
        if abs(combined_polarity) < 0.05:
            sentiment = sentiment_counts.most_common(1)[0][0]
        
        return {
            'sentiment': sentiment,
            'polarity': combined_polarity,
            'confidence': combined_confidence,
            'lexicon_sentiment': lexicon_result['sentiment'],
            'textblob_sentiment': textblob_result['sentiment'],
            'pattern_sentiment': pattern_result['sentiment']
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Main sentiment analysis method"""
        return self.ensemble_sentiment(text)
    
    def batch_analyze(self, texts: List[str], batch_size: int = 100) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {min(i + len(batch), len(texts))}/{len(texts)} texts")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text_clean') -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame"""
        df_copy = df.copy()
        
        print(f"Analyzing sentiment for {len(df_copy)} reviews using enhanced model...")
        
        # Analyze sentiment
        results = self.batch_analyze(df_copy[text_column].tolist())
        
        # Add results to DataFrame
        for key in ['sentiment', 'polarity', 'confidence', 'lexicon_sentiment', 'textblob_sentiment', 'pattern_sentiment']:
            df_copy[f'enhanced_{key}'] = [result.get(key, 0) for result in results]
        
        print("Enhanced sentiment analysis completed!")
        return df_copy
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze which features contribute most to sentiment prediction"""
        # Word frequency by sentiment
        positive_words = []
        negative_words = []
        
        for _, row in df.iterrows():
            words = row['review_text_clean'].split()
            if row['enhanced_sentiment'] == 'positive':
                positive_words.extend(words)
            elif row['enhanced_sentiment'] == 'negative':
                negative_words.extend(words)
        
        pos_freq = Counter(positive_words)
        neg_freq = Counter(negative_words)
        
        # Calculate sentiment strength for each word
        word_sentiment_strength = {}
        all_words = set(positive_words + negative_words)
        
        for word in all_words:
            pos_count = pos_freq.get(word, 0)
            neg_count = neg_freq.get(word, 0)
            total_count = pos_count + neg_count
            
            if total_count >= 5:  # Minimum frequency threshold
                sentiment_strength = (pos_count - neg_count) / total_count
                word_sentiment_strength[word] = {
                    'word': word,
                    'positive_count': pos_count,
                    'negative_count': neg_count,
                    'sentiment_strength': sentiment_strength,
                    'total_count': total_count
                }
        
        # Convert to DataFrame and sort by absolute sentiment strength
        feature_df = pd.DataFrame(word_sentiment_strength.values())
        feature_df = feature_df.sort_values('sentiment_strength', key=abs, ascending=False)
        
        return feature_df.head(50)