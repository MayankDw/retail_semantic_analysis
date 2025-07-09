"""
Real data loader for Amazon reviews from Kaggle
Handles FastText format files
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import re

class RealAmazonDataLoader:
    """Load and process real Amazon reviews data from Kaggle"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fasttext_file(self, file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load FastText format file (Amazon reviews)
        
        Args:
            file_path: Path to the FastText file
            max_samples: Maximum number of samples to load (for testing)
            
        Returns:
            pd.DataFrame: Loaded and parsed data
        """
        reviews = []
        labels = []
        
        print(f"Loading {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                line = line.strip()
                if line:
                    # FastText format: __label__1 or __label__2 followed by text
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        label = parts[0]
                        text = parts[1]
                        
                        # Convert label to sentiment (1=negative, 2=positive)
                        if label == '__label__1':
                            sentiment = 'negative'
                        elif label == '__label__2':
                            sentiment = 'positive'
                        else:
                            continue  # Skip invalid labels
                        
                        reviews.append(text)
                        labels.append(sentiment)
                
                # Progress indicator
                if i % 100000 == 0 and i > 0:
                    print(f"Processed {i:,} lines...")
        
        df = pd.DataFrame({
            'review_text': reviews,
            'sentiment': labels
        })
        
        print(f"Loaded {len(df):,} reviews")
        return df
    
    def load_train_data(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load training data"""
        train_path = self.raw_dir / "train.ft.txt"
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        return self.load_fasttext_file(train_path, max_samples)
    
    def load_test_data(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load test data"""
        test_path = self.raw_dir / "test.ft.txt"
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        return self.load_fasttext_file(test_path, max_samples)
    
    def load_combined_data(self, max_train: Optional[int] = None, 
                          max_test: Optional[int] = None) -> pd.DataFrame:
        """
        Load and combine train and test data
        
        Args:
            max_train: Maximum training samples
            max_test: Maximum test samples
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        # Load training data
        train_df = self.load_train_data(max_train)
        train_df['split'] = 'train'
        
        # Load test data
        test_df = self.load_test_data(max_test)
        test_df['split'] = 'test'
        
        # Combine
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Add basic statistics
        combined_df['review_length'] = combined_df['review_text'].str.len()
        combined_df['word_count'] = combined_df['review_text'].str.split().str.len()
        
        # Add synthetic product categories based on review content
        combined_df['product_category'] = combined_df['review_text'].apply(self._infer_category)
        
        print(f"Combined dataset: {len(combined_df):,} reviews")
        print(f"Sentiment distribution: {combined_df['sentiment'].value_counts().to_dict()}")
        
        return combined_df
    
    def _infer_category(self, text: str) -> str:
        """
        Infer product category from review text
        
        Args:
            text: Review text
            
        Returns:
            str: Inferred category
        """
        text_lower = text.lower()
        
        # Electronics keywords
        electronics_keywords = [
            'battery', 'charger', 'cable', 'phone', 'computer', 'laptop', 'camera',
            'headphones', 'speaker', 'tv', 'dvd', 'cd', 'mp3', 'ipod', 'iphone',
            'android', 'samsung', 'apple', 'sony', 'electronic', 'digital',
            'screen', 'display', 'pixel', 'resolution', 'usb', 'wireless',
            'bluetooth', 'wifi', 'software', 'app', 'device', 'gadget'
        ]
        
        # Books keywords
        books_keywords = [
            'book', 'read', 'reading', 'novel', 'story', 'author', 'writer',
            'chapter', 'page', 'paperback', 'hardcover', 'kindle', 'ebook',
            'fiction', 'nonfiction', 'biography', 'history', 'romance',
            'mystery', 'fantasy', 'science fiction', 'textbook', 'manual',
            'guide', 'reference', 'literature', 'poetry', 'comic'
        ]
        
        # Movies/Music keywords
        movies_music_keywords = [
            'movie', 'film', 'dvd', 'blu-ray', 'video', 'actor', 'actress',
            'director', 'soundtrack', 'music', 'album', 'song', 'track',
            'artist', 'band', 'singer', 'cd', 'vinyl', 'concert', 'guitar',
            'piano', 'drums', 'bass', 'rock', 'pop', 'jazz', 'classical',
            'country', 'hip hop', 'rap', 'electronic music'
        ]
        
        # Home & Garden keywords
        home_keywords = [
            'kitchen', 'home', 'house', 'room', 'furniture', 'chair', 'table',
            'bed', 'mattress', 'pillow', 'blanket', 'curtain', 'lamp',
            'decoration', 'garden', 'plant', 'flower', 'tool', 'cleaning',
            'vacuum', 'dishwasher', 'refrigerator', 'microwave', 'oven',
            'cookware', 'utensil', 'dish', 'cup', 'glass', 'bottle'
        ]
        
        # Sports keywords
        sports_keywords = [
            'sport', 'fitness', 'exercise', 'workout', 'gym', 'running',
            'bike', 'bicycle', 'swimming', 'tennis', 'golf', 'basketball',
            'football', 'soccer', 'baseball', 'hockey', 'outdoor', 'camping',
            'hiking', 'fishing', 'hunting', 'skiing', 'snowboard', 'skateboard'
        ]
        
        # Fashion keywords
        fashion_keywords = [
            'shirt', 'pants', 'dress', 'shoes', 'boots', 'sneakers',
            'jacket', 'coat', 'sweater', 'jeans', 'skirt', 'blouse',
            'suit', 'tie', 'hat', 'cap', 'belt', 'bag', 'purse',
            'jewelry', 'watch', 'necklace', 'ring', 'earring',
            'fashion', 'style', 'clothing', 'apparel', 'wear'
        ]
        
        # Count matches for each category
        categories = {
            'Electronics': electronics_keywords,
            'Books': books_keywords,
            'Movies & Music': movies_music_keywords,
            'Home & Garden': home_keywords,
            'Sports & Outdoors': sports_keywords,
            'Fashion': fashion_keywords
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score, or 'Other' if no matches
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'Other'
    
    def create_balanced_sample(self, df: pd.DataFrame, n_samples: int = 10000) -> pd.DataFrame:
        """
        Create a balanced sample from the dataset
        
        Args:
            df: Input DataFrame
            n_samples: Total number of samples to create
            
        Returns:
            pd.DataFrame: Balanced sample
        """
        # Sample equal numbers from each sentiment
        samples_per_sentiment = n_samples // 2
        
        positive_samples = df[df['sentiment'] == 'positive'].sample(
            n=min(samples_per_sentiment, sum(df['sentiment'] == 'positive')),
            random_state=42
        )
        
        negative_samples = df[df['sentiment'] == 'negative'].sample(
            n=min(samples_per_sentiment, sum(df['sentiment'] == 'negative')),
            random_state=42
        )
        
        balanced_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created balanced sample with {len(balanced_df):,} reviews")
        print(f"Sentiment distribution: {balanced_df['sentiment'].value_counts().to_dict()}")
        
        return balanced_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed dataset"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed dataset"""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """Get comprehensive dataset information"""
        info = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_review_length': df['review_length'].mean(),
            'average_word_count': df['word_count'].mean(),
            'category_distribution': df['product_category'].value_counts().to_dict(),
            'split_distribution': df['split'].value_counts().to_dict() if 'split' in df.columns else None
        }
        return info
    
    def print_dataset_summary(self, df: pd.DataFrame) -> None:
        """Print comprehensive dataset summary"""
        info = self.get_dataset_info(df)
        
        print("=" * 60)
        print("AMAZON REVIEWS DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Reviews: {info['total_reviews']:,}")
        print(f"Average Review Length: {info['average_review_length']:.1f} characters")
        print(f"Average Word Count: {info['average_word_count']:.1f} words")
        
        print(f"\nSentiment Distribution:")
        for sentiment, count in info['sentiment_distribution'].items():
            pct = (count / info['total_reviews']) * 100
            print(f"  {sentiment}: {count:,} ({pct:.1f}%)")
        
        print(f"\nCategory Distribution:")
        for category, count in info['category_distribution'].items():
            pct = (count / info['total_reviews']) * 100
            print(f"  {category}: {count:,} ({pct:.1f}%)")
        
        if info['split_distribution']:
            print(f"\nSplit Distribution:")
            for split, count in info['split_distribution'].items():
                pct = (count / info['total_reviews']) * 100
                print(f"  {split}: {count:,} ({pct:.1f}%)")
        
        print("=" * 60)