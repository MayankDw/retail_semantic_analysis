"""
Data loading utilities for retail sentiment analysis
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import kaggle
from typing import Optional, Tuple

class RetailDataLoader:
    """Class for loading and managing retail review datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_amazon_reviews(self, dataset_name: str = "bittlingmayer/amazonreviews") -> bool:
        """
        Download Amazon reviews dataset from Kaggle
        
        Args:
            dataset_name: Kaggle dataset identifier
            
        Returns:
            bool: Success status
        """
        try:
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=self.raw_dir, 
                unzip=True
            )
            print(f"Successfully downloaded {dataset_name}")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def load_amazon_reviews(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Amazon reviews dataset
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path is None:
            # Look for common Amazon review file names
            possible_files = [
                "train.ft.txt",
                "test.ft.txt", 
                "amazon_reviews.csv",
                "reviews.csv"
            ]
            
            for filename in possible_files:
                filepath = self.raw_dir / filename
                if filepath.exists():
                    file_path = filepath
                    break
            
            if file_path is None:
                raise FileNotFoundError("No Amazon reviews file found. Please download dataset first.")
        
        # Handle different file formats
        if str(file_path).endswith('.txt'):
            return self._load_fasttext_format(file_path)
        else:
            return pd.read_csv(file_path)
    
    def _load_fasttext_format(self, file_path: str) -> pd.DataFrame:
        """Load FastText format files (Amazon reviews)"""
        reviews = []
        sentiments = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # FastText format: __label__1 or __label__2 followed by text
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        label = parts[0]
                        text = parts[1]
                        
                        # Convert label to sentiment
                        sentiment = 'positive' if label == '__label__2' else 'negative'
                        
                        reviews.append(text)
                        sentiments.append(sentiment)
        
        return pd.DataFrame({
            'review_text': reviews,
            'sentiment': sentiments
        })
    
    def create_sample_dataset(self, size: int = 10000) -> pd.DataFrame:
        """
        Create a sample dataset for testing purposes
        
        Args:
            size: Number of samples to create
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        np.random.seed(42)
        
        # Sample positive and negative review templates
        positive_templates = [
            "This product is amazing! I love it so much.",
            "Excellent quality and fast shipping. Highly recommend!",
            "Perfect item, exactly as described. Five stars!",
            "Great value for money. Will buy again.",
            "Outstanding customer service and product quality."
        ]
        
        negative_templates = [
            "Poor quality product. Very disappointed.",
            "Terrible experience. Product broke immediately.",
            "Not as described. Waste of money.",
            "Slow shipping and damaged item.",
            "Would not recommend. Poor customer service."
        ]
        
        reviews = []
        sentiments = []
        ratings = []
        
        for i in range(size):
            if np.random.random() < 0.6:  # 60% positive reviews
                review = np.random.choice(positive_templates)
                sentiment = 'positive'
                rating = np.random.choice([4, 5])
            else:
                review = np.random.choice(negative_templates)
                sentiment = 'negative'
                rating = np.random.choice([1, 2])
            
            reviews.append(review)
            sentiments.append(sentiment)
            ratings.append(rating)
        
        df = pd.DataFrame({
            'review_text': reviews,
            'sentiment': sentiments,
            'rating': ratings,
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], size)
        })
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed dataset"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed dataset"""
        filepath = self.processed_dir / filename
        return pd.read_csv(filepath)