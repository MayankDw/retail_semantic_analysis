"""
Topic modeling implementation for retail reviews
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RetailTopicModeler:
    """Comprehensive topic modeling for retail reviews"""
    
    def __init__(self, method: str = 'lda'):
        """
        Initialize topic modeler
        
        Args:
            method: Topic modeling method ('lda', 'nmf', 'gensim_lda')
        """
        self.method = method
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.dictionary = None
        self.corpus = None
        self.coherence_scores = []
    
    def prepare_data(self, texts: List[str], 
                    method: str = 'tfidf',
                    max_features: int = 5000,
                    min_df: int = 2,
                    max_df: float = 0.8) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare text data for topic modeling
        
        Args:
            texts: List of preprocessed texts
            method: Vectorization method ('tfidf', 'count')
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Tuple of feature matrix and feature names
        """
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return doc_term_matrix, self.feature_names
    
    def prepare_gensim_data(self, texts: List[str]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        Prepare data for Gensim topic modeling
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Tuple of dictionary and corpus
        """
        # Split texts into tokens
        texts_tokens = [text.split() for text in texts]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(texts_tokens)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=2, no_above=0.8)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts_tokens]
        
        return self.dictionary, self.corpus
    
    def find_optimal_topics(self, texts: List[str], 
                          max_topics: int = 20,
                          step: int = 2) -> Dict[int, float]:
        """
        Find optimal number of topics using coherence score
        
        Args:
            texts: List of preprocessed texts
            max_topics: Maximum number of topics to test
            step: Step size for topic range
            
        Returns:
            Dict mapping number of topics to coherence scores
        """
        if self.method == 'gensim_lda':
            return self._find_optimal_topics_gensim(texts, max_topics, step)
        else:
            return self._find_optimal_topics_sklearn(texts, max_topics, step)
    
    def _find_optimal_topics_gensim(self, texts: List[str], 
                                   max_topics: int, step: int) -> Dict[int, float]:
        """Find optimal topics using Gensim coherence"""
        # Prepare data
        texts_tokens = [text.split() for text in texts]
        dictionary = corpora.Dictionary(texts_tokens)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in texts_tokens]
        
        coherence_scores = {}
        
        for num_topics in range(2, max_topics + 1, step):
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts_tokens,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            coherence_scores[num_topics] = coherence_score
            
            print(f"Topics: {num_topics}, Coherence: {coherence_score:.4f}")
        
        return coherence_scores
    
    def _find_optimal_topics_sklearn(self, texts: List[str], 
                                    max_topics: int, step: int) -> Dict[int, float]:
        """Find optimal topics using sklearn methods"""
        doc_term_matrix, _ = self.prepare_data(texts)
        
        coherence_scores = {}
        
        for num_topics in range(2, max_topics + 1, step):
            if self.method == 'lda':
                model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=10
                )
            else:  # nmf
                model = NMF(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=200
                )
            
            model.fit(doc_term_matrix)
            
            # Calculate perplexity for LDA (lower is better)
            if self.method == 'lda':
                perplexity = model.perplexity(doc_term_matrix)
                coherence_scores[num_topics] = -perplexity  # Negative for consistency
            else:
                # For NMF, use reconstruction error
                reconstruction_error = model.reconstruction_err_
                coherence_scores[num_topics] = -reconstruction_error
            
            print(f"Topics: {num_topics}, Score: {coherence_scores[num_topics]:.4f}")
        
        return coherence_scores
    
    def fit_topic_model(self, texts: List[str], 
                       num_topics: int = 10,
                       **kwargs) -> None:
        """
        Fit topic model to texts
        
        Args:
            texts: List of preprocessed texts
            num_topics: Number of topics
            **kwargs: Additional arguments for specific models
        """
        if self.method == 'gensim_lda':
            self._fit_gensim_lda(texts, num_topics, **kwargs)
        elif self.method == 'lda':
            self._fit_sklearn_lda(texts, num_topics, **kwargs)
        elif self.method == 'nmf':
            self._fit_sklearn_nmf(texts, num_topics, **kwargs)
    
    def _fit_gensim_lda(self, texts: List[str], num_topics: int, **kwargs):
        """Fit Gensim LDA model"""
        self.dictionary, self.corpus = self.prepare_gensim_data(texts)
        
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=kwargs.get('passes', 10),
            alpha=kwargs.get('alpha', 'auto'),
            per_word_topics=True
        )
    
    def _fit_sklearn_lda(self, texts: List[str], num_topics: int, **kwargs):
        """Fit sklearn LDA model"""
        doc_term_matrix, self.feature_names = self.prepare_data(texts, method='count')
        
        self.model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=kwargs.get('max_iter', 10),
            learning_method='batch'
        )
        
        self.model.fit(doc_term_matrix)
    
    def _fit_sklearn_nmf(self, texts: List[str], num_topics: int, **kwargs):
        """Fit NMF model"""
        doc_term_matrix, self.feature_names = self.prepare_data(texts, method='tfidf')
        
        self.model = NMF(
            n_components=num_topics,
            random_state=42,
            max_iter=kwargs.get('max_iter', 200),
            alpha=kwargs.get('alpha', 0.1),
            l1_ratio=kwargs.get('l1_ratio', 0.5)
        )
        
        self.model.fit(doc_term_matrix)
    
    def get_topic_words(self, num_words: int = 10) -> List[List[str]]:
        """
        Get top words for each topic
        
        Args:
            num_words: Number of words per topic
            
        Returns:
            List of lists containing top words for each topic
        """
        if self.method == 'gensim_lda':
            return self._get_gensim_topic_words(num_words)
        else:
            return self._get_sklearn_topic_words(num_words)
    
    def _get_gensim_topic_words(self, num_words: int) -> List[List[str]]:
        """Get topic words from Gensim model"""
        topics = []
        for topic_id in range(self.model.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append([word for word, _ in topic_words])
        return topics
    
    def _get_sklearn_topic_words(self, num_words: int) -> List[List[str]]:
        """Get topic words from sklearn model"""
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            topic_words = [self.feature_names[i] for i in top_words_idx]
            topics.append(topic_words)
        return topics
    
    def get_document_topics(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """
        Get topic distributions for documents
        
        Args:
            texts: List of texts
            
        Returns:
            List of topic distributions for each document
        """
        if self.method == 'gensim_lda':
            return self._get_gensim_document_topics(texts)
        else:
            return self._get_sklearn_document_topics(texts)
    
    def _get_gensim_document_topics(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """Get document topics from Gensim model"""
        texts_tokens = [text.split() for text in texts]
        doc_topics = []
        
        for tokens in texts_tokens:
            bow = self.dictionary.doc2bow(tokens)
            topics = self.model.get_document_topics(bow)
            doc_topics.append(topics)
        
        return doc_topics
    
    def _get_sklearn_document_topics(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """Get document topics from sklearn model"""
        doc_term_matrix = self.vectorizer.transform(texts)
        doc_topic_dist = self.model.transform(doc_term_matrix)
        
        doc_topics = []
        for dist in doc_topic_dist:
            topics = [(i, prob) for i, prob in enumerate(dist) if prob > 0.01]
            topics = sorted(topics, key=lambda x: x[1], reverse=True)
            doc_topics.append(topics)
        
        return doc_topics
    
    def assign_dominant_topics(self, texts: List[str]) -> List[int]:
        """
        Assign dominant topic to each document
        
        Args:
            texts: List of texts
            
        Returns:
            List of dominant topic indices
        """
        doc_topics = self.get_document_topics(texts)
        dominant_topics = []
        
        for topics in doc_topics:
            if topics:
                dominant_topic = max(topics, key=lambda x: x[1])[0]
                dominant_topics.append(dominant_topic)
            else:
                dominant_topics.append(-1)  # No dominant topic
        
        return dominant_topics
    
    def create_topic_summary(self, texts: List[str], 
                           topic_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create comprehensive topic summary
        
        Args:
            texts: List of texts
            topic_names: Optional custom topic names
            
        Returns:
            DataFrame with topic summary
        """
        topic_words = self.get_topic_words()
        dominant_topics = self.assign_dominant_topics(texts)
        
        # Count documents per topic
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        
        summary_data = []
        for topic_id, words in enumerate(topic_words):
            topic_name = topic_names[topic_id] if topic_names else f"Topic {topic_id}"
            doc_count = topic_counts.get(topic_id, 0)
            
            summary_data.append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'top_words': ', '.join(words[:5]),
                'all_words': ', '.join(words),
                'document_count': doc_count,
                'percentage': (doc_count / len(texts)) * 100
            })
        
        return pd.DataFrame(summary_data)
    
    def create_topic_wordcloud(self, topic_id: int, 
                             save_path: Optional[str] = None) -> WordCloud:
        """
        Create word cloud for specific topic
        
        Args:
            topic_id: Topic ID
            save_path: Optional path to save image
            
        Returns:
            WordCloud object
        """
        if self.method == 'gensim_lda':
            topic_words = dict(self.model.show_topic(topic_id, topn=50))
        else:
            topic = self.model.components_[topic_id]
            top_words_idx = topic.argsort()[-50:][::-1]
            topic_words = {
                self.feature_names[idx]: topic[idx] 
                for idx in top_words_idx
            }
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(topic_words)
        
        if save_path:
            wordcloud.to_file(save_path)
        
        return wordcloud
    
    def visualize_topics(self, save_path: Optional[str] = None):
        """
        Create interactive topic visualization
        
        Args:
            save_path: Optional path to save HTML file
        """
        if self.method == 'gensim_lda':
            vis = pyLDAvis.gensim_models.prepare(
                self.model, self.corpus, self.dictionary
            )
            
            if save_path:
                pyLDAvis.save_html(vis, save_path)
            
            return vis
        else:
            print("Interactive visualization only available for Gensim LDA")
            return None
    
    def topic_evolution(self, df: pd.DataFrame, 
                       texts: List[str],
                       date_column: str = 'date',
                       periods: int = 12) -> pd.DataFrame:
        """
        Analyze topic evolution over time
        
        Args:
            df: DataFrame with date information
            texts: List of texts
            date_column: Column with dates
            periods: Number of time periods
            
        Returns:
            DataFrame with topic evolution
        """
        df_copy = df.copy()
        df_copy['dominant_topic'] = self.assign_dominant_topics(texts)
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # Create time periods
        df_copy['period'] = pd.cut(
            df_copy[date_column], 
            bins=periods, 
            labels=False
        )
        
        # Calculate topic proportions by period
        evolution = df_copy.groupby(['period', 'dominant_topic']).size().unstack(fill_value=0)
        evolution = evolution.div(evolution.sum(axis=1), axis=0)
        
        return evolution