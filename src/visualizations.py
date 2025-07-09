"""
Publication-ready visualizations for retail semantic analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RetailVisualizationGenerator:
    """Generate publication-ready visualizations for retail analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization generator
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#4682B4',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c'
        }
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, 
                                  sentiment_column: str = 'sentiment_sentiment',
                                  title: str = "Sentiment Distribution in Customer Reviews",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create sentiment distribution plot
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column with sentiment labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sentiment_counts = df[sentiment_column].value_counts()
        colors = [self.colors.get(sent, self.colors['primary']) for sent in sentiment_counts.index]
        
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
        ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment', fontsize=12)
        ax1.set_ylabel('Number of Reviews', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(sentiment_counts.values, 
                                          labels=sentiment_counts.index,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title(f'{title} - Percentages', fontsize=14, fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sentiment_by_category(self, df: pd.DataFrame,
                                  category_column: str = 'product_category',
                                  sentiment_column: str = 'sentiment_sentiment',
                                  title: str = "Sentiment Distribution by Product Category",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create sentiment distribution by category plot
        
        Args:
            df: DataFrame with sentiment and category data
            category_column: Column with categories
            sentiment_column: Column with sentiment labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df[category_column], df[sentiment_column], normalize='index')
        
        # Create stacked bar chart
        crosstab.plot(kind='bar', ax=ax, stacked=True, 
                     color=[self.colors.get(col, self.colors['primary']) for col in crosstab.columns])
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Product Category', fontsize=12)
        ax.set_ylabel('Proportion of Reviews', fontsize=12)
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_topic_distribution(self, topic_summary: pd.DataFrame,
                              title: str = "Topic Distribution in Customer Reviews",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create topic distribution plot
        
        Args:
            topic_summary: DataFrame with topic information
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Horizontal bar chart
        y_pos = np.arange(len(topic_summary))
        bars = ax1.barh(y_pos, topic_summary['percentage'], 
                       color=plt.cm.Set3(np.linspace(0, 1, len(topic_summary))))
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"Topic {i}: {words[:30]}..." 
                            for i, words in enumerate(topic_summary['top_words'])])
        ax1.set_xlabel('Percentage of Reviews', fontsize=12)
        ax1.set_title(f'{title} - Bar Chart', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=10, fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(topic_summary['percentage'], 
                                          labels=[f"Topic {i}" for i in range(len(topic_summary))],
                                          autopct='%1.1f%%',
                                          colors=plt.cm.Set3(np.linspace(0, 1, len(topic_summary))))
        ax2.set_title(f'{title} - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_word_cloud(self, text_data: str, 
                         title: str = "Word Cloud",
                         save_path: Optional[str] = None) -> WordCloud:
        """
        Create word cloud visualization
        
        Args:
            text_data: Combined text data
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            WordCloud object
        """
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            collocations=False
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return wordcloud
    
    def plot_sentiment_confidence(self, df: pd.DataFrame,
                                 confidence_column: str = 'sentiment_confidence',
                                 sentiment_column: str = 'sentiment_sentiment',
                                 title: str = "Sentiment Confidence Distribution",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create sentiment confidence distribution plot
        
        Args:
            df: DataFrame with sentiment and confidence data
            confidence_column: Column with confidence scores
            sentiment_column: Column with sentiment labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create violin plot
        sentiments = df[sentiment_column].unique()
        colors = [self.colors.get(sent, self.colors['primary']) for sent in sentiments]
        
        parts = ax.violinplot([df[df[sentiment_column] == sent][confidence_column] 
                              for sent in sentiments], 
                             positions=range(len(sentiments)), 
                             widths=0.6, showmeans=True)
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(sentiments)))
        ax.set_xticklabels(sentiments)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_topic_words_heatmap(self, topic_words: List[List[str]],
                               title: str = "Topic-Word Association Heatmap",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create topic-word association heatmap
        
        Args:
            topic_words: List of word lists for each topic
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Create binary matrix for word presence in topics
        all_words = set()
        for words in topic_words:
            all_words.update(words[:10])  # Top 10 words per topic
        
        all_words = sorted(list(all_words))
        
        matrix = np.zeros((len(topic_words), len(all_words)))
        for i, words in enumerate(topic_words):
            for j, word in enumerate(all_words):
                if word in words:
                    matrix[i, j] = words.index(word) + 1
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(matrix, 
                   xticklabels=all_words,
                   yticklabels=[f"Topic {i}" for i in range(len(topic_words))],
                   cmap='YlOrRd',
                   annot=False,
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Topics', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_sentiment_plot(self, df: pd.DataFrame,
                                        sentiment_column: str = 'sentiment_sentiment',
                                        category_column: str = 'product_category',
                                        title: str = "Interactive Sentiment Analysis",
                                        save_path: Optional[str] = None):
        """
        Create interactive sentiment visualization using Plotly
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column with sentiment labels
            category_column: Column with categories
            title: Plot title
            save_path: Optional path to save HTML file
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment by Category',
                           'Sentiment Confidence', 'Review Length Distribution'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Sentiment distribution
        sentiment_counts = df[sentiment_column].value_counts()
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  name='Sentiment Count', marker_color=['green', 'red', 'blue']),
            row=1, col=1
        )
        
        # Sentiment by category
        if category_column in df.columns:
            category_sentiment = df.groupby([category_column, sentiment_column]).size().unstack(fill_value=0)
            for sentiment in category_sentiment.columns:
                fig.add_trace(
                    go.Bar(x=category_sentiment.index, y=category_sentiment[sentiment],
                          name=f'{sentiment}'),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame,
                                     topic_summary: pd.DataFrame,
                                     sentiment_column: str = 'sentiment_sentiment',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive analysis dashboard
        
        Args:
            df: DataFrame with analysis results
            topic_summary: DataFrame with topic information
            sentiment_column: Column with sentiment labels
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts = df[sentiment_column].value_counts()
        colors = [self.colors.get(sent, self.colors['primary']) for sent in sentiment_counts.index]
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
               colors=colors, autopct='%1.1f%%')
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Topic distribution
        ax2 = fig.add_subplot(gs[0, 1])
        y_pos = np.arange(len(topic_summary))
        ax2.barh(y_pos, topic_summary['percentage'][:5], 
                color=plt.cm.Set3(np.linspace(0, 1, 5)))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"T{i}" for i in range(5)])
        ax2.set_title('Top 5 Topics', fontsize=14, fontweight='bold')
        
        # 3. Review length distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'review_text_clean_word_count' in df.columns:
            ax3.hist(df['review_text_clean_word_count'], bins=30, 
                    color=self.colors['primary'], alpha=0.7)
            ax3.set_title('Review Length Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Word Count')
            ax3.set_ylabel('Frequency')
        
        # 4. Sentiment confidence
        ax4 = fig.add_subplot(gs[1, :])
        if 'sentiment_confidence' in df.columns:
            sentiments = df[sentiment_column].unique()
            for i, sentiment in enumerate(sentiments):
                data = df[df[sentiment_column] == sentiment]['sentiment_confidence']
                ax4.hist(data, bins=20, alpha=0.7, label=sentiment,
                        color=self.colors.get(sentiment, self.colors['primary']))
            ax4.set_title('Sentiment Confidence Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        # 5. Word cloud space
        ax5 = fig.add_subplot(gs[2, :])
        ax5.text(0.5, 0.5, 'Word Cloud Placeholder\n(Generate separately)', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # Main title
        fig.suptitle('Retail Sentiment Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, df: pd.DataFrame, topic_summary: pd.DataFrame,
                      output_dir: str = "figures/") -> None:
        """
        Generate and save all visualization plots
        
        Args:
            df: DataFrame with analysis results
            topic_summary: DataFrame with topic information
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_sentiment_distribution(df, save_path=f"{output_dir}/sentiment_distribution.png")
        
        if 'product_category' in df.columns:
            self.plot_sentiment_by_category(df, save_path=f"{output_dir}/sentiment_by_category.png")
        
        self.plot_topic_distribution(topic_summary, save_path=f"{output_dir}/topic_distribution.png")
        
        if 'sentiment_confidence' in df.columns:
            self.plot_sentiment_confidence(df, save_path=f"{output_dir}/sentiment_confidence.png")
        
        self.create_comprehensive_dashboard(df, topic_summary, 
                                          save_path=f"{output_dir}/dashboard.png")
        
        print(f"All plots saved to {output_dir}")