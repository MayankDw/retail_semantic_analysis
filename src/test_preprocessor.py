import pytest
import pandas as pd
from src.preprocessor import RetailTextPreprocessor

def test_preprocess_dataframe():
    preprocessor = RetailTextPreprocessor()
    df = pd.DataFrame({'review_text': ['This is a great product!', '', None, '   ']})
    df_processed = preprocessor.preprocess_dataframe(df, text_column='review_text')
    assert len(df_processed) == 1
    assert df_processed['review_text_clean'].iloc[0] == 'great product'

def test_remove_stopwords():
    preprocessor = RetailTextPreprocessor()
    assert preprocessor.remove_stopwords('This is a test') == 'test'
    assert preprocessor.remove_stopwords('') == ''
    assert preprocessor.remove_stopwords(None) == None
    assert preprocessor.remove_stopwords('   ') == ''