#!/usr/bin/env python3
"""
Setup script to download required NLTK data
Run this script before using the demo notebook
"""

import nltk
import ssl

def download_nltk_data():
    """Download all required NLTK data"""
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK resources
    nltk_resources = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'vader_lexicon',
        'omw-1.4'
    ]
    
    print("Downloading NLTK resources...")
    
    for resource in nltk_resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=False)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"⚠ Warning: Could not download {resource}: {e}")
    
    print("\n" + "="*50)
    print("NLTK setup complete!")
    print("You can now run the demo notebook.")
    print("="*50)

if __name__ == "__main__":
    download_nltk_data()