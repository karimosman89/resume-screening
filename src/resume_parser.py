import os
import re
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Spacy's English model for text processing
nlp = spacy.load("en_core_web_sm")

def load_resumes(data_path):
    """
    Load resumes from a specified folder.
    Assumes resumes are in plain text format.
    """
    resumes = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r", encoding="utf-8") as file:
                resumes.append(file.read())
    return resumes

def preprocess_text(text):
    """
    Preprocess text data by cleaning and tokenizing.
    - Convert to lowercase
    - Remove special characters
    - Tokenize and remove stopwords
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def preprocess_resumes(resumes):
    """
    Apply preprocessing to each resume in the dataset.
    """
    return [preprocess_text(resume) for resume in resumes]

def extract_features(corpus, method="tfidf"):
    """
    Extract features using TF-IDF or another vectorization technique.
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000)
        features = vectorizer.fit_transform(corpus)
    # Placeholder for embeddings if needed in the future
    return features

