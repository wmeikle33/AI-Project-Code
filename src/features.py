from __future__ import annotations
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

def build_vectorizer():
    return TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))

def fit_transform(vectorizer, X_train):
    Xv = vectorizer.fit_transform(X_train)
    return Xv, vectorizer

def transform(vectorizer, X):
    return vectorizer.transform(X)

def save_vectorizer(vectorizer, path):
    dump(vectorizer, path)

def load_vectorizer(path):
    return load(path)
