from __future__ import annotations
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from joblib import dump, load

def train_logreg(X, y, C: float = 1.0):
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.C = C
    clf.fit(X, y)
    return clf

def train_nb(X, y, alpha: float = 1.0):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X, y)
    return clf

def evaluate(model, X, y) -> dict:
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        y_prob = proba(X)[:,1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = None
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    return {"accuracy": acc, "auc": auc, "report": report}

def save_model(model, path):
    dump(model, path)

def load_model(path):
    return load(path)
