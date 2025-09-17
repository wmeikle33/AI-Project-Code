#!/usr/bin/env python
"""Load saved model + vectorizer and score new text inputs."""
import argparse
from src.features import load_vectorizer, transform
from src.models import load_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/model.joblib")
    ap.add_argument("--vec", type=str, default="models/tfidf.joblib")
    ap.add_argument("texts", nargs="+", help="One or more job posting texts to score")
    args = ap.parse_args()

    vec = load_vectorizer(args.vec)
    clf = load_model(args.model)

    X = transform(vec, args.texts)
    proba = getattr(clf, "predict_proba")(X)[:,1]
    for t, p in zip(args.texts, proba):
        print(f"{p:.4f}\t{t[:80].replace('\n',' ')}...")

if __name__ == "__main__":
    main()
