#!/usr/bin/env python
"""Train a text classifier on the Fake Job Postings dataset."""
import argparse
from pathlib import Path
from src.config import Config
from src import data, preprocess, features, models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/raw/fake_job_postings.csv", help="Path to CSV dataset")
    ap.add_argument("--model", choices=["logreg", "nb"], default="logreg")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha for MultinomialNB")
    ap.add_argument("--C", type=float, default=1.0, help="C for LogisticRegression")
    args = ap.parse_args()

    cfg = Config()
    df = data.load_fake_jobs(args.csv)
    df = data.make_text_field(df, cfg.text_columns)
    # Basic clean + optional lemmatization
    X_text = preprocess.basic_cleanup(df["textdata"]).fillna("")
    # X_text = preprocess.spacy_lemmatize(X_text)  # uncomment if spaCy installed

    X_train, X_test, y_train, y_test = data.train_test_split_df(df.assign(textdata=X_text), cfg.label_column, test_size=0.2, random_state=cfg.random_state)

    vec = features.build_vectorizer()
    X_train_vec, vec = features.fit_transform(vec, X_train)
    X_test_vec = features.transform(vec, X_test)

    if args.model == "logreg":
        clf = models.train_logreg(X_train_vec, y_train, C=args.C)
    else:
        clf = models.train_nb(X_train_vec, y_train, alpha=args.alpha)

    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    features.save_vectorizer(vec, cfg.vectorizer_path)
    models.save_model(clf, cfg.model_path)

    metrics_train = models.evaluate(clf, X_train_vec, y_train)
    metrics_test = models.evaluate(clf, X_test_vec, y_test)

    print("Saved vectorizer →", cfg.vectorizer_path)
    print("Saved model     →", cfg.model_path)
    print("Train:", {k: round(v,4) if isinstance(v,(int,float)) and v is not None else v for k,v in metrics_train.items() if k!="report"})
    print("Test:",  {k: round(v,4) if isinstance(v,(int,float)) and v is not None else v for k,v in metrics_test.items() if k!="report"})

if __name__ == "__main__":
    main()
