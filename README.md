# Fake Job Postings — Text Classifier

This repo splits a notebook-style script into a clean, modular project:

```
fake-jobs-detector/
├── src/
│   ├── config.py            # constants & artifact paths
│   ├── data.py              # load data, build text field, train/test split
│   ├── preprocess.py        # text cleaning, optional spaCy lemmatization
│   ├── features.py          # TF-IDF vectorizer build/save/load
│   └── models.py            # train/evaluate and save/load models
├── scripts/
│   ├── train.py             # CLI: train a model and save artifacts
│   └── predict.py           # CLI: load model and score new texts
├── data/raw/                # put fake_job_postings.csv here (gitignored)
├── models/                  # saved artifacts (gitignored)
├── reports/figures/         # plots if you add EDA (gitignored)
├── tests/                   # add unit tests here
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (Optional) spaCy English model for lemmatization
python -m spacy download en_core_web_sm
```

## Data
Download **fake_job_postings.csv** from Kaggle and place it at `data/raw/fake_job_postings.csv`.

## Train
```bash
python scripts/train.py --model logreg
# or
python scripts/train.py --model nb --alpha 0.5
```

## Predict
```bash
python scripts/predict.py --model models/model.joblib --vec models/tfidf.joblib "Sample job text here"
```

## Notes
- The original monolithic script mixed EDA, cleaning, and modeling. This refactor isolates concerns, so you can add tests and iterate.- Extend `preprocess.py` with richer normalization (HTML stripping, language detection, etc.).- Swap TF-IDF with embeddings later if you want.
