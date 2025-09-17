# AI Project Code
This is a code from a project I completed in my Artificial Intelligence course at Beijing University, in which we used different machine learning techniques and applied them to an existing research problem. For this project, we analyzed data related job postings in order to create a model to classify them as fake or real. This was a supervised learning task. Specifically, we incorporated several different machine learning algorithsm to accomplish our task, including Logistic Regression and Bayesian Probability, amoungst others. The dataset we used was from Kaggle.com and consisted of 18,000 job descriptions out of which 866 were fake. The original data included both textual information and meta-information, with 18 distinct features. 5 of these features were longer texts and the other 13 were numeric fields or categorical data. The column “Fraudulent” distinguishes between real and fake job postings (value of 1 indicates a fake job posting, while 0 denotes a real job). This is another one of my earlier Python projects, helping to demonstrate my development over time.

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
