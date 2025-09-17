from __future__ import annotations
import re, string
from typing import Iterable, List
import pandas as pd

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    _NLP = None  # spaCy optional

def basic_cleanup(texts: Iterable[str]) -> pd.Series:
    """Lowercase, strip HTML tags, remove URLs and punctuation/numbers."""
    def clean(t: str) -> str:
        t = t.lower()
        t = re.sub(r"https?://\S+|www\.[^\s]+", " ", t)
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    return pd.Series([clean(t) for t in texts])

def spacy_lemmatize(texts: Iterable[str]) -> pd.Series:
    """Optional: lemmatize with spaCy if model is available."""
    if _NLP is None:
        return pd.Series(list(texts))
    docs = _NLP.pipe(texts, batch_size=256)
    out: List[str] = []
    for doc in docs:
        toks = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
        out.append(" ".join(toks))
    return pd.Series(out)
