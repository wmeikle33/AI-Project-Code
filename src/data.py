from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional

def load_fake_jobs(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the 'fake_job_postings.csv' dataset.
    If csv_path is None, expects file at data/raw/fake_job_postings.csv.
    """
    if csv_path is None:
        csv_path = Path("data/raw/fake_job_postings.csv")
    df = pd.read_csv(csv_path)
    return df

def make_text_field(df: pd.DataFrame, text_columns) -> pd.DataFrame:
    # Fill NaNs and concatenate relevant text fields
    df = df.copy()
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna(" ")
        else:
            df[col] = " "  # ensure column exists
    df["textdata"] = df[list(text_columns)].agg(" ".join, axis=1)
    return df

def train_test_split_df(df: pd.DataFrame, label: str, test_size: float = 0.2, random_state: int = 0):
    from sklearn.model_selection import train_test_split
    X = df["textdata"].values
    y = df[label].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
