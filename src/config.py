from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    random_state: int = 0
    text_columns: tuple = (
        "title", "location", "department", "company_profile", "description",
        "requirements", "benefits", "employment_type", "required_experience",
        "required_education", "industry", "function"
    )
    label_column: str = "fraudulent"
    artifacts_dir: Path = Path("models")
    vectorizer_path: Path = artifacts_dir / "tfidf.joblib"
    model_path: Path = artifacts_dir / "model.joblib"
