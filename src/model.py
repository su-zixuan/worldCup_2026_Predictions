from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

FEATURES = FEATURE_COLUMNS
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_outcome_model.joblib"


def train_model(training_df: pd.DataFrame, save_path: Path = MODEL_PATH) -> tuple[Pipeline, dict]:
    data = training_df.dropna(subset=FEATURES + ["outcome"]).copy()
    X = data[FEATURES]
    y = data["outcome"]

    test_size = 0.2
    estimated_test_rows = max(1, int(round(len(data) * test_size)))
    enough_rows_for_stratify = estimated_test_rows >= y.nunique() and y.value_counts().min() >= 2
    stratify = y if enough_rows_for_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "rows_used": int(len(data)),
        "feature_count": int(len(FEATURES)),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model, metrics


def load_model(path: Path = MODEL_PATH) -> Pipeline | None:
    if path.exists():
        return joblib.load(path)
    return None


def predict_match(model: Pipeline, feature_row: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(feature_row[FEATURES])[0]
    labels = model.classes_
    out = pd.DataFrame({"Outcome": labels, "Probability": probs})
    return out.sort_values("Probability", ascending=False).reset_index(drop=True)
