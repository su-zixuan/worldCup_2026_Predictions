from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_results.csv"


def load_results(use_online: bool = True) -> pd.DataFrame:
    """Load international football match results.

    The online source is a public historical men's international results CSV.
    If online loading fails, the app falls back to a tiny sample dataset so the
    dashboard still runs for demos.
    """
    try:
        if use_online:
            df = pd.read_csv(DATA_URL)
        else:
            df = pd.read_csv(SAMPLE_PATH)
    except Exception:
        df = pd.read_csv(SAMPLE_PATH)

    required = {
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    return df.sort_values("date").reset_index(drop=True)


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["outcome"] = df["goal_diff"].map(lambda x: "Home Win" if x > 0 else "Away Win" if x < 0 else "Draw")
    df["total_goals"] = df["home_score"] + df["away_score"]
    return df
