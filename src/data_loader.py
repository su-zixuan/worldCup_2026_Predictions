from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_results.csv"
CURRENT_TEAM_STATS_PATH = ROOT / "data" / "current_team_stats.csv"

CURRENT_TEAM_STAT_COLUMNS = [
    "team",
    "fifa_rank",
    "current_elo_rating",
    "coach_tenure_years",
    "avg_squad_age",
    "injured_players",
    "roster_stability",
    "squad_market_value_m",
    "world_cup_experience",
]

CURRENT_TEAM_DEFAULTS = {
    "fifa_rank": 100.0,
    "current_elo_rating": 1500.0,
    "coach_tenure_years": 1.0,
    "avg_squad_age": 27.0,
    "injured_players": 0.0,
    "roster_stability": 0.50,
    "squad_market_value_m": 250.0,
    "world_cup_experience": 0.0,
}


def load_results(use_online: bool = True) -> pd.DataFrame:
    """Load international football match results.

    The online source is a public historical men's international results CSV.
    If online loading fails, the app falls back to the included sample dataset so
    the dashboard still runs for demos and class presentations.
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


def load_current_team_stats(teams: list[str] | None = None, path: Path = CURRENT_TEAM_STATS_PATH) -> pd.DataFrame:
    """Load editable current-team context for V2 predictions.

    This file is intentionally local and editable because player injuries,
    coaches, squads, rankings, and roster stability change often. Missing teams
    are filled with neutral default values so any team in the historical dataset
    can still be predicted.
    """
    if path.exists():
        stats = pd.read_csv(path)
    else:
        stats = pd.DataFrame(columns=CURRENT_TEAM_STAT_COLUMNS)

    missing_columns = set(CURRENT_TEAM_STAT_COLUMNS).difference(stats.columns)
    if missing_columns:
        raise ValueError(f"current_team_stats.csv is missing columns: {sorted(missing_columns)}")

    stats = stats[CURRENT_TEAM_STAT_COLUMNS].copy()
    stats["team"] = stats["team"].astype(str)

    for column, default in CURRENT_TEAM_DEFAULTS.items():
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(default)

    stats = stats.drop_duplicates(subset="team", keep="last")

    if teams is not None:
        existing = set(stats["team"])
        rows = []
        for team in sorted(set(teams)):
            if team not in existing:
                row = {"team": team}
                row.update(CURRENT_TEAM_DEFAULTS)
                rows.append(row)
        if rows:
            stats = pd.concat([stats, pd.DataFrame(rows)], ignore_index=True)

    return stats.sort_values("team").reset_index(drop=True)


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["outcome"] = df["goal_diff"].map(lambda x: "Home Win" if x > 0 else "Away Win" if x < 0 else "Draw")
    df["total_goals"] = df["home_score"] + df["away_score"]
    return df
