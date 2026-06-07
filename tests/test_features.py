from __future__ import annotations

import pandas as pd

from src.data_loader import load_current_team_stats
from src.features import FEATURE_COLUMNS, build_training_table, make_prediction_features


def sample_results() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "home_team": ["A", "B", "A", "C"],
        "away_team": ["B", "A", "C", "A"],
        "home_score": [2, 0, 1, 1],
        "away_score": [1, 0, 3, 2],
        "neutral": [True, True, False, True],
        "tournament": ["Friendly", "Friendly", "Friendly", "Friendly"],
        "city": ["X", "X", "Y", "Z"],
        "country": ["X", "X", "Y", "Z"],
    })


def sample_current_stats() -> pd.DataFrame:
    return pd.DataFrame({
        "team": ["A", "B", "C"],
        "fifa_rank": [10, 20, 30],
        "current_elo_rating": [1600, 1500, 1450],
        "coach_tenure_years": [2.0, 1.0, 0.5],
        "avg_squad_age": [27.0, 26.5, 28.0],
        "injured_players": [1, 3, 0],
        "roster_stability": [0.8, 0.5, 0.6],
        "squad_market_value_m": [500, 300, 200],
        "world_cup_experience": [5, 2, 1],
    })


def test_build_training_table_contains_v2_features():
    training = build_training_table(sample_results(), start_year=2020, current_stats=sample_current_stats())
    assert len(training) == 4
    for column in FEATURE_COLUMNS:
        assert column in training.columns
    assert "fifa_rank_diff" in training.columns
    assert "elo_diff" in training.columns
    assert "outcome" in training.columns


def test_make_prediction_features_contains_current_context():
    features = make_prediction_features(
        sample_results(), "A", "B", neutral=True, current_stats=sample_current_stats()
    )
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "home_fifa_rank"] == 10
    assert features.loc[0, "away_fifa_rank"] == 20
    assert features.loc[0, "fifa_rank_diff"] == 10


def test_load_current_team_stats_adds_missing_teams():
    stats = load_current_team_stats(["Argentina", "Imaginary FC"])
    assert "Imaginary FC" in set(stats["team"])
