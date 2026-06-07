import pandas as pd

from src.data_loader import add_outcome
from src.features import build_training_table


def test_build_training_table_creates_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "home_score": [1, 0, 2],
            "away_score": [0, 0, 3],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "city": ["X", "X", "X"],
            "country": ["Y", "Y", "Y"],
            "neutral": [True, True, True],
        }
    )
    training = build_training_table(add_outcome(df), start_year=2020)
    assert len(training) == 3
    assert "home_win_rate" in training.columns
    assert "outcome" in training.columns
