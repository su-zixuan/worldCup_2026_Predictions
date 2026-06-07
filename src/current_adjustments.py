from __future__ import annotations

import math

import pandas as pd


ADJUSTMENT_WEIGHTS = {
    "fifa_rank_diff": 0.0020,
    "current_elo_diff": 0.0012,
    "coach_tenure_diff": 0.0200,
    "injury_diff": 0.0300,
    "roster_stability_diff": 0.1500,
    "squad_market_value_diff": 0.0002,
    "world_cup_experience_diff": 0.0150,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20), -20)))


def current_context_score(feature_row: pd.DataFrame) -> float:
    """Return a transparent Team A advantage score from current-context features.

    Positive values favor Team A/home. Negative values favor Team B/away. This
    is intentionally modest because injuries, roster stability, and coach data
    are manually maintained and should not overpower historical performance.
    """
    row = feature_row.iloc[0]
    score = 0.0
    for column, weight in ADJUSTMENT_WEIGHTS.items():
        if column in row:
            score += float(row[column]) * weight
    return float(score)


def apply_current_context_adjustment(prediction: pd.DataFrame, feature_row: pd.DataFrame, strength: float = 0.12) -> pd.DataFrame:
    """Slightly adjust model probabilities using current team context.

    The Random Forest learns from historical match patterns. This post-model
    layer lets the dashboard respond to constantly changing variables such as
    injuries, squad stability, and coach tenure without pretending those manual
    values are perfect historical labels.
    """
    adjusted = prediction.copy()
    score = current_context_score(feature_row)
    home_boost = (_sigmoid(score) - 0.5) * 2.0 * strength

    for idx, outcome in adjusted["Outcome"].items():
        if outcome == "Home Win":
            adjusted.loc[idx, "Probability"] *= 1.0 + home_boost
        elif outcome == "Away Win":
            adjusted.loc[idx, "Probability"] *= 1.0 - home_boost
        else:
            adjusted.loc[idx, "Probability"] *= 1.0 - abs(home_boost) * 0.25

    total = adjusted["Probability"].sum()
    if total > 0:
        adjusted["Probability"] = adjusted["Probability"] / total
    adjusted["Context Score"] = score
    return adjusted.sort_values("Probability", ascending=False).reset_index(drop=True)
