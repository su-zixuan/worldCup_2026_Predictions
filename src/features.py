from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "neutral",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_avg_goals_for",
    "away_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_against",
    "home_recent_matches",
    "away_recent_matches",
]


def _team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals_for": df["home_score"],
        "goals_against": df["away_score"],
        "result": df["home_score"].sub(df["away_score"]).map(
            lambda x: "Win" if x > 0 else "Loss" if x < 0 else "Draw"
        ),
    })

    away = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals_for": df["away_score"],
        "goals_against": df["home_score"],
        "result": df["away_score"].sub(df["home_score"]).map(
            lambda x: "Win" if x > 0 else "Loss" if x < 0 else "Draw"
        ),
    })

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    long_df["is_win"] = (long_df["result"] == "Win").astype(int)
    long_df["is_draw"] = (long_df["result"] == "Draw").astype(int)

    return long_df


def _add_rolling_stats(long_df: pd.DataFrame, n_recent: int = 10) -> pd.DataFrame:
    long_df = long_df.copy()

    grouped = long_df.groupby("team", group_keys=False)

    long_df["win_rate"] = grouped["is_win"].apply(
        lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
    )

    long_df["draw_rate"] = grouped["is_draw"].apply(
        lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
    )

    long_df["avg_goals_for"] = grouped["goals_for"].apply(
        lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
    )

    long_df["avg_goals_against"] = grouped["goals_against"].apply(
        lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
    )

    long_df["recent_matches"] = grouped.cumcount().clip(upper=n_recent)

    long_df[["win_rate", "draw_rate", "avg_goals_for", "avg_goals_against"]] = (
        long_df[["win_rate", "draw_rate", "avg_goals_for", "avg_goals_against"]]
        .fillna({
            "win_rate": 0.33,
            "draw_rate": 0.34,
            "avg_goals_for": 1.0,
            "avg_goals_against": 1.0,
        })
    )

    return long_df


def build_training_table(
    df: pd.DataFrame,
    start_year: int = 1994,
    n_recent: int = 10,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    long_df = _team_long_table(df)
    long_df = _add_rolling_stats(long_df, n_recent=n_recent)

    stat_cols = [
        "date",
        "team",
        "win_rate",
        "draw_rate",
        "avg_goals_for",
        "avg_goals_against",
        "recent_matches",
    ]

    home_stats = long_df[stat_cols].rename(columns={
        "team": "home_team",
        "win_rate": "home_win_rate",
        "draw_rate": "home_draw_rate",
        "avg_goals_for": "home_avg_goals_for",
        "avg_goals_against": "home_avg_goals_against",
        "recent_matches": "home_recent_matches",
    })

    away_stats = long_df[stat_cols].rename(columns={
        "team": "away_team",
        "win_rate": "away_win_rate",
        "draw_rate": "away_draw_rate",
        "avg_goals_for": "away_avg_goals_for",
        "avg_goals_against": "away_avg_goals_against",
        "recent_matches": "away_recent_matches",
    })

    training = df.merge(
        home_stats,
        on=["date", "home_team"],
        how="left",
    ).merge(
        away_stats,
        on=["date", "away_team"],
        how="left",
    )

    training = training[training["date"].dt.year >= start_year].copy()

    if "outcome" not in training.columns:
        training["goal_diff"] = training["home_score"] - training["away_score"]
        training["outcome"] = training["goal_diff"].apply(
            lambda x: "Home Win" if x > 0 else "Away Win" if x < 0 else "Draw"
        )

    training["neutral"] = training["neutral"].astype(int)

    return training[
        FEATURE_COLUMNS + ["outcome", "home_team", "away_team", "date", "home_score", "away_score"]
    ].dropna()


def make_prediction_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral: bool = True,
    n_recent: int = 10,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    long_df = _team_long_table(df)

    def latest_stats(team: str) -> dict:
        team_matches = long_df[long_df["team"] == team].tail(n_recent)

        if team_matches.empty:
            return {
                "win_rate": 0.33,
                "draw_rate": 0.34,
                "avg_goals_for": 1.0,
                "avg_goals_against": 1.0,
                "recent_matches": 0,
            }

        return {
            "win_rate": float(team_matches["is_win"].mean()),
            "draw_rate": float(team_matches["is_draw"].mean()),
            "avg_goals_for": float(team_matches["goals_for"].mean()),
            "avg_goals_against": float(team_matches["goals_against"].mean()),
            "recent_matches": int(len(team_matches)),
        }

    home = latest_stats(home_team)
    away = latest_stats(away_team)

    row = {
        "neutral": int(neutral),
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_draw_rate": home["draw_rate"],
        "away_draw_rate": away["draw_rate"],
        "home_avg_goals_for": home["avg_goals_for"],
        "away_avg_goals_for": away["avg_goals_for"],
        "home_avg_goals_against": home["avg_goals_against"],
        "away_avg_goals_against": away["avg_goals_against"],
        "home_recent_matches": home["recent_matches"],
        "away_recent_matches": away["recent_matches"],
    }

    return pd.DataFrame([row])