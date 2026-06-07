from __future__ import annotations

import math

import pandas as pd

from src.data_loader import CURRENT_TEAM_DEFAULTS, load_current_team_stats

BASE_FEATURE_COLUMNS = [
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
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
]

CURRENT_CONTEXT_FEATURE_COLUMNS = [
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "home_current_elo_rating",
    "away_current_elo_rating",
    "current_elo_diff",
    "home_coach_tenure_years",
    "away_coach_tenure_years",
    "coach_tenure_diff",
    "home_avg_squad_age",
    "away_avg_squad_age",
    "avg_squad_age_diff",
    "home_injured_players",
    "away_injured_players",
    "injury_diff",
    "home_roster_stability",
    "away_roster_stability",
    "roster_stability_diff",
    "home_squad_market_value_m",
    "away_squad_market_value_m",
    "squad_market_value_diff",
    "home_world_cup_experience",
    "away_world_cup_experience",
    "world_cup_experience_diff",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CURRENT_CONTEXT_FEATURE_COLUMNS


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


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for < goals_against:
        return 0.0
    return 0.5


def _add_elo_before(df: pd.DataFrame, k_factor: float = 20.0, start_rating: float = 1500.0) -> pd.DataFrame:
    """Add pre-match Elo ratings without using future results."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = ratings.get(home, start_rating)
        away_rating = ratings.get(away, start_rating)
        home_elos.append(home_rating)
        away_elos.append(away_rating)

        expected_home = _expected_score(home_rating, away_rating)
        actual_home = _actual_score(row.home_score, row.away_score)
        rating_change = k_factor * (actual_home - expected_home)
        ratings[home] = home_rating + rating_change
        ratings[away] = away_rating - rating_change

    df["home_elo_before"] = home_elos
    df["away_elo_before"] = away_elos
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    return df


def latest_elo_ratings(df: pd.DataFrame, k_factor: float = 20.0, start_rating: float = 1500.0) -> dict[str, float]:
    """Return the latest Elo rating for every team in the results dataset."""
    ratings: dict[str, float] = {}
    ordered = df.sort_values("date").reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = ratings.get(home, start_rating)
        away_rating = ratings.get(away, start_rating)
        expected_home = _expected_score(home_rating, away_rating)
        actual_home = _actual_score(row.home_score, row.away_score)
        rating_change = k_factor * (actual_home - expected_home)
        ratings[home] = home_rating + rating_change
        ratings[away] = away_rating - rating_change
    return ratings


def _prepare_current_stats(current_stats: pd.DataFrame | None, teams: list[str]) -> pd.DataFrame:
    if current_stats is None:
        current_stats = load_current_team_stats(teams)
    else:
        current_stats = current_stats.copy()

    for column, default in CURRENT_TEAM_DEFAULTS.items():
        if column not in current_stats.columns:
            current_stats[column] = default
        current_stats[column] = pd.to_numeric(current_stats[column], errors="coerce").fillna(default)

    if "team" not in current_stats.columns:
        current_stats["team"] = []

    existing = set(current_stats["team"])
    missing_rows = []
    for team in sorted(set(teams)):
        if team not in existing:
            row = {"team": team}
            row.update(CURRENT_TEAM_DEFAULTS)
            missing_rows.append(row)
    if missing_rows:
        current_stats = pd.concat([current_stats, pd.DataFrame(missing_rows)], ignore_index=True)

    return current_stats.drop_duplicates(subset="team", keep="last")


def _add_current_context_features(training: pd.DataFrame, current_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    teams = sorted(set(training["home_team"]).union(set(training["away_team"])))
    stats = _prepare_current_stats(current_stats, teams)

    home_stats = stats.rename(columns={
        "team": "home_team",
        "fifa_rank": "home_fifa_rank",
        "current_elo_rating": "home_current_elo_rating",
        "coach_tenure_years": "home_coach_tenure_years",
        "avg_squad_age": "home_avg_squad_age",
        "injured_players": "home_injured_players",
        "roster_stability": "home_roster_stability",
        "squad_market_value_m": "home_squad_market_value_m",
        "world_cup_experience": "home_world_cup_experience",
    })
    away_stats = stats.rename(columns={
        "team": "away_team",
        "fifa_rank": "away_fifa_rank",
        "current_elo_rating": "away_current_elo_rating",
        "coach_tenure_years": "away_coach_tenure_years",
        "avg_squad_age": "away_avg_squad_age",
        "injured_players": "away_injured_players",
        "roster_stability": "away_roster_stability",
        "squad_market_value_m": "away_squad_market_value_m",
        "world_cup_experience": "away_world_cup_experience",
    })

    training = training.merge(home_stats, on="home_team", how="left")
    training = training.merge(away_stats, on="away_team", how="left")

    for side in ["home", "away"]:
        for source_column, default in CURRENT_TEAM_DEFAULTS.items():
            column = f"{side}_{source_column}"
            if column not in training.columns:
                # current_elo_rating keeps its original full name; all others follow source column names.
                continue
            training[column] = pd.to_numeric(training[column], errors="coerce").fillna(default)

    # Lower FIFA rank is better, so this is away rank minus home rank: positive favors Team A/home.
    training["fifa_rank_diff"] = training["away_fifa_rank"] - training["home_fifa_rank"]
    training["current_elo_diff"] = training["home_current_elo_rating"] - training["away_current_elo_rating"]
    training["coach_tenure_diff"] = training["home_coach_tenure_years"] - training["away_coach_tenure_years"]
    training["avg_squad_age_diff"] = training["home_avg_squad_age"] - training["away_avg_squad_age"]
    # More injuries are worse, so away injuries minus home injuries: positive favors Team A/home.
    training["injury_diff"] = training["away_injured_players"] - training["home_injured_players"]
    training["roster_stability_diff"] = training["home_roster_stability"] - training["away_roster_stability"]
    training["squad_market_value_diff"] = training["home_squad_market_value_m"] - training["away_squad_market_value_m"]
    training["world_cup_experience_diff"] = training["home_world_cup_experience"] - training["away_world_cup_experience"]
    return training


def build_training_table(
    df: pd.DataFrame,
    start_year: int = 1994,
    n_recent: int = 10,
    current_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = _add_elo_before(df)

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

    training = df.merge(home_stats, on=["date", "home_team"], how="left")
    training = training.merge(away_stats, on=["date", "away_team"], how="left")
    training = training[training["date"].dt.year >= start_year].copy()

    if "outcome" not in training.columns:
        training["goal_diff"] = training["home_score"] - training["away_score"]
        training["outcome"] = training["goal_diff"].apply(
            lambda x: "Home Win" if x > 0 else "Away Win" if x < 0 else "Draw"
        )

    training["neutral"] = training["neutral"].astype(int)
    training = _add_current_context_features(training, current_stats=current_stats)

    return training[
        FEATURE_COLUMNS + ["outcome", "home_team", "away_team", "date", "home_score", "away_score"]
    ].dropna()


def make_prediction_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral: bool = True,
    n_recent: int = 10,
    current_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    long_df = _team_long_table(df)
    latest_elos = latest_elo_ratings(df)

    def latest_form_stats(team: str) -> dict:
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

    home = latest_form_stats(home_team)
    away = latest_form_stats(away_team)

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
        "home_elo_before": float(latest_elos.get(home_team, 1500.0)),
        "away_elo_before": float(latest_elos.get(away_team, 1500.0)),
    }
    row["elo_diff"] = row["home_elo_before"] - row["away_elo_before"]

    feature_row = pd.DataFrame([{"home_team": home_team, "away_team": away_team, **row}])
    feature_row = _add_current_context_features(feature_row, current_stats=current_stats)
    return feature_row[FEATURE_COLUMNS]
