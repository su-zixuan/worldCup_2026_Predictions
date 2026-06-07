from __future__ import annotations

import pandas as pd


def _team_stats_before_date(df: pd.DataFrame, team: str, match_date: pd.Timestamp, n_recent: int = 10) -> dict:
    past = df[df["date"] < match_date]
    games = past[(past["home_team"] == team) | (past["away_team"] == team)].tail(n_recent)

    if games.empty:
        return {
            "win_rate": 0.33,
            "draw_rate": 0.34,
            "avg_goals_for": 1.0,
            "avg_goals_against": 1.0,
            "recent_matches": 0,
        }

    wins = draws = goals_for = goals_against = 0
    for _, row in games.iterrows():
        is_home = row["home_team"] == team
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]
        goals_for += gf
        goals_against += ga
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1

    total = len(games)
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "avg_goals_for": goals_for / total,
        "avg_goals_against": goals_against / total,
        "recent_matches": total,
    }


def build_training_table(df: pd.DataFrame, start_year: int = 1994, n_recent: int = 10) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    rows = []
    usable = df[df["date"].dt.year >= start_year]

    for _, row in usable.iterrows():
        home = _team_stats_before_date(df, row["home_team"], row["date"], n_recent)
        away = _team_stats_before_date(df, row["away_team"], row["date"], n_recent)
        rows.append({
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "tournament": row["tournament"],
            "neutral": int(row["neutral"]),
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
            "outcome": "Home Win" if row["home_score"] > row["away_score"] else "Away Win" if row["home_score"] < row["away_score"] else "Draw",
        })
    return pd.DataFrame(rows)


def make_prediction_features(df: pd.DataFrame, home_team: str, away_team: str, neutral: bool = True, n_recent: int = 10) -> pd.DataFrame:
    match_date = pd.Timestamp.today()
    home = _team_stats_before_date(df, home_team, match_date, n_recent)
    away = _team_stats_before_date(df, away_team, match_date, n_recent)
    return pd.DataFrame([{
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
    }])
