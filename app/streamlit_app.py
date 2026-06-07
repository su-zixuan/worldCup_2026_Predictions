from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import add_outcome, load_results
from src.features import build_training_table, make_prediction_features
from src.model import FEATURES, load_model, predict_match, train_model
from src.simulator import simulate_knockout_tournament

st.set_page_config(
    page_title="2026 World Cup Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def get_data(use_online: bool) -> pd.DataFrame:
    return add_outcome(load_results(use_online=use_online))

@st.cache_data(show_spinner=False)
def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return build_training_table(df, start_year=1994, n_recent=10)

@st.cache_resource(show_spinner=False)
def get_or_train_model(training_df: pd.DataFrame):
    model = load_model()
    metrics = None
    if model is None:
        model, metrics = train_model(training_df)
    return model, metrics

st.title("2026 FIFA World Cup Prediction Dashboard")
st.caption("Historical international football data + machine learning match outcome forecasting")

with st.sidebar:
    st.header("Settings")
    use_online = st.toggle("Use live GitHub dataset", value=True)
    st.write("Turn this off if you are offline; the app will use the included sample dataset.")

results = get_data(use_online)
teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
training_df = get_training_data(results)
model, metrics = get_or_train_model(training_df)

latest_date = results["date"].max().date()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches loaded", f"{len(results):,}")
col2.metric("Teams", f"{len(teams):,}")
col3.metric("Latest match date", str(latest_date))
col4.metric("Model rows", f"{len(training_df):,}")

if metrics:
    st.info(f"A new model was trained automatically. Holdout accuracy: {metrics['accuracy']:.2%}")

st.divider()

tab_predict, tab_team, tab_history, tab_simulator, tab_about = st.tabs([
    "Match Predictor", "Team Analytics", "Match History", "Tournament Simulator", "About"
])

with tab_predict:
    st.subheader("Predict a Match")
    c1, c2, c3 = st.columns([2, 2, 1])
    default_home = teams.index("United States") if "United States" in teams else 0
    default_away = teams.index("Brazil") if "Brazil" in teams else min(1, len(teams) - 1)
    home_team = c1.selectbox("Team A", teams, index=default_home)
    away_team = c2.selectbox("Team B", teams, index=default_away)
    neutral = c3.checkbox("Neutral venue", value=True)

    if home_team == away_team:
        st.warning("Choose two different teams.")
    else:
        feature_row = make_prediction_features(results, home_team, away_team, neutral=neutral)
        prediction = predict_match(model, feature_row)
        top = prediction.iloc[0]
        st.success(f"Most likely result: {top['Outcome']} ({top['Probability']:.1%})")

        fig = px.bar(
            prediction,
            x="Outcome",
            y="Probability",
            text=prediction["Probability"].map(lambda x: f"{x:.1%}"),
            title=f"Predicted Outcome: {home_team} vs {away_team}",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.write("Model input features")
        st.dataframe(feature_row[FEATURES], use_container_width=True)

with tab_team:
    st.subheader("Team Analytics")
    selected_team = st.selectbox("Select team", teams, key="team_analytics")
    team_matches = results[(results["home_team"] == selected_team) | (results["away_team"] == selected_team)].copy()

    def team_result(row):
        is_home = row["home_team"] == selected_team
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]
        if gf > ga:
            return "Win"
        if gf < ga:
            return "Loss"
        return "Draw"

    team_matches["team_result"] = team_matches.apply(team_result, axis=1)
    recent = team_matches.tail(20)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total matches", f"{len(team_matches):,}")
    t2.metric("Wins", int((team_matches["team_result"] == "Win").sum()))
    t3.metric("Draws", int((team_matches["team_result"] == "Draw").sum()))
    t4.metric("Losses", int((team_matches["team_result"] == "Loss").sum()))

    result_counts = team_matches["team_result"].value_counts().reset_index()
    result_counts.columns = ["Result", "Count"]
    st.plotly_chart(px.pie(result_counts, names="Result", values="Count", title=f"{selected_team} all-time results"), use_container_width=True)

    st.write("Most recent matches")
    st.dataframe(
        recent[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "team_result"]].sort_values("date", ascending=False),
        use_container_width=True,
    )

with tab_history:
    st.subheader("Explore Historical Matches")
    tournaments = ["All"] + sorted(results["tournament"].dropna().unique().tolist())
    selected_tournament = st.selectbox("Tournament", tournaments)
    year_range = st.slider(
        "Year range",
        int(results["date"].dt.year.min()),
        int(results["date"].dt.year.max()),
        (2000, int(results["date"].dt.year.max())),
    )
    filtered = results[
        (results["date"].dt.year >= year_range[0]) &
        (results["date"].dt.year <= year_range[1])
    ]
    if selected_tournament != "All":
        filtered = filtered[filtered["tournament"] == selected_tournament]

    st.metric("Filtered matches", f"{len(filtered):,}")
    goals_by_year = filtered.groupby(filtered["date"].dt.year)["total_goals"].mean().reset_index()
    goals_by_year.columns = ["Year", "Average Goals"]
    st.plotly_chart(px.line(goals_by_year, x="Year", y="Average Goals", title="Average goals per match by year"), use_container_width=True)
    st.dataframe(filtered.sort_values("date", ascending=False).head(500), use_container_width=True)


with tab_simulator:
    st.subheader("Tournament Simulator")
    st.write("Choose 4, 8, 16, or 32 teams. The app pairs teams in the order selected and simulates a knockout bracket.")
    default_sim_teams = [team for team in ["Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal", "Netherlands"] if team in teams]
    selected_sim_teams = st.multiselect(
        "Select teams",
        teams,
        default=default_sim_teams[:8] if len(default_sim_teams) >= 4 else teams[:4],
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    valid_sizes = {4, 8, 16, 32}
    if len(selected_sim_teams) not in valid_sizes:
        st.warning("Please select exactly 4, 8, 16, or 32 teams.")
    else:
        bracket = simulate_knockout_tournament(results, model, selected_sim_teams, seed=int(seed))
        champion = bracket.iloc[-1]["winner"]
        st.success(f"Simulated champion: {champion}")
        display = bracket.copy()
        display["team_a_probability"] = display["team_a_probability"].map(lambda x: f"{x:.1%}")
        display["team_b_probability"] = display["team_b_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True)
        wins_by_round = bracket.groupby(["round_name", "winner"]).size().reset_index(name="wins")
        st.plotly_chart(px.bar(wins_by_round, x="winner", y="wins", color="round_name", title="Teams advancing by round"), use_container_width=True)

with tab_about:
    st.subheader("About this project")
    st.write(
        "This dashboard predicts international football match outcomes using historical match results. "
        "It is designed as a semester-ready data science project: data loading, feature engineering, "
        "model training, prediction, and interactive visualization are separated into clean files."
    )
    st.write(
        "Important limitation: football is unpredictable. This model uses historical team form and scoring patterns, "
        "not player injuries, lineups, coaching changes, betting markets, or real-time FIFA rankings. "
        "Predictions should be treated as educational estimates, not guarantees."
    )
