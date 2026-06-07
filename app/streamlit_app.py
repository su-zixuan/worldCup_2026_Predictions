from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.current_adjustments import apply_current_context_adjustment, current_context_score
from src.data_loader import add_outcome, load_current_team_stats, load_results
from src.features import build_training_table, make_prediction_features
from src.model import FEATURES, load_model, predict_match, train_model
from src.simulator import simulate_knockout_tournament

st.set_page_config(
    page_title="2026 World Cup Prediction Dashboard V2",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data(use_online: bool) -> pd.DataFrame:
    return add_outcome(load_results(use_online=use_online))


@st.cache_data(show_spinner=False)
def get_current_stats(teams: tuple[str, ...]) -> pd.DataFrame:
    return load_current_team_stats(list(teams))


@st.cache_data(show_spinner=False)
def get_training_data(df: pd.DataFrame, current_stats: pd.DataFrame) -> pd.DataFrame:
    return build_training_table(df, start_year=1994, n_recent=10, current_stats=current_stats)


@st.cache_resource(show_spinner=False)
def get_or_train_model(training_df: pd.DataFrame):
    model = load_model()
    metrics = None
    expected_features = set(FEATURES)
    model_features = set(getattr(model, "feature_names_in_", [])) if model is not None else set()
    if model is None or not expected_features.issubset(model_features):
        model, metrics = train_model(training_df)
    return model, metrics


st.title("2026 FIFA World Cup Prediction Dashboard V2")
st.caption("Historical results + rolling Elo + editable current team context for injuries, rankings, coaches, and squad stability")

with st.sidebar:
    st.header("Settings")
    use_online = st.toggle("Use live GitHub match-results dataset", value=True)
    use_current_adjustment = st.toggle("Use current-team V2 adjustment", value=True)
    st.write("Turn online data off if you are offline; the app will use the included sample dataset.")

results = get_data(use_online)
teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
current_stats = get_current_stats(tuple(teams))
training_df = get_training_data(results, current_stats)
model, metrics = get_or_train_model(training_df)

latest_date = results["date"].max().date()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Matches loaded", f"{len(results):,}")
col2.metric("Teams", f"{len(teams):,}")
col3.metric("Latest match date", str(latest_date))
col4.metric("Model rows", f"{len(training_df):,}")
col5.metric("Features", f"{len(FEATURES)}")

if metrics:
    st.info(f"A new V2 model was trained automatically. Holdout accuracy: {metrics['accuracy']:.2%}")

st.divider()

tab_predict, tab_team, tab_current, tab_history, tab_simulator, tab_about = st.tabs([
    "Match Predictor", "Team Analytics", "Current Team Context", "Match History", "Tournament Simulator", "About"
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
        feature_row = make_prediction_features(results, home_team, away_team, neutral=neutral, current_stats=current_stats)
        base_prediction = predict_match(model, feature_row)
        prediction = apply_current_context_adjustment(base_prediction, feature_row) if use_current_adjustment else base_prediction
        top = prediction.iloc[0]
        st.success(f"Most likely result: {top['Outcome']} ({top['Probability']:.1%})")

        chart_data = prediction[["Outcome", "Probability"]]
        fig = px.bar(
            chart_data,
            x="Outcome",
            y="Probability",
            text=chart_data["Probability"].map(lambda x: f"{x:.1%}"),
            title=f"Predicted Outcome: {home_team} vs {away_team}",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.write("Base model probabilities")
            st.dataframe(
                base_prediction.assign(Probability=base_prediction["Probability"].map(lambda x: f"{x:.1%}")),
                use_container_width=True,
            )
        with b:
            st.write("Current-context comparison")
            context_score = current_context_score(feature_row)
            st.metric("Team A context advantage", f"{context_score:+.2f}")
            comparison_cols = [
                "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
                "home_current_elo_rating", "away_current_elo_rating", "current_elo_diff",
                "home_coach_tenure_years", "away_coach_tenure_years", "coach_tenure_diff",
                "home_injured_players", "away_injured_players", "injury_diff",
                "home_roster_stability", "away_roster_stability", "roster_stability_diff",
                "home_squad_market_value_m", "away_squad_market_value_m", "squad_market_value_diff",
            ]
            st.dataframe(feature_row[comparison_cols], use_container_width=True)

        with st.expander("Show all model input features"):
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
    team_context = current_stats[current_stats["team"] == selected_team].head(1)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total matches", f"{len(team_matches):,}")
    t2.metric("Wins", int((team_matches["team_result"] == "Win").sum()))
    t3.metric("Draws", int((team_matches["team_result"] == "Draw").sum()))
    t4.metric("Losses", int((team_matches["team_result"] == "Loss").sum()))

    if not team_context.empty:
        st.write("Current team context")
        st.dataframe(team_context, use_container_width=True)

    result_counts = team_matches["team_result"].value_counts().reset_index()
    result_counts.columns = ["Result", "Count"]
    st.plotly_chart(px.pie(result_counts, names="Result", values="Count", title=f"{selected_team} all-time results"), use_container_width=True)

    st.write("Most recent matches")
    st.dataframe(
        recent[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "team_result"]].sort_values("date", ascending=False),
        use_container_width=True,
    )

with tab_current:
    st.subheader("Current Team Context")
    st.write(
        "These V2 features are stored in `data/current_team_stats.csv`. Update this CSV as rankings, injuries, coaches, and squads change. "
        "Teams missing from the CSV receive neutral default values so the dashboard still works."
    )
    search = st.text_input("Filter teams", "")
    display_stats = current_stats.copy()
    if search:
        display_stats = display_stats[display_stats["team"].str.contains(search, case=False, na=False)]
    st.dataframe(display_stats, use_container_width=True)
    st.download_button(
        "Download current_team_stats.csv",
        data=current_stats.to_csv(index=False),
        file_name="current_team_stats.csv",
        mime="text/csv",
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
        bracket = simulate_knockout_tournament(
            results,
            model,
            selected_sim_teams,
            seed=int(seed),
            current_stats=current_stats,
            use_current_adjustment=use_current_adjustment,
        )
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
        "V2 keeps the original historical-results model, then expands it with rolling Elo ratings and a current-context layer. "
        "The current-context layer makes the dashboard more realistic because national teams change over time: players get injured, coaches change, squads age, and rosters become more or less stable."
    )
    st.write(
        "Important limitation: the current-team CSV is manually maintained sample data. For a real production model, these values should be updated from reliable sports-data sources before each prediction. "
        "Predictions should be treated as educational estimates, not guarantees."
    )
