from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.current_adjustments import apply_current_context_adjustment
from src.features import make_prediction_features
from src.model import predict_match


@dataclass
class SimulatedMatch:
    round_name: str
    team_a: str
    team_b: str
    winner: str
    team_a_probability: float
    team_b_probability: float


def _team_win_probability(predictions: pd.DataFrame) -> float:
    """Convert 3-way match probabilities into Team A advancement chance.

    Draw probability is split evenly because knockout games cannot end in a draw.
    """
    probs = dict(zip(predictions["Outcome"], predictions["Probability"]))
    return float(probs.get("Home Win", 0.0) + 0.5 * probs.get("Draw", 0.0))


def simulate_knockout_round(
    results: pd.DataFrame,
    model,
    teams: list[str],
    round_name: str,
    seed: int | None = None,
    current_stats: pd.DataFrame | None = None,
    use_current_adjustment: bool = True,
) -> tuple[list[str], list[SimulatedMatch]]:
    if len(teams) % 2 != 0:
        raise ValueError("Knockout simulation requires an even number of teams.")
    rng = random.Random(seed)
    winners: list[str] = []
    matches: list[SimulatedMatch] = []

    for i in range(0, len(teams), 2):
        team_a, team_b = teams[i], teams[i + 1]
        features = make_prediction_features(results, team_a, team_b, neutral=True, current_stats=current_stats)
        pred = predict_match(model, features)
        if use_current_adjustment:
            pred = apply_current_context_adjustment(pred, features)
        p_a = _team_win_probability(pred)
        winner = team_a if rng.random() < p_a else team_b
        winners.append(winner)
        matches.append(
            SimulatedMatch(
                round_name=round_name,
                team_a=team_a,
                team_b=team_b,
                winner=winner,
                team_a_probability=p_a,
                team_b_probability=1 - p_a,
            )
        )
    return winners, matches


def simulate_knockout_tournament(
    results: pd.DataFrame,
    model,
    teams: Iterable[str],
    seed: int | None = None,
    current_stats: pd.DataFrame | None = None,
    use_current_adjustment: bool = True,
) -> pd.DataFrame:
    current = list(teams)
    if len(current) < 2 or len(current) & (len(current) - 1) != 0:
        raise ValueError("Number of teams must be a power of 2, such as 4, 8, 16, or 32.")

    round_names = {
        32: "Round of 32",
        16: "Round of 16",
        8: "Quarterfinal",
        4: "Semifinal",
        2: "Final",
    }
    all_matches: list[SimulatedMatch] = []
    round_seed = seed
    while len(current) > 1:
        round_name = round_names.get(len(current), f"Round of {len(current)}")
        current, matches = simulate_knockout_round(
            results,
            model,
            current,
            round_name,
            seed=round_seed,
            current_stats=current_stats,
            use_current_adjustment=use_current_adjustment,
        )
        all_matches.extend(matches)
        if round_seed is not None:
            round_seed += 1

    return pd.DataFrame([m.__dict__ for m in all_matches])
