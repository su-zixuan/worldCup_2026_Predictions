from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse

from src.current_adjustments import apply_current_context_adjustment
from src.data_loader import add_outcome, load_current_team_stats, load_results
from src.features import build_training_table, make_prediction_features
from src.model import load_model, predict_match, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one international football match.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true", help="Use neutral venue flag")
    parser.add_argument("--no-current-adjustment", action="store_true", help="Disable V2 current-team adjustment")
    args = parser.parse_args()

    results = add_outcome(load_results(use_online=True))
    teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
    current_stats = load_current_team_stats(teams)

    model = load_model()
    if model is None:
        model, _ = train_model(build_training_table(results, current_stats=current_stats))

    features = make_prediction_features(
        results,
        args.home_team,
        args.away_team,
        neutral=args.neutral,
        current_stats=current_stats,
    )
    prediction = predict_match(model, features)
    if not args.no_current_adjustment:
        prediction = apply_current_context_adjustment(prediction, features)
    print(prediction[["Outcome", "Probability"]].to_string(index=False, formatters={"Probability": "{:.1%}".format}))


if __name__ == "__main__":
    main()
