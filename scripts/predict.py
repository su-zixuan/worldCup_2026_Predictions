from __future__ import annotations

import argparse

from src.data_loader import add_outcome, load_results
from src.features import make_prediction_features
from src.model import load_model, predict_match, train_model
from src.features import build_training_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one international football match.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true", help="Use neutral venue flag")
    args = parser.parse_args()

    results = add_outcome(load_results(use_online=True))
    model = load_model()
    if model is None:
        model, _ = train_model(build_training_table(results))

    features = make_prediction_features(results, args.home_team, args.away_team, neutral=args.neutral)
    prediction = predict_match(model, features)
    print(prediction.to_string(index=False, formatters={"Probability": "{:.1%}".format}))


if __name__ == "__main__":
    main()
