from src.data_loader import add_outcome, load_current_team_stats, load_results
from src.features import build_training_table
from src.model import train_model


def main() -> None:
    print("Loading match data...")
    results = add_outcome(load_results(use_online=True))
    teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
    current_stats = load_current_team_stats(teams)
    print(f"Loaded {len(results):,} matches")
    print(f"Loaded current context for {len(current_stats):,} teams")

    print("Building V2 features...")
    training = build_training_table(results, start_year=1994, n_recent=10, current_stats=current_stats)
    print(f"Training rows: {len(training):,}")

    print("Training model...")
    _, metrics = train_model(training)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Features used: {metrics['feature_count']}")
    print("Saved model to models/match_outcome_model.joblib")


if __name__ == "__main__":
    main()
