from src.data_loader import add_outcome, load_results
from src.features import build_training_table
from src.model import train_model


def main() -> None:
    print("Loading match data...")
    results = add_outcome(load_results(use_online=True))
    print(f"Loaded {len(results):,} matches")

    print("Building features...")
    training = build_training_table(results, start_year=1994, n_recent=10)
    print(f"Training rows: {len(training):,}")

    print("Training model...")
    _, metrics = train_model(training)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Saved model to models/match_outcome_model.joblib")


if __name__ == "__main__":
    main()
