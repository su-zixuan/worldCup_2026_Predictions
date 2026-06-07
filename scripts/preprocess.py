from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_loader import add_outcome, load_results
from src.features import build_training_table

OUTPUT = Path("data/processed_training.csv")


def main() -> None:
    results = add_outcome(load_results(use_online=True))
    training = build_training_table(results, start_year=1994, n_recent=10)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT, index=False)
    print(f"Saved {len(training):,} training rows to {OUTPUT}")


if __name__ == "__main__":
    main()
