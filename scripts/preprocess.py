from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pathlib import Path

from src.data_loader import add_outcome, load_current_team_stats, load_results
from src.features import build_training_table


def main() -> None:
    results = add_outcome(load_results(use_online=True))
    teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
    current_stats = load_current_team_stats(teams)
    training = build_training_table(results, current_stats=current_stats)
    out = Path(__file__).resolve().parents[1] / "data" / "processed_training.csv"
    training.to_csv(out, index=False)
    print(f"Saved {len(training):,} V2 training rows to {out}")


if __name__ == "__main__":
    main()
