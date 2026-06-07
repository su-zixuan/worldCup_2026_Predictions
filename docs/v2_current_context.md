# V2 Current-Team Context

The original dashboard trained on historical international match results only. That is useful, but national teams are not stable over time. A country can have a new coach, different players, a younger or older squad, important injuries, or a much more stable roster than it had in the past.

V2 adds two improvements:

1. **Rolling Elo features** calculated from historical results without looking into the future.
2. **Editable current-team context** stored in `data/current_team_stats.csv`.

## Current-team CSV columns

`data/current_team_stats.csv` contains these columns:

- `team`: national team name matching the historical results dataset.
- `fifa_rank`: current FIFA ranking. Lower is better.
- `current_elo_rating`: current team strength estimate. Higher is better.
- `coach_tenure_years`: how long the current coach has managed the team.
- `avg_squad_age`: average age of the expected squad.
- `injured_players`: number of important unavailable or injured players.
- `roster_stability`: value from 0 to 1 estimating how stable the squad/lineup is.
- `squad_market_value_m`: approximate squad market value in millions.
- `world_cup_experience`: approximate number of World Cup appearances or experience score.

These values are intentionally editable because they change frequently. Before presenting or deploying the project, update this CSV with the latest information you want to use.

## How V2 uses the current context

The Random Forest model still learns from historical patterns. The current-context layer then makes a modest probability adjustment after the model predicts. This is safer than pretending that today’s injuries or coach tenure existed in older historical matches.

For example, if Team A has better ranking, higher current Elo, fewer injuries, more roster stability, and more experience, the V2 adjustment can slightly increase Team A’s win probability. If Team B has the advantage, it can slightly increase Team B’s probability.

## Important limitation

The sample CSV is not official live data. It is a project-ready structure showing how to include changing team information. For a serious prediction model, update the CSV from reliable sports-data sources before each use.
