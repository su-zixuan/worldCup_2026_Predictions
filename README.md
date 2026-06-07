# 2026 FIFA World Cup Prediction Dashboard V2

This is a GitHub-ready data science project for forecasting international football match results and exploring historical football data. The dashboard uses historical men's international match results, rolling team-form features, rolling Elo ratings, and an editable current-team context file for changing factors such as FIFA rank, coach tenure, squad age, injuries, roster stability, and squad market value.

## Main Features

- Match predictor for any two national teams in the dataset.
- Win/draw/loss probability chart.
- Rolling Elo features calculated from historical match results.
- V2 current-team adjustment for rankings, coaches, injuries, squad stability, squad age, squad value, and World Cup experience.
- Team analytics page with wins, draws, losses, current context, and recent matches.
- Current Team Context page for viewing and downloading `data/current_team_stats.csv`.
- Historical match explorer with tournament and year filters.
- Knockout tournament simulator for 4, 8, 16, or 32 teams.
- Separate scripts for preprocessing, training, and command-line prediction.
- Tests and documentation for local setup and deployment.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib
- Pytest

## Project Structure

```text
worldcup_2026_dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── current_team_stats.csv
│   └── sample_results.csv
├── docs/
│   ├── deployment.md
│   ├── project_checklist.md
│   └── v2_current_context.md
├── models/
├── scripts/
│   ├── predict.py
│   └── preprocess.py
├── src/
│   ├── current_adjustments.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── simulator.py
├── tests/
│   └── test_features.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the V2 model

```bash
python train_model.py
```

This saves a trained model to:

```text
models/match_outcome_model.joblib
```

### 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

If your Mac does not recognize `python` or `streamlit`, use:

```bash
.venv/bin/python train_model.py
.venv/bin/python -m streamlit run app/streamlit_app.py
```

## Optional Scripts

Preprocess data into a reusable training CSV:

```bash
python scripts/preprocess.py
```

Predict one match from the command line:

```bash
python scripts/predict.py Argentina France --neutral
```

Run tests:

```bash
pytest
```

or:

```bash
.venv/bin/python -m pytest
```

## How the Model Works

The project turns each match into a training example. For every historical match, it calculates both teams' recent form before that match happened. This avoids using future match results by accident.

The model uses features such as:

- Home/team A recent win rate
- Away/team B recent win rate
- Recent draw rate
- Average goals scored
- Average goals conceded
- Number of recent matches available
- Neutral venue flag
- Rolling Elo rating before the match
- Elo rating difference
- FIFA rank context
- Current Elo context
- Coach tenure
- Average squad age
- Injured players
- Roster stability
- Squad market value
- World Cup experience

The target label is one of three outcomes:

- Home Win
- Draw
- Away Win

A Random Forest classifier learns patterns from these historical examples and outputs probability estimates for future matchups. V2 can then apply a small current-context adjustment so the dashboard responds to changing team conditions.

## Updating Current Team Information

Edit this file:

```text
data/current_team_stats.csv
```

Update values such as FIFA rank, current Elo rating, coach tenure, squad age, injuries, roster stability, squad value, and World Cup experience. The dashboard will load the file automatically. Teams missing from the file receive neutral default values.

See:

```text
docs/v2_current_context.md
```

## Limitations

This is an educational forecasting project, not a professional betting model. The current-team CSV is sample/manual data and should be updated from reliable sources before serious use. Football is unpredictable, and predictions should be treated as data-driven estimates rather than guarantees.

## Future Improvements

- Connect an official or third-party sports API for live rankings, injuries, and squad lists.
- Add official 2026 World Cup fixtures when finalized.
- Add group-stage simulation.
- Compare multiple ML models.
- Add expected-goals prediction.
- Add a cleaner visual bracket component.

## Author

Zixuan Su
