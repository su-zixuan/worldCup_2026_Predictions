# 2026 FIFA World Cup Prediction Dashboard

This is a GitHub-ready data science project for forecasting international football match results and exploring historical football data. The dashboard uses historical men's international match results, creates recent-form features for each team, trains a machine-learning classifier, and displays prediction probabilities in an interactive Streamlit app.

## Main Features

- Match predictor for any two national teams in the dataset.
- Win/draw/loss probability chart.
- Team analytics page with wins, draws, losses, and recent matches.
- Historical match explorer with tournament and year filters.
- Knockout tournament simulator for 4, 8, 16, or 32 teams.
- Separate scripts for preprocessing, training, and command-line prediction.
- Documentation for local setup, GitHub, and Streamlit deployment.

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
│   └── sample_results.csv
├── docs/
│   ├── deployment.md
│   └── project_checklist.md
├── models/
├── scripts/
│   ├── predict.py
│   └── preprocess.py
├── src/
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

### 1. Clone your repository

```bash
git clone <your-repository-url>
cd worldcup_2026_dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This saves a trained model to:

```text
models/match_outcome_model.joblib
```

### 5. Run the dashboard

```bash
streamlit run app/streamlit_app.py
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

## How the Model Works

The project turns each match into a training example. For every historical match, it calculates both teams' recent form before that match happened. This avoids using future data by accident.

The model uses features such as:

- Home team recent win rate
- Away team recent win rate
- Home team recent draw rate
- Away team recent draw rate
- Average goals scored
- Average goals conceded
- Number of recent matches available
- Neutral venue flag

The target label is one of three outcomes:

- Home Win
- Draw
- Away Win

A Random Forest classifier learns patterns from these historical examples and outputs probability estimates for future matchups.

## Dashboard Pages

### Match Predictor

Choose two teams and receive predicted probabilities for each outcome.

### Team Analytics

Select a team and view its historical match count, wins, draws, losses, result distribution, and recent results.

### Match History

Filter historical matches by tournament and year range. The page also shows average goals per match over time.

### Tournament Simulator

Select 4, 8, 16, or 32 teams. The app predicts each knockout match and simulates which team advances until a champion is selected.

## Deployment

See `docs/deployment.md` for Streamlit Community Cloud deployment steps.

## Limitations

This is an educational forecasting project, not a professional betting model. It does not include player injuries, current rosters, coaching changes, FIFA rankings, Elo ratings, betting odds, travel distance, rest days, or live lineup news. Predictions should be treated as data-driven estimates, not guarantees.

## Future Improvements

- Add FIFA ranking or Elo rating data.
- Add official 2026 World Cup fixtures when finalized.
- Add group-stage simulation.
- Compare multiple ML models.
- Add expected-goals prediction.
- Add a cleaner visual bracket component.

## Author

Zixuan Su
