# API Reference

## Core Modules

### `src/data_acquisition.py`

#### `fetch_nba_games(seasons, output_path)`
Fetch NBA game data from the NBA Stats API.

**Parameters:**
- `seasons` (list[str]): List of seasons to fetch (e.g., `['2023-24', '2024-25']`)
- `output_path` (str): Path to save the output CSV

**Returns:** `pd.DataFrame` with columns:
- `game_id`, `game_date`, `season`
- `home_team`, `away_team`
- `home_points`, `away_points`
- `home_rebounds`, `away_rebounds`, etc.
- `home_rest_days`, `away_rest_days`
- `home_season_win_pct`, `away_season_win_pct`
- `home_streak`, `away_streak`

---

### `src/feature_engineering.py`

#### `build_enhanced_features(df)`
Build ML-ready features from game data.

**Parameters:**
- `df` (pd.DataFrame): Raw game data

**Returns:** `pd.DataFrame` with added feature columns:
- Rolling averages (5 and 10 game): `home_points_rolling_5`, `away_points_rolling_10`, etc.
- Lag features: `home_points_lag_1`, etc.
- Differential features: `rolling_points_diff`, `rest_diff`, etc.
- Target variables: `home_win`, `home_cover`, `total_over`

#### `get_enhanced_feature_columns(df)`
Get list of available feature columns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with computed features

**Returns:** `list[str]` of feature column names

---

### `src/io_utils.py`

#### `load_game_data(path)`
Load game data from CSV with proper date parsing.

**Parameters:**
- `path` (str): Path to CSV file

**Returns:** `pd.DataFrame` with `game_date` as datetime

#### `save_predictions(df, path)`
Save predictions to CSV.

**Parameters:**
- `df` (pd.DataFrame): Predictions dataframe
- `path` (str): Output path

---

### `src/series_simulation.py`

#### `simulate_series(home_prob_higher, home_prob_lower, n_sims=100000)`
Monte Carlo simulation of a playoff series.

**Parameters:**
- `home_prob_higher` (float): Home win probability for higher seed
- `home_prob_lower` (float): Home win probability for lower seed
- `n_sims` (int): Number of simulations

**Returns:** `dict` with:
- `higher_seed_prob`: Probability higher seed wins series
- `lower_seed_prob`: Probability lower seed wins series
- `expected_games`: Expected number of games
- `outcomes`: Distribution of outcomes (e.g., "Higher in 5": 0.23)

---

## Metaflow Flows

### `TrainNBAModelFlow`

Training pipeline for NBA prediction model.

**Parameters:**
- `--sport`: Sport to filter (default: "nba")
- `--target`: Prediction target (default: "home_win")
- `--start-date`: Start date for training data
- `--end-date`: End date for training data
- `--input-path`: Path to input data

**Usage:**
```bash
python flows/train_model.py run
python flows/train_model.py run --end-date 2026-12-31
```

**Artifacts:**
- `best_model`: Trained sklearn model
- `feature_columns`: List of feature names
- `best_auc`: Validation AUC score

### `PredictNBAModelFlow`

Prediction pipeline using trained model.

**Parameters:**
- `--input-path`: Path to upcoming games CSV
- `--historical-path`: Path to historical games CSV
- `--output-path`: Where to save predictions
- `--model-tag`: Optional tag to select specific training run

**Usage:**
```bash
python flows/predict_nba_model.py run
python flows/predict_nba_model.py run --input-path data/my_games.csv
```

---

## CLI Scripts

### `scripts/predict_series.py`

Predict playoff series outcomes.

**Usage:**
```bash
python scripts/predict_series.py
```

**Output:** `results/playoff_predictions_current.csv`

### `scripts/predict_next_games.py`

Predict individual upcoming games with situational analysis.

**Usage:**
```bash
python scripts/predict_next_games.py
```

**Output:** `results/next_game_predictions.csv`

**Factors considered:**
- Model baseline probability
- Rest days differential
- Travel fatigue
- Recent team form
- Star player momentum (hot/cold)
