# NBA Playoff Predictor

A machine learning pipeline for predicting NBA playoff outcomes using [Metaflow](https://metaflow.org/).

Built as part of the [Metaflow + Outerbounds NBA Tutorial](docs/TUTORIAL.md), this project demonstrates how to build production-ready ML pipelines for sports prediction.

## Features

- **End-to-end ML pipeline** with Metaflow for training and inference
- **42 engineered features** including rolling averages, season records, and momentum indicators
- **Multiple model comparison** (XGBoost, Random Forest, Gradient Boosting, Logistic Regression)
- **Playoff series simulation** using Monte Carlo methods
- **Next-game predictions** with player performance and situational factors
- **Anti-leakage design** ensuring only past data is used for predictions

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch NBA data

```bash
python src/data_acquisition.py
```

### 3. Train the model

```bash
python flows/train_model.py run
```

### 4. Generate predictions

```bash
# Playoff series predictions
python scripts/predict_series.py

# Individual game predictions
python scripts/predict_next_games.py
```

## Project Structure

```
├── src/                    # Core library
│   ├── data_acquisition.py # Fetch data from NBA API
│   ├── feature_engineering.py # Build ML features
│   ├── advanced_features.py # Schedule/momentum features
│   ├── ensemble.py         # Model ensembling
│   ├── injury_data.py      # ESPN injury API
│   ├── betting_lines.py    # Betting lines integration
│   ├── io_utils.py         # Data I/O utilities
│   └── series_simulation.py # Monte Carlo simulation
│
├── flows/                  # Metaflow pipelines
│   ├── train_model.py      # Model training flow
│   └── predict_nba_model.py # Prediction flow
│
├── scripts/                # CLI utilities
│   ├── predict_series.py   # Playoff series predictions
│   └── predict_next_games.py # Single game predictions
│
├── examples/               # Usage examples
│   ├── quickstart.py
│   └── custom_prediction.py
│
├── data/                   # Data directory
│   └── sample/             # Sample data (100 games)
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System design
│   ├── API.md              # API reference
│   ├── MODEL_IMPROVEMENTS.md # Experiment results
│   └── TUTORIAL.md         # Original tutorial
│
└── results/                # Prediction outputs
```

## How It Works

### Training Pipeline

```
Load Data → Feature Engineering → Parallel Model Tuning → Select Best → Save
```

The training flow (`flows/train_model.py`):
1. Loads NBA game data with box scores
2. Engineers 42 features (rolling averages, season records, differentials)
3. Runs 10 model configurations in parallel using Metaflow's `foreach`
4. Selects the best model by AUC
5. Generates an evaluation card with metrics and visualizations

### Prediction

Two prediction modes:

**Series Predictions** (`scripts/predict_series.py`):
- Loads the trained model
- Calculates each team's home win probability
- Runs Monte Carlo simulation (100k iterations)
- Outputs series win probabilities and championship predictions

**Next-Game Predictions** (`scripts/predict_next_games.py`):
- Incorporates player performance trends
- Adjusts for rest days and travel fatigue
- Considers star player momentum (hot/cold streaks)
- Outputs game-by-game predictions with confidence

## Model Performance

| Metric | Value |
|--------|-------|
| AUC | 0.715 |
| Accuracy | 65.2% |
| Best Model | Random Forest (200 trees, depth 10) |

### Improvement Experiments

We tested several approaches to improve the baseline:

| Approach | Result |
|----------|--------|
| Ensemble (RF + XGB + GBM) | -1.5% AUC |
| Advanced features (+6) | -0.2% AUC |
| Calibration (isotonic) | -2.9% AUC |
| Feature selection (top 25) | -0.4% AUC |

**Conclusion**: The baseline model is well-optimized for available data. Further gains require more training data, player-level injury data, or real betting lines. See [Model Improvements](docs/MODEL_IMPROVEMENTS.md) for details.

## Example Output

```
================================================================================
NEXT GAME PREDICTIONS - MAY 5, 2026
================================================================================

Western Semis G1: LAL @ OKC
  OKC home win prob: 82.4%
  LAL travel fatigue: 0.50
  
  Key players:
    Shai Gilgeous-Alexander: 36.7 ppg (64.7% FG) 🔥
    LeBron James: 21.0 ppg (35.7% FG) ➖
  
  → PICK: OKC (85.4%)
```

## Make Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make data           # Fetch NBA data
make train          # Train model
make predict-series # Run series predictions
make predict-games  # Run game predictions
make card           # View evaluation card
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [API Reference](docs/API.md) - Module and function documentation
- [Model Improvements](docs/MODEL_IMPROVEMENTS.md) - Experiment results and findings
- [Tutorial](docs/TUTORIAL.md) - Original Metaflow tutorial
- [Learnings](docs/learnings/) - Project retrospective

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- Built with [Metaflow](https://metaflow.org/) by Outerbounds
- Data from [NBA Stats API](https://github.com/swar/nba_api)
- Inspired by the Metaflow NBA prediction tutorial
