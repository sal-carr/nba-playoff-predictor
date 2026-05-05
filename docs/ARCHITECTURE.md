# Architecture

This document describes the architecture of the NBA Playoff Predictor.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NBA API                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Acquisition                              │
│                  src/data_acquisition.py                         │
│                                                                  │
│  • Fetches game data from NBA Stats API                         │
│  • Computes season records, streaks, rest days                  │
│  • Saves to data/processed/                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
│                   flows/train_model.py                          │
│                                                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │  start   │───▶│ feature_engineer│───▶│   tune_model     │   │
│  │          │    │                 │    │   (parallel x10) │   │
│  └──────────┘    └─────────────────┘    └────────┬─────────┘   │
│                                                   │             │
│                                          ┌───────▼────────┐    │
│                                          │  join_tuning   │    │
│                                          │  (select best) │    │
│                                          └───────┬────────┘    │
│                                                   │             │
│                                          ┌───────▼────────┐    │
│                                          │      end       │    │
│                                          └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Model stored in Metaflow artifacts
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Prediction Scripts                             │
│                                                                  │
│  scripts/predict_series.py      - Playoff series simulation     │
│  scripts/predict_next_games.py  - Individual game predictions   │
│                                                                  │
│  • Load model via Metaflow Client API                           │
│  • Fetch latest player/team data                                │
│  • Apply situational adjustments (rest, travel, momentum)       │
│  • Monte Carlo simulation for series outcomes                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Data Layer (`src/`)

| Module | Purpose |
|--------|---------|
| `data_acquisition.py` | Fetches data from NBA Stats API |
| `feature_engineering.py` | Computes ML features with anti-leakage safeguards |
| `io_utils.py` | Data loading, validation, and saving |
| `series_simulation.py` | Monte Carlo playoff series simulation |

### Training Flow (`flows/train_model.py`)

The training pipeline uses Metaflow for:
- **Checkpointing**: Each step saves state automatically
- **Parallelization**: `foreach` runs 10 model configs in parallel
- **Artifact storage**: Models persist across runs
- **Evaluation cards**: Visual reports tied to runs

### Features (42 total)

**Rolling Stats (5 and 10 game windows):**
- Points, rebounds, assists, turnovers
- Field goal %, 3-point %, steals, blocks

**Season Context:**
- Overall win percentage
- Home/road win percentage
- Current win/loss streak

**Differential Features:**
- Rest days advantage
- Rolling stats differential
- Season record differential

### Models Evaluated

| Model | Best AUC |
|-------|----------|
| Random Forest | 0.715 |
| XGBoost | 0.710 |
| Gradient Boosting | 0.705 |
| Logistic Regression | 0.680 |

## Data Flow

1. **Acquisition**: NBA API → Raw CSV
2. **Processing**: Raw CSV → Enhanced CSV (with computed fields)
3. **Training**: Enhanced CSV → Metaflow artifacts (model, features)
4. **Prediction**: Live API + Model → Predictions CSV

## Anti-Leakage Design

All features use `shift(1)` to ensure we only use information available *before* each game:

```python
# Correct: uses only past games
df[f"{col}_rolling_5"] = df.groupby("team")[col].shift(1).rolling(5).mean()

# Wrong: would leak future information
df[f"{col}_rolling_5"] = df.groupby("team")[col].rolling(5).mean()
```

## Deployment Options

### Local (Current)
- Metaflow runs locally with `.metaflow/` datastore
- Suitable for development and small-scale use

### Outerbounds/Cloud
- Deploy to Argo Workflows or AWS Step Functions
- Schedule retraining with `@schedule` decorator
- See `documents/outerbounds_handoff.md` for setup
