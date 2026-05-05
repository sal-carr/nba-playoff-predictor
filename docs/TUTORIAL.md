# Building a Production-Grade Sports Analytics ML Pipeline with Metaflow and Outerbounds

## Introduction

Metaflow is an open-source Python framework that was originally built at Netflix to help data scientists and ML engineers move from exploratory work to production workflows without rewriting everything for a different platform. The core idea is simple: keep workflow authoring in plain Python, but make execution, versioning, orchestration, and artifact tracking first-class so you are not stuck juggling notebooks, ad hoc scripts, cron jobs, and mystery folders full of model files.

Why does that matter for sports analytics? Because a serious sports pipeline is not one script. It is a system that loads fresh data, engineers features without leakage, trains and compares multiple models, stores every run, publishes evaluation artifacts, and can be re-run on a schedule or triggered after new games land. Plain Python scripts can do each piece, but they do not give you workflow structure, reproducibility, built-in lineage, or easy local-to-cloud movement. Metaflow was designed to solve exactly that class of problem.

Outerbounds extends that open-source foundation with a managed platform, including a production orchestrator, UI, and deployment experience built on Metaflow.

In this guide, you will build an **NBA playoff game prediction pipeline** that is realistic enough to matter and structured enough to teach you Metaflow properly. The pipeline will:

- ingest historical NBA game data
- engineer rolling and lagged team features
- fan out hyperparameter tuning in parallel
- train an XGBoost model to predict a target such as `home_win`
- generate a visual evaluation report with a Metaflow Card
- version the trained model so you can retrieve it later
- create a separate prediction flow that loads the latest successful model
- run locally first, then scale to AWS Batch or Kubernetes
- optionally deploy and schedule on Argo Workflows via Outerbounds or on AWS Step Functions in self-hosted Metaflow setups

The *business* context will stay anchored to the modeling approach defined earlier:

- reusable across sports and targets
- game-level predictions first
- later usable for recursive series simulation
- versioned and retrievable
- schedulable for nightly or weekly retraining
- local-first, cloud-second

That means you are not building a toy Iris classifier. You are building the **first durable layer** of an NBA playoff forecasting engine.

---

## Prerequisites

In this section, you will prepare your machine, install Metaflow, and create a project structure that will still make sense once the tutorial grows into a real codebase.

### Required knowledge, tools, and accounts

You should be comfortable with:

- basic Python
- virtual environments
- pandas DataFrames
- train/test splits and simple ML evaluation
- reading terminal output

You will also need:

- Python 3.10 or newer
- `pip` or `uv`
- a terminal
- a code editor
- optionally, an Outerbounds account or a self-hosted Metaflow stack if you want scheduling and cloud orchestration beyond local runs

Metaflow can be installed and used locally with `pip install metaflow`. The docs also recommend a fuller dev stack when you want UI, cloud compute, and deployment features, but local runs work with the basic package alone.

### Install Metaflow locally

Create a working directory and a virtual environment:

```bash
mkdir sports-metaflow
cd sports-metaflow

python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

pip install --upgrade pip
pip install metaflow pandas numpy scikit-learn xgboost matplotlib pyarrow
```

Expected terminal output will look roughly like this:

```text
Collecting metaflow
Collecting pandas
Collecting xgboost
...
Successfully installed metaflow-<version> pandas-<version> xgboost-<version> ...
```

### What Just Happened?

You created an isolated Python environment and installed Metaflow plus the libraries you will use for data loading, feature engineering, training, and plotting. Metaflow itself is small compared to a full ML stack; most of the heavy lifting still comes from standard Python libraries. Metaflow’s job is to orchestrate and version the workflow around them.

Verify the setup:

```bash
python -c "import metaflow; print(metaflow.__version__)"
```

Expected output:

```text
2.x.y
```

### Configure Outerbounds or stay local

You have two valid modes for this tutorial:

**Mode A — Local-only**  
Use Metaflow locally, run the flow from your laptop, inspect results with the Client API and Cards, and skip deployment-specific commands until later.

**Mode B — Outerbounds / cloud-backed**  
Use the same code, but connect it to a managed orchestration environment so you can deploy to Argo Workflows, schedule retraining, and monitor runs in the Outerbounds UI.

For your first pass, local-only mode is the best choice. It mirrors how Metaflow itself is intended to be used: develop locally, then move the same flow to cloud compute or a production orchestrator later.

### Project folder structure

Create this layout:

```text
sports-metaflow/
├── data/
│   ├── raw/
│   │   └── nba_games.csv
│   └── prediction/
│       └── upcoming_games.csv
├── flows/
│   ├── train_nba_model.py
│   └── predict_nba_model.py
├── src/
│   ├── io_utils.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── reports/
├── notebooks/
├── tests/
└── requirements.txt
```

Why this structure?

- `flows/` contains the actual Metaflow `FlowSpec` entry points
- `src/` contains reusable Python functions so your steps stay readable
- `data/raw/` holds local development data
- `data/prediction/` is a convenient place for upcoming games or inference input
- `reports/` can store exported charts or offline summaries
- `notebooks/` is optional, but useful for exploring Metaflow results through the Client API
- `tests/` is where your regular Python tests should live

**Beginner mistake:** putting all feature logic inside one giant Metaflow step.  
That makes the flow hard to read, hard to test, and painful to debug. Keep orchestration in the flow and transformation logic in normal Python modules.

---

## Step 1 — Understanding Metaflow Flows and Steps

In this step, you will learn the mental model that makes Metaflow click.

### What a `FlowSpec` is

A `FlowSpec` is the class you define to describe a workflow. Metaflow executes that workflow as a directed graph of operations, where each node is a `@step` and each edge is a transition declared with `self.next(...)`. Metaflow follows the dataflow paradigm: steps are the units of execution, and the graph defines what can run after what.

### What a `@step` does

A `@step` marks a method as a workflow node. Each step:

- runs as one task
- either succeeds or fails as a whole
- persists all instance variables assigned on `self` when it completes successfully
- transitions to one or more next steps using `self.next(...)`

That “persist on success” rule is one of the most important ideas in Metaflow. It means steps act like checkpoints. If a downstream step fails, you do not need to rerun all the upstream work unless you want to.

### How artifacts work

If you assign a value to `self.x` in a step, Metaflow treats it as an artifact and persists it automatically when the step finishes. Later steps can read `self.x` as if it were a normal instance attribute, but under the hood Metaflow has serialized and stored it for you. This is the mechanism that gives you run versioning, reproducibility, and the ability to inspect past runs from the Client API.

### Hello World flow

Create `flows/hello_nba_flow.py`:

```python
# Import the two core building blocks we need:
# - FlowSpec: the base class for any Metaflow workflow
# - step: the decorator that marks a method as a workflow step
from metaflow import FlowSpec, step


class HelloNBAFlow(FlowSpec):
    # Every Metaflow flow is a Python class that inherits from FlowSpec.
    # Metaflow inspects this class and turns the @step methods into a DAG.

    @step
    def start(self):
        # This line creates an artifact called "message".
        # Because it is assigned to self, Metaflow will persist it automatically
        # after this step completes.
        self.message = "Hello from Metaflow. Next, we will build an NBA model."

        # self.next tells Metaflow which step comes next in the DAG.
        # Here the graph is simple: start -> end
        self.next(self.end)

    @step
    def end(self):
        # This step can read self.message because Metaflow persisted it
        # at the end of the start step and made it available here.
        print(self.message)


if __name__ == "__main__":
    # This line makes the class executable from the command line.
    HelloNBAFlow()
```

> **Understanding the Code**  
> `FlowSpec` is the workflow container. `@step` turns ordinary Python methods into executable graph nodes. `self.next(self.end)` is not just “call the next function”; it defines the DAG edge. `self.message` is not just a normal attribute either; it becomes a persisted artifact that later steps can read.

Run the flow:

```bash
python flows/hello_nba_flow.py run
```

Expected output will look something like:

```text
Metaflow 2.x.y executing HelloNBAFlow for user: ...
Validating your flow...
    The graph looks good!
Running pylint...
    Pylint is happy!
2026-... [start/...] Task is starting.
2026-... [start/...] Task finished successfully.
2026-... [end/...] Hello from Metaflow. Next, we will build an NBA model.
2026-... [end/...] Task finished successfully.
Done!
```

### What Just Happened?

Metaflow validated the graph, executed `start`, stored the `message` artifact, then executed `end` with that artifact available. You did not write any serialization logic, checkpoint code, or DAG plumbing by hand. That is the point of Metaflow.

**Beginner mistake:** forgetting the `if __name__ == "__main__":` block.  
Without it, the file defines a flow but does not expose the command-line entry point correctly.

---

## Step 2 — Parameters and Configuration

In this step, you will learn how to make the same flow reusable across sports, targets, time windows, and model variants.

### What Metaflow Parameters are

Metaflow `Parameter`s are class-level inputs that become part of the run metadata. Compared with `argparse` or loose environment variables, they are better for ML workflows because they are typed, visible on the command line, stored with the run, and accessible as read-only artifacts in every step. Parameters are defined once at the class level and automatically available throughout the flow.

### Why they matter for this sports pipeline

You want one flow that can be reused across:

- sport (`nba`, later `nfl`, `mlb`, etc.)
- target variable (`home_win`, `home_cover`, `total_over`)
- date range
- model type (`xgboost`, `logreg`)
- whether to train on playoff-only games or a broader sample

Create `flows/parameterized_nba_flow.py`:

```python
# Import Parameter so we can define runtime-configurable inputs for the flow.
from metaflow import FlowSpec, step, Parameter


class ParameterizedNBAFlow(FlowSpec):
    # sport is the broadest reuse point. We start with nba, but the flow
    # structure itself can later support nfl or mlb if your data schema matches.
    sport = Parameter(
        "sport",
        default="nba",
        help="Sport to model, for example: nba"
    )

    # target tells the training step what to predict.
    # home_win is the simplest and most stable starting target.
    target = Parameter(
        "target",
        default="home_win",
        help="Target variable: home_win, home_cover, or total_over"
    )

    # start_date and end_date let us carve out training windows
    # without editing code. That is exactly what Parameters are for.
    start_date = Parameter(
        "start-date",
        default="2023-10-01",
        help="Inclusive lower bound for training data"
    )

    end_date = Parameter(
        "end-date",
        default="2026-04-30",
        help="Inclusive upper bound for training data"
    )

    # model_type lets you switch between model families while keeping
    # the workflow identical.
    model_type = Parameter(
        "model-type",
        default="xgboost",
        help="Which model family to train: xgboost or logreg"
    )

    @step
    def start(self):
        print("sport:", self.sport)
        print("target:", self.target)
        print("date range:", self.start_date, "to", self.end_date)
        print("model:", self.model_type)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ParameterizedNBAFlow()
```

Run it with defaults:

```bash
python flows/parameterized_nba_flow.py run
```

Run it with overrides:

```bash
python flows/parameterized_nba_flow.py run \
  --sport nba \
  --target home_cover \
  --start-date 2024-10-01 \
  --end-date 2026-04-30 \
  --model-type xgboost
```

Expected output:

```text
sport: nba
target: home_cover
date range: 2024-10-01 to 2026-04-30
model: xgboost
```

### What Just Happened?

You turned hard-coded assumptions into run metadata. That is not just more convenient; it is what makes every run auditable. Months later, you can look at a model artifact and know exactly which target, dates, and model family produced it. Parameters are persisted at run start, which is why they are much more useful than “I think I set that environment variable last week.”

**Beginner mistake:** using Parameters for huge config trees.  
Use Parameters for small runtime controls. If your config gets hierarchical, Metaflow’s `Config` support is better. For this tutorial, Parameters are enough.

---

## Step 3 — Data Ingestion Step

In this step, you will load sports data inside a step, validate it, and store the raw DataFrame as an artifact.

### What to load in `start` versus later steps

A good default is:

- `start`: load data, perform lightweight validation, and store the raw dataset
- later steps: do transformations that are logically separate, such as feature engineering, train/test split logic, or model training

### Example raw data schema

Assume `data/raw/nba_games.csv` has one row per completed game with columns like:

- `game_id`
- `game_date`
- `season`
- `sport`
- `playoff_game`
- `home_team`
- `away_team`
- `home_points`
- `away_points`
- `home_rebounds`
- `away_rebounds`
- `home_turnovers`
- `away_turnovers`
- `home_rest_days`
- `away_rest_days`
- `closing_spread`
- `closing_total`

From this raw table you will later derive targets such as:

- `home_win`
- `home_cover`
- `total_over`

Create `src/io_utils.py`:

```python
# Standard imports for file handling and tabular data.
from pathlib import Path
import pandas as pd


def load_game_data(csv_path: str) -> pd.DataFrame:
    # Resolve the path early so path mistakes fail fast.
    path = Path(csv_path)

    # Raise a clean error if the file does not exist.
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    # Parse dates as real timestamps now, not later.
    # That keeps sorting and date filtering correct.
    df = pd.read_csv(path, parse_dates=["game_date"])

    return df


def validate_raw_games(df: pd.DataFrame) -> None:
    # Define the minimum schema you expect.
    required = {
        "game_id", "game_date", "sport", "playoff_game",
        "home_team", "away_team",
        "home_points", "away_points",
        "home_rebounds", "away_rebounds",
        "home_turnovers", "away_turnovers",
        "home_rest_days", "away_rest_days",
        "closing_spread", "closing_total"
    }

    # Figure out whether any required columns are missing.
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Basic sanity checks.
    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["game_date"].isna().any():
        raise ValueError("Some game_date values could not be parsed.")

    if (df["home_team"] == df["away_team"]).any():
        raise ValueError("A game row has the same team on both sides.")
```

Create the ingestion flow in `flows/train_nba_model.py`:

```python
from metaflow import FlowSpec, step, Parameter
from src.io_utils import load_game_data, validate_raw_games


class TrainNBAModelFlow(FlowSpec):
    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2023-10-01")
    end_date = Parameter("end-date", default="2026-04-30")
    model_type = Parameter("model-type", default="xgboost")
    input_path = Parameter("input-path", default="data/raw/nba_games.csv")

    @step
    def start(self):
        # Load the raw file from disk.
        df = load_game_data(self.input_path)

        # Validate the incoming schema before you do any work.
        validate_raw_games(df)

        # Filter for sport and date range.
        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()

        # Store the raw filtered dataframe as an artifact.
        # This is valuable for lineage, debugging, and reproducibility.
        self.raw_df = df

        # A tiny summary artifact is useful too.
        self.n_rows = len(df)
        self.n_teams = len(set(df["home_team"]).union(set(df["away_team"])))

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        # You will fill this in during Step 4.
        self.next(self.end)

    @step
    def end(self):
        print(f"Loaded {self.n_rows} rows across {self.n_teams} teams.")


if __name__ == "__main__":
    TrainNBAModelFlow()
```

Run it:

```bash
python flows/train_nba_model.py run \
  --sport nba \
  --target home_win \
  --start-date 2023-10-01 \
  --end-date 2026-04-30 \
  --input-path data/raw/nba_games.csv
```

Expected output:

```text
Loaded 4,832 rows across 30 teams.
```

### What Just Happened?

You moved data loading into a durable workflow step. `self.raw_df` is now a versioned artifact tied to this exact run. If a later feature-engineering or training step fails, you can inspect the same raw frame from the Metaflow Client API rather than re-running the loader and hoping the data has not changed.

**Beginner mistake:** doing too much in `start`.  
If `start` both loads raw data and performs extensive feature engineering, you lose a valuable checkpoint boundary.

---

## Step 4 — Feature Engineering Step

In this step, you will turn raw game logs into model-ready features.

### Why this should be its own step

Feature engineering is often where leakage enters sports models. If you keep it separate from data ingestion, you can inspect exactly what the model saw, compare versions of the engineered table, and avoid accidental dependence on post-game information.

### Feature goals for the NBA game model

You want features that can later support both **next-game forecasts** and **series-level simulations**. So you will compute:

- rolling team scoring averages
- rolling rebounds
- rolling turnovers
- lagged last-game values
- home vs away rest differential
- playoff flag
- series game number if available

Create `src/feature_engineering.py`:

```python
import pandas as pd
import numpy as np


def _to_long_team_view(df: pd.DataFrame) -> pd.DataFrame:
    # Build a "team game log" view with one row per team per game.
    # This is often easier for rolling features than one-row-per-game data.
    home = df[[
        "game_id", "game_date", "home_team", "away_team",
        "home_points", "home_rebounds", "home_turnovers",
        "home_rest_days", "playoff_game"
    ]].copy()

    # Normalize the home-team schema into generic team-side columns.
    home.columns = [
        "game_id", "game_date", "team", "opponent",
        "points", "rebounds", "turnovers",
        "rest_days", "playoff_game"
    ]
    home["is_home"] = 1

    away = df[[
        "game_id", "game_date", "away_team", "home_team",
        "away_points", "away_rebounds", "away_turnovers",
        "away_rest_days", "playoff_game"
    ]].copy()

    away.columns = [
        "game_id", "game_date", "team", "opponent",
        "points", "rebounds", "turnovers",
        "rest_days", "playoff_game"
    ]
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    return long_df


def build_game_level_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to a team-centric history table.
    long_df = _to_long_team_view(df)

    # Rolling averages must use ONLY past games, never the current one.
    # shift(1) is the key anti-leakage move here.
    for col in ["points", "rebounds", "turnovers"]:
        long_df[f"{col}_lag_1"] = long_df.groupby("team")[col].shift(1)
        long_df[f"{col}_rolling_5"] = (
            long_df.groupby("team")[col]
            .shift(1)
            .rolling(5, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Keep the columns you want to merge back later.
    keep = [
        "game_id", "team", "is_home",
        "points_lag_1", "points_rolling_5",
        "rebounds_lag_1", "rebounds_rolling_5",
        "turnovers_lag_1", "turnovers_rolling_5",
        "rest_days", "playoff_game"
    ]
    long_df = long_df[keep]

    # Split into home and away feature tables.
    home_feats = long_df[long_df["is_home"] == 1].copy()
    away_feats = long_df[long_df["is_home"] == 0].copy()

    # Prefix columns so the final game table is explicit and easy to debug.
    home_feats = home_feats.rename(columns={
        "team": "home_team",
        "points_lag_1": "home_points_lag_1",
        "points_rolling_5": "home_points_rolling_5",
        "rebounds_lag_1": "home_rebounds_lag_1",
        "rebounds_rolling_5": "home_rebounds_rolling_5",
        "turnovers_lag_1": "home_turnovers_lag_1",
        "turnovers_rolling_5": "home_turnovers_rolling_5",
        "rest_days": "home_rest_days_feature"
    })

    away_feats = away_feats.rename(columns={
        "team": "away_team",
        "points_lag_1": "away_points_lag_1",
        "points_rolling_5": "away_points_rolling_5",
        "rebounds_lag_1": "away_rebounds_lag_1",
        "rebounds_rolling_5": "away_rebounds_rolling_5",
        "turnovers_lag_1": "away_turnovers_lag_1",
        "turnovers_rolling_5": "away_turnovers_rolling_5",
        "rest_days": "away_rest_days_feature"
    })

    # Merge features back onto the original one-row-per-game frame.
    out = df.merge(home_feats.drop(columns=["is_home", "playoff_game"]), on=["game_id", "home_team"], how="left")
    out = out.merge(away_feats.drop(columns=["is_home", "playoff_game"]), on=["game_id", "away_team"], how="left")

    # Targets: these come from finished games and are fine because training rows are historical.
    out["home_win"] = (out["home_points"] > out["away_points"]).astype(int)
    out["home_cover"] = ((out["home_points"] - out["away_points"]) > out["closing_spread"]).astype(int)
    out["total_over"] = ((out["home_points"] + out["away_points"]) > out["closing_total"]).astype(int)

    # A very useful interaction feature in sports is the difference between two teams.
    out["rest_diff"] = out["home_rest_days_feature"] - out["away_rest_days_feature"]
    out["rolling_points_diff"] = out["home_points_rolling_5"] - out["away_points_rolling_5"]
    out["rolling_rebounds_diff"] = out["home_rebounds_rolling_5"] - out["away_rebounds_rolling_5"]
    out["rolling_turnovers_diff"] = out["home_turnovers_rolling_5"] - out["away_turnovers_rolling_5"]

    return out
```

Update the flow:

```python
from metaflow import FlowSpec, step, Parameter
from src.io_utils import load_game_data, validate_raw_games
from src.feature_engineering import build_game_level_features


class TrainNBAModelFlow(FlowSpec):
    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2023-10-01")
    end_date = Parameter("end-date", default="2026-04-30")
    model_type = Parameter("model-type", default="xgboost")
    input_path = Parameter("input-path", default="data/raw/nba_games.csv")

    @step
    def start(self):
        df = load_game_data(self.input_path)
        validate_raw_games(df)

        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()

        self.raw_df = df
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        # Build the full model table from the historical game log.
        features_df = build_game_level_features(self.raw_df)

        # Drop rows where rolling features do not exist yet.
        # Early-season rows often lack enough history.
        features_df = features_df.dropna().copy()

        self.features_df = features_df
        self.feature_columns = [
            "home_points_lag_1", "home_points_rolling_5",
            "away_points_lag_1", "away_points_rolling_5",
            "home_rebounds_lag_1", "home_rebounds_rolling_5",
            "away_rebounds_lag_1", "away_rebounds_rolling_5",
            "home_turnovers_lag_1", "home_turnovers_rolling_5",
            "away_turnovers_lag_1", "away_turnovers_rolling_5",
            "rest_diff", "rolling_points_diff",
            "rolling_rebounds_diff", "rolling_turnovers_diff",
            "playoff_game"
        ]

        self.next(self.end)

    @step
    def end(self):
        print("Feature table shape:", self.features_df.shape)
```

Run it:

```bash
python flows/train_nba_model.py run --target home_win
```

Expected output:

```text
Feature table shape: (4218, 34)
```

### What Just Happened?

You built a leakage-aware feature table and stored it as an artifact. That means later steps do not need to recompute the features, and you can inspect the exact engineered table that any model version saw. This is one of the biggest practical benefits of using Metaflow instead of a notebook that silently mutates state.

> **Understanding the Code**  
> The crucial line in the feature builder is `shift(1)`. Without it, your rolling averages would include the current game and leak the answer into the features. In sports modeling, that is one of the easiest mistakes to make and one of the easiest ways to build a model that looks incredible in development and collapses in production.

### Inspect intermediate artifacts between steps using the Metaflow client

Create a small inspection script:

```python
from metaflow import Flow

# Grab the latest successful run of the training flow.
run = Flow("TrainNBAModelFlow").latest_successful_run

# Access the feature dataframe artifact.
df = run["feature_engineering"].task.data.features_df

print(df.head())
```

### What Just Happened?

Metaflow’s Client API let you inspect the exact feature table from a past run. That is one of the reasons artifacts matter: every step output can be revisited later, including intermediate steps, not just the final model.

---

## Step 5 — Parallel Hyperparameter Tuning with foreach

In this step, you will fan out multiple model configurations in parallel.

### What foreach does and how Metaflow fans out parallel tasks

`foreach` is Metaflow’s built-in pattern for embarrassingly parallel work. You create a list artifact, then call `self.next(self.some_step, foreach="artifact_name")`. Metaflow creates one task per list item and exposes the current item as `self.input` inside the fanned-out step. Later, you join the branches in a `join(self, inputs)` step.

### Why use it here?

Hyperparameter tuning is a perfect `foreach` use case because each model config can train independently. That means:

- no cross-task communication
- easy local testing
- automatic scale-out later on Batch or Kubernetes

Update the flow with a train/validation split and parameter grid:

```python
from metaflow import FlowSpec, step, Parameter
from sklearn.model_selection import train_test_split
from src.io_utils import load_game_data, validate_raw_games
from src.feature_engineering import build_game_level_features


class TrainNBAModelFlow(FlowSpec):
    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2023-10-01")
    end_date = Parameter("end-date", default="2026-04-30")
    model_type = Parameter("model-type", default="xgboost")
    input_path = Parameter("input-path", default="data/raw/nba_games.csv")

    @step
    def start(self):
        df = load_game_data(self.input_path)
        validate_raw_games(df)
        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()
        self.raw_df = df
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        features_df = build_game_level_features(self.raw_df).dropna().copy()

        self.feature_columns = [
            "home_points_lag_1", "home_points_rolling_5",
            "away_points_lag_1", "away_points_rolling_5",
            "home_rebounds_lag_1", "home_rebounds_rolling_5",
            "away_rebounds_lag_1", "away_rebounds_rolling_5",
            "home_turnovers_lag_1", "home_turnovers_rolling_5",
            "away_turnovers_lag_1", "away_turnovers_rolling_5",
            "rest_diff", "rolling_points_diff",
            "rolling_rebounds_diff", "rolling_turnovers_diff",
            "playoff_game"
        ]

        X = features_df[self.feature_columns]
        y = features_df[self.target]

        # For a tutorial, random split is okay.
        # In production sports work, a chronological split is usually safer.
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define a small but realistic hyperparameter search space.
        self.param_grid = [
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
        ]

        # Fan out one task per config.
        self.next(self.tune_model, foreach="param_grid")

    @step
    def tune_model(self):
        # self.input contains the hyperparameter dictionary for this branch.
        self.current_params = self.input
        self.next(self.join_tuning)

    @step
    def join_tuning(self, inputs):
        # You will fill this in after training.
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainNBAModelFlow()
```

> **Understanding the Code**  
> `self.param_grid` is just a Python list until you pass it to `foreach`. At that moment, Metaflow uses it as the branching artifact and spawns one task per configuration. Inside `tune_model`, `self.input` becomes “the config for this branch.” In `join_tuning`, `inputs` is the collection of finished branch tasks.

Run it:

```bash
python flows/train_nba_model.py run
```

Expected output will include multiple `tune_model` tasks:

```text
[start/...]
[feature_engineering/...]
[tune_model (1/3) ...]
[tune_model (2/3) ...]
[tune_model (3/3) ...]
[join_tuning/...]
[end/...]
```

### What Just Happened?

Metaflow fanned out your model search into parallel tasks. Locally, those tasks may execute with limited parallelism depending on your environment. In cloud execution, the same `foreach` pattern can launch many independent tasks remotely. Metaflow treats `foreach` as a first-class graph construct, not a hand-rolled multiprocessing loop.

**Beginner mistake:** making `param_grid` huge on your first run.  
Start small and scale deliberately.

---

## Step 6 — Model Training Step

In this step, you will actually train the models inside the fanned-out branches, store model objects as artifacts, and then select the best one in the join step.

### Why Metaflow versioning means you never lose a trained model

Metaflow versions artifacts automatically. That means your trained model object is not “the file in `models/latest.pkl`.” It is “the model produced by this exact run, step, code package, parameters, and upstream data artifacts.” That is a much stronger guarantee.

Update the flow:

```python
from metaflow import FlowSpec, step, Parameter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.io_utils import load_game_data, validate_raw_games
from src.feature_engineering import build_game_level_features


class TrainNBAModelFlow(FlowSpec):
    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2023-10-01")
    end_date = Parameter("end-date", default="2026-04-30")
    model_type = Parameter("model-type", default="xgboost")
    input_path = Parameter("input-path", default="data/raw/nba_games.csv")

    @step
    def start(self):
        df = load_game_data(self.input_path)
        validate_raw_games(df)
        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()
        self.raw_df = df
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        features_df = build_game_level_features(self.raw_df).dropna().copy()

        self.feature_columns = [
            "home_points_lag_1", "home_points_rolling_5",
            "away_points_lag_1", "away_points_rolling_5",
            "home_rebounds_lag_1", "home_rebounds_rolling_5",
            "away_rebounds_lag_1", "away_rebounds_rolling_5",
            "home_turnovers_lag_1", "home_turnovers_rolling_5",
            "away_turnovers_lag_1", "away_turnovers_rolling_5",
            "rest_diff", "rolling_points_diff",
            "rolling_rebounds_diff", "rolling_turnovers_diff",
            "playoff_game"
        ]

        X = features_df[self.feature_columns]
        y = features_df[self.target]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.param_grid = [
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
        ]

        self.next(self.tune_model, foreach="param_grid")

    @step
    def tune_model(self):
        # Save the branch-specific hyperparameters for inspection later.
        self.current_params = self.input

        # Build the model for this branch.
        model = XGBClassifier(
            max_depth=self.current_params["max_depth"],
            learning_rate=self.current_params["learning_rate"],
            n_estimators=self.current_params["n_estimators"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

        # Train on the branch-local training split.
        model.fit(self.X_train, self.y_train)

        # Predict probabilities because AUC and log loss need probabilities.
        preds_proba = model.predict_proba(self.X_valid)[:, 1]
        preds_label = (preds_proba >= 0.5).astype(int)

        # Persist metrics as artifacts.
        self.valid_accuracy = accuracy_score(self.y_valid, preds_label)
        self.valid_auc = roc_auc_score(self.y_valid, preds_proba)
        self.valid_logloss = log_loss(self.y_valid, preds_proba)

        # Persist the trained model object itself as an artifact.
        self.model = model

        self.next(self.join_tuning)

    @step
    def join_tuning(self, inputs):
        # Collect all branch results into a simple list of dicts.
        results = []
        for branch in inputs:
            results.append({
                "params": branch.current_params,
                "accuracy": branch.valid_accuracy,
                "auc": branch.valid_auc,
                "logloss": branch.valid_logloss,
                "model": branch.model,
            })

        # Higher AUC is better, so choose the model with max AUC.
        best = max(results, key=lambda x: x["auc"])

        # Save the winning branch artifacts to the main flow.
        self.best_params = best["params"]
        self.best_accuracy = best["accuracy"]
        self.best_auc = best["auc"]
        self.best_logloss = best["logloss"]
        self.best_model = best["model"]

        # Save the full search history too.
        self.tuning_results = results

        self.next(self.end)

    @step
    def end(self):
        print("Best params:", self.best_params)
        print("Best AUC:", self.best_auc)


if __name__ == "__main__":
    TrainNBAModelFlow()
```

Run it:

```bash
python flows/train_nba_model.py run
```

Expected output:

```text
Best params: {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 300}
Best AUC: 0.6812
```

### What Just Happened?

Each `tune_model` branch trained a separate model and stored both its metrics and the model object itself. The join step compared all branches and promoted the best one into `self.best_model`. That model is now versioned with the run. You did not save it manually to a file, but you also did not lose it. That is one of the biggest conceptual upgrades Metaflow gives you.

**Beginner mistake:** picking the best model only by accuracy.  
For sports classification, accuracy can be misleading if classes are imbalanced or if you care about probabilities. AUC and log loss are often better comparisons.

---

## Step 7 — Model Evaluation and the `@card` Decorator

In this step, you will create a visual evaluation report that travels with the run.

### How to add a `@card` to generate a visual HTML report for a run

Metaflow Cards exist so you can attach human-readable reports directly to workflow steps. You can add a default card with almost no work, or build custom reports using Markdown, tables, images, and charts. Cards can be viewed from the CLI, in notebooks, or in the Metaflow UI.

### What to put in the evaluation card

For this NBA model, a useful evaluation card should include:

- target name
- best hyperparameters
- accuracy, AUC, log loss
- confusion matrix
- feature importance
- calibration plot

Update the flow:

```python
from metaflow import FlowSpec, step, Parameter, card, current
from metaflow.cards import Markdown, Table, Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.io_utils import load_game_data, validate_raw_games
from src.feature_engineering import build_game_level_features


class TrainNBAModelFlow(FlowSpec):
    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2023-10-01")
    end_date = Parameter("end-date", default="2026-04-30")
    model_type = Parameter("model-type", default="xgboost")
    input_path = Parameter("input-path", default="data/raw/nba_games.csv")

    @step
    def start(self):
        df = load_game_data(self.input_path)
        validate_raw_games(df)
        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()
        self.raw_df = df
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        features_df = build_game_level_features(self.raw_df).dropna().copy()

        self.feature_columns = [
            "home_points_lag_1", "home_points_rolling_5",
            "away_points_lag_1", "away_points_rolling_5",
            "home_rebounds_lag_1", "home_rebounds_rolling_5",
            "away_rebounds_lag_1", "away_rebounds_rolling_5",
            "home_turnovers_lag_1", "home_turnovers_rolling_5",
            "away_turnovers_lag_1", "away_turnovers_rolling_5",
            "rest_diff", "rolling_points_diff",
            "rolling_rebounds_diff", "rolling_turnovers_diff",
            "playoff_game"
        ]

        X = features_df[self.feature_columns]
        y = features_df[self.target]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.param_grid = [
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
        ]

        self.next(self.tune_model, foreach="param_grid")

    @step
    def tune_model(self):
        self.current_params = self.input

        model = XGBClassifier(
            max_depth=self.current_params["max_depth"],
            learning_rate=self.current_params["learning_rate"],
            n_estimators=self.current_params["n_estimators"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(self.X_train, self.y_train)

        preds_proba = model.predict_proba(self.X_valid)[:, 1]
        preds_label = (preds_proba >= 0.5).astype(int)

        self.valid_accuracy = accuracy_score(self.y_valid, preds_label)
        self.valid_auc = roc_auc_score(self.y_valid, preds_proba)
        self.valid_logloss = log_loss(self.y_valid, preds_proba)
        self.model = model

        self.next(self.join_tuning)

    @card(type="blank")
    @step
    def join_tuning(self, inputs):
        results = []
        for branch in inputs:
            results.append({
                "params": branch.current_params,
                "accuracy": branch.valid_accuracy,
                "auc": branch.valid_auc,
                "logloss": branch.valid_logloss,
                "model": branch.model,
            })

        best = max(results, key=lambda x: x["auc"])

        self.best_params = best["params"]
        self.best_accuracy = best["accuracy"]
        self.best_auc = best["auc"]
        self.best_logloss = best["logloss"]
        self.best_model = best["model"]
        self.tuning_results = results

        # Recompute validation predictions for the card visuals.
        preds_proba = self.best_model.predict_proba(self.X_valid)[:, 1]
        preds_label = (preds_proba >= 0.5).astype(int)

        # Build a confusion matrix table.
        cm = confusion_matrix(self.y_valid, preds_label)
        cm_df = pd.DataFrame(
            cm,
            index=["actual_0", "actual_1"],
            columns=["pred_0", "pred_1"]
        )

        # Build feature importance dataframe.
        feat_imp = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.best_model.feature_importances_
        }).sort_values("importance", ascending=False).head(10)

        # Calibration plot as a Matplotlib figure.
        fig, ax = plt.subplots(figsize=(6, 4))
        CalibrationDisplay.from_predictions(self.y_valid, preds_proba, n_bins=10, ax=ax)
        ax.set_title("Calibration Plot")

        # Populate the card.
        current.card.append(Markdown("# NBA Model Evaluation"))
        current.card.append(Markdown(f"**Target:** `{self.target}`"))
        current.card.append(Markdown(f"**Best Params:** `{self.best_params}`"))

        current.card.append(
            Table([
                ["Accuracy", f"{self.best_accuracy:.4f}"],
                ["AUC", f"{self.best_auc:.4f}"],
                ["Log Loss", f"{self.best_logloss:.4f}"],
            ], headers=["Metric", "Value"])
        )

        current.card.append(Markdown("## Confusion Matrix"))
        current.card.append(Table.from_dataframe(cm_df))

        current.card.append(Markdown("## Top Feature Importance"))
        current.card.append(Table.from_dataframe(feat_imp))

        current.card.append(Markdown("## Calibration"))
        current.card.append(Image.from_matplotlib(fig))

        self.next(self.end)

    @step
    def end(self):
        print("Training complete.")


if __name__ == "__main__":
    TrainNBAModelFlow()
```

> **Understanding the Code**  
> `@card(type="blank")` gives you an empty canvas. Inside the step, `current.card.append(...)` lets you add visual components. The card becomes part of the run itself, which is why it is so valuable for model review and operational monitoring later.

Run it:

```bash
python flows/train_nba_model.py run
```

Then view the card:

```bash
python flows/train_nba_model.py card view join_tuning
```

Expected output:

```text
Opening card for step join_tuning in your browser...
```

### What Just Happened?

You generated a shareable HTML report for the model selection step. Cards are one of the best Metaflow features for production-grade ML because they make runs inspectable by humans, not just machines. They also work locally, remotely, and in UIs.

**Beginner mistake:** building giant static HTML strings by hand.  
Use card components first. They are easier to maintain and look much better by default.

---

## Step 8 — Reusing Models Across Flows

In this step, you will use the Metaflow Client API to retrieve a trained model from a past run and build a separate prediction flow.

### How to use the Metaflow Client API to load a trained model from a past run

The Client API is designed for exactly this pattern: inspect past runs, retrieve artifacts, and use tags and namespaces to organize which runs count as “production.”

### The pattern for “latest successful run” model retrieval

The Client API gives you both `latest_run` and `latest_successful_run`. That second one is what you almost always want for loading models. You can also filter by tags, which is useful for separating production from experiments.

Create `flows/predict_nba_model.py`:

```python
from metaflow import FlowSpec, step, Parameter, Flow


class PredictNBAModelFlow(FlowSpec):
    # Path to a file containing upcoming games with the same feature columns
    # as the training flow expects after feature engineering.
    input_path = Parameter(
        "input-path",
        default="data/prediction/upcoming_games.csv",
        help="CSV containing upcoming NBA games with model-ready features"
    )

    # Optional tag filter so you can explicitly choose production runs.
    model_tag = Parameter(
        "model-tag",
        default="production",
        help="Tag used to identify approved training runs"
    )

    @step
    def start(self):
        # Access the training flow by name.
        train_flow = Flow("TrainNBAModelFlow")

        # Filter runs by tag if available; otherwise fall back to latest successful.
        tagged_runs = list(train_flow.runs(self.model_tag))
        if tagged_runs:
            latest = next((r for r in tagged_runs if r.successful), None)
            if latest is None:
                raise ValueError(f"No successful runs found with tag '{self.model_tag}'")
        else:
            latest = train_flow.latest_successful_run
            if latest is None:
                raise ValueError("No successful TrainNBAModelFlow runs found.")

        # Pull the final promoted model artifact from join_tuning.
        self.model = latest["join_tuning"].task.data.best_model
        self.feature_columns = latest["feature_engineering"].task.data.feature_columns
        self.source_run_id = latest.id

        self.next(self.load_input)

    @step
    def load_input(self):
        import pandas as pd

        # Load upcoming games for inference.
        df = pd.read_csv(self.input_path)

        # Select exactly the same columns the model was trained on.
        X = df[self.feature_columns]

        # Predict probabilities.
        preds = self.model.predict_proba(X)[:, 1]

        # Attach predictions to the dataframe.
        df["pred_home_win_proba"] = preds
        self.predictions = df

        self.next(self.end)

    @step
    def end(self):
        print(f"Used model from TrainNBAModelFlow run: {self.source_run_id}")
        print(self.predictions.head())


if __name__ == "__main__":
    PredictNBAModelFlow()
```

Run it:

```bash
python flows/predict_nba_model.py run --model-tag production
```

Expected output:

```text
Used model from TrainNBAModelFlow run: 184
   game_id home_team away_team pred_home_win_proba
0  ...
```

### What Just Happened?

The prediction flow did not retrain anything. It used the Client API to load the latest successful model artifact from a prior training run. That is a production-friendly pattern because it keeps training and inference decoupled while preserving lineage.

### How to use tags to label production vs. experimental runs

A common pattern is to tag runs after review:

- `production`
- `candidate`
- `experiment`
- `nightly-retrain`

You can then filter the training flow by tag when loading models in downstream flows.

**Beginner mistake:** always loading `latest_run`.  
If the latest run failed halfway through tuning, that is not the model you want. Prefer `latest_successful_run` or tag-filtered successful runs.

---

## Step 9 — Scheduling and Production Deployment

In this step, you will learn how to move from “I run this manually” to “the platform runs this reliably.”

### How to deploy to Argo Workflows (Outerbounds) or AWS Step Functions

For **Outerbounds**, the natural production path is **Argo Workflows** on the managed platform.

For **self-hosted AWS Metaflow**, the common path is **AWS Step Functions**, where Metaflow maps a `FlowSpec` to a Step Functions state machine and runs steps on AWS Batch under the hood.

### How to add `@schedule` for automatic nightly/weekly retraining

Use a flow-level decorator above the class:

```python
from metaflow import FlowSpec, step, schedule

@schedule(daily=True)
class TrainNBAModelFlow(FlowSpec):
    ...
```

`schedule` supports common cadences like `hourly`, `daily`, `weekly`, and custom cron expressions when deployed to an external orchestrator such as Argo Workflows or AWS Step Functions.

### How to use `@trigger_on_finish` to chain flows

If you want one flow to start automatically after another completes, use `@trigger_on_finish`.

For example, you might create:

- `IngestNBADataFlow`
- `TrainNBAModelFlow`
- `PredictNBAModelFlow`

And annotate the training flow like:

```python
from metaflow import FlowSpec, step, trigger_on_finish

@trigger_on_finish(flow="IngestNBADataFlow")
class TrainNBAModelFlow(FlowSpec):
    ...
```

### How to monitor scheduled runs in the Outerbounds UI

Once deployed on Outerbounds, the UI becomes the main place to:

- view deployments
- inspect scheduled runs
- open cards
- compare runs
- examine logs

### Deploy to Argo on Outerbounds

Example command:

```bash
python flows/train_nba_model.py --with retry argo-workflows create
```

Expected output will look roughly like:

```text
Deploying trainnbamodelflow to Argo Workflows...
Workflow trainnbamodelflow pushed to Argo Workflows successfully.
This workflow triggers automatically via the CronWorkflow ...
```

### What Just Happened?

You packaged the current code and deployed it to a production orchestrator. With `@schedule`, that deployment can now run automatically. On Outerbounds, the deployment appears in the Deployments view, and runs appear in the Runs view.

### Deploy to AWS Step Functions

Example command:

```bash
python flows/train_nba_model.py --with retry step-functions create
```

Trigger manually:

```bash
python flows/train_nba_model.py step-functions trigger --target home_win
```

### What Just Happened?

Metaflow exported the flow as a Step Functions state machine. The production run uses the same Python code, but orchestration happens in AWS and task execution happens through AWS Batch for the steps that run remotely.

**Beginner mistake:** scheduling before your flow is stable locally.  
Always get `python flow.py run` working first. Deployment should feel like packaging a working pipeline, not debugging one remotely.

---

## Step 10 — Scaling to Cloud Compute

In this step, you will move selected steps from your laptop to cloud compute.

### How to add `@batch` or `@kubernetes` to a step to move it to cloud compute

Metaflow supports remote execution with `@batch` for AWS Batch and `@kubernetes` for Kubernetes. You can also choose the compute layer at runtime with `--with batch` or `--with kubernetes`, which is one reason the “local first, cloud second” workflow feels so natural in Metaflow.

### How to use `@pypi` or `@conda` to manage dependencies per step

Use `@pypi` or `@conda` when you want step-scoped dependency isolation. Metaflow’s dependency support makes remote execution reproducible by snapshotting environments per step.

Example:

```python
from metaflow import step, pypi

@pypi(packages={
    "xgboost": "2.1.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "matplotlib": "3.9.2"
})
@step
def tune_model(self):
    # The required libraries are installed for this step's environment.
    # This matters most when the step runs remotely.
    ...
```

### How to request GPU resources for a step if needed

Use `@resources` with `@kubernetes` or `@batch`.

Example:

```python
from metaflow import step, resources, kubernetes

@kubernetes
@resources(cpu=4, memory=16000, gpu=1)
@step
def tune_model(self):
    # This step will use local resources during local development,
    # but when run with a configured Kubernetes backend it can request
    # 4 CPUs, 16 GB of RAM, and 1 GPU.
    ...
```

> **Understanding the Code**  
> `@pypi` and `@conda` are not just convenience wrappers around `pip install`. Under the hood, Metaflow builds isolated execution environments for steps, packages your local code, and snapshots the dependency resolution so the remote step can be re-created reliably later.

### The local → cloud workflow: develop locally, deploy to cloud with one flag

You can keep the code undecorated and request cloud execution at runtime:

```bash
python flows/train_nba_model.py run --with kubernetes
```

Or:

```bash
python flows/train_nba_model.py run --with batch
```

Expected output will show remote tasks being launched instead of local execution.

### What Just Happened?

You used the same flow code but changed the execution environment. That is one of Metaflow’s best ideas: the workflow definition stays stable while the compute layer can change underneath it.

**Beginner mistake:** assuming local imports automatically exist in the cloud.  
If your remote step depends on third-party packages, declare them explicitly with `@pypi`, `@conda`, or a proper project environment strategy.

---

## Step 11 — Inspecting and Debugging

In this step, you will learn how to inspect old runs, debug failures, and resume from checkpoints.

### How to use the Metaflow CLI to inspect past runs

The Client API is the most reliable way to inspect run history, artifacts, and metadata. It gives you objects like `Flow`, `Run`, `Step`, and `Task`, and it exposes helpers such as `latest_run`, `latest_successful_run`, filtering by tags, and direct access to `run.data`.

Example inspection script:

```python
from metaflow import Flow

flow = Flow("TrainNBAModelFlow")

# Look up the latest successful run.
run = flow.latest_successful_run

print("Run ID:", run.id)
print("Was successful?", run.successful)
print("Best AUC:", run["join_tuning"].task.data.best_auc)
print("Best Params:", run["join_tuning"].task.data.best_params)
```

### How to use logs

Metaflow exposes logs from past runs through flow-specific CLI commands.

Example:

```bash
python flows/train_nba_model.py logs 15/join_tuning
```

Expected output:

```text
2026-... [15/join_tuning/...]
Best params: ...
Best AUC: ...
```

### How to resume a failed run from a checkpoint step

If a flow fails, Metaflow’s `resume` command reuses successful upstream steps and re-executes from the failed point, rather than starting over. This is one of the most valuable debugging features in the framework.

Run a failed flow, fix the bug, then resume:

```bash
python flows/train_nba_model.py resume
```

Or resume from a specific origin run:

```bash
python flows/train_nba_model.py resume --origin-run-id 184
```

### What Just Happened?

Metaflow reused artifacts from successful upstream steps and replayed only the parts that needed to run again. That works because each successful step is a checkpoint.

### Common errors and how to fix them

**Error: `ModuleNotFoundError` in remote step**  
Cause: you installed a library locally but did not declare it in `@pypi`, `@conda`, or your environment strategy.  
Fix: make dependencies explicit.

**Error: feature leakage makes metrics unrealistically high**  
Cause: rolling features were built without `shift(1)` or with post-game values.  
Fix: enforce a “past games only” policy in feature engineering.

**Error: `foreach` launches too many tasks**  
Cause: oversized grid or accidental fan-out.  
Fix: shrink the grid first or control parallelism explicitly.

**Error: prediction flow loads the wrong model**  
Cause: using `latest_run` instead of `latest_successful_run`, or not filtering by production tag.  
Fix: always retrieve the latest successful tagged run.

---

## Conclusion

You now have the blueprint for a production-grade sports analytics pipeline in Metaflow.

### Full architecture diagram of the complete pipeline

```text
Raw NBA game data
        |
        v
+----------------------+
| TrainNBAModelFlow    |
|----------------------|
| start                |  -> load + validate raw data
| feature_engineering  |  -> rolling / lag / differential features
| tune_model (foreach) |  -> parallel hyperparameter tuning
| join_tuning          |  -> select best model + build evaluation card
| end                  |
+----------------------+
        |
        | artifacts + metadata + card + tags
        v
Metaflow datastore / metadata service
        |
        v
+----------------------+
| PredictNBAModelFlow  |
|----------------------|
| start                |  -> load latest successful tagged model
| load_input           |  -> score upcoming games
| end                  |
+----------------------+
        |
        +--> optional @schedule for nightly/weekly retraining
        +--> optional @trigger_on_finish for flow chaining
        +--> optional Argo / Step Functions deployment
        +--> optional Batch / Kubernetes remote compute
```

### Summary of every Metaflow concept introduced

You used:

- `FlowSpec` to define the workflow as a DAG
- `@step` to define execution nodes
- `self.next(...)` to define transitions
- artifacts (`self.x`) to persist data between steps
- `Parameter` to make runs configurable and reproducible
- a separate ingestion step for cleaner lineage
- a separate feature step for leakage control and inspectability
- `foreach` for parallel hyperparameter tuning
- a join step with `inputs` to collect branch results
- model objects as versioned artifacts
- `@card` and card components to generate evaluation reports
- the Client API to inspect past runs and load prior models
- tags to distinguish production from experiments
- `@schedule` for automated retraining
- `@trigger_on_finish` to chain flows
- `@batch`, `@kubernetes`, and `@resources` for cloud compute
- `@pypi` / `@conda` for dependency isolation
- `resume` and logs for debugging

### What to explore next

Once this base pipeline feels natural, the most useful next topics are:

- Metaflow + Ray for more advanced distributed workloads
- checkpointing long model training with `@checkpoint`
- richer card dashboards with dynamic updates
- recursive series simulation as a downstream flow
- a separate betting-edge flow that compares your fair line to the market
- eventually, a RAG or agent workflow that summarizes injury news and coaching adjustments before the game model runs

The important part is that you now understand **why** Metaflow exists, **when** each concept becomes useful, and **how** the pieces fit together. You are not just copying a pipeline. You are learning the control surface of the system.

The next natural move is to turn this tutorial into actual runnable project files so you can execute it locally step by step and then deploy it.
