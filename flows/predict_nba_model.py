"""
NBA Model Prediction Flow

A Metaflow pipeline that:
1. Loads the latest successful trained model from TrainNBAModelFlow
2. Loads upcoming games for prediction
3. Engineers features for those games
4. Generates predictions
5. Saves results

Usage:
    python flows/predict_nba_model.py run
    python flows/predict_nba_model.py run --input-path data/prediction/custom_games.csv
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import numpy as np

from src.io_utils import load_game_data, save_predictions
from src.feature_engineering import build_enhanced_features, get_enhanced_feature_columns


class PredictNBAModelFlow(FlowSpec):
    """
    Prediction flow for NBA games.

    Loads the trained model from TrainNBAModelFlow and generates
    predictions for upcoming games.
    """

    input_path = Parameter(
        "input-path",
        default="data/prediction/upcoming_games.csv",
        help="CSV containing upcoming NBA games"
    )

    historical_path = Parameter(
        "historical-path",
        default="data/processed/nba_games_processed.csv",
        help="CSV containing historical games for feature calculation"
    )

    output_path = Parameter(
        "output-path",
        default="results/predictions.csv",
        help="Where to save prediction results"
    )

    model_tag = Parameter(
        "model-tag",
        default="",
        help="Optional tag to filter training runs (empty = latest successful)"
    )

    @step
    def start(self):
        """Load the trained model from a previous training run."""
        print("Loading trained model...")

        train_flow = Flow("TrainNBAModelFlow")

        # Find the appropriate training run
        if self.model_tag:
            tagged_runs = list(train_flow.runs(self.model_tag))
            if tagged_runs:
                latest = next((r for r in tagged_runs if r.successful), None)
                if latest is None:
                    raise ValueError(f"No successful runs found with tag '{self.model_tag}'")
            else:
                print(f"No runs with tag '{self.model_tag}', using latest successful run")
                latest = train_flow.latest_successful_run
        else:
            latest = train_flow.latest_successful_run

        if latest is None:
            raise ValueError("No successful TrainNBAModelFlow runs found. Run the training flow first.")

        # Load artifacts from the training run
        join_step = latest["join_tuning"].task.data
        fe_step = latest["feature_engineering"].task.data

        self.model = join_step.best_model
        self.feature_columns = fe_step.feature_columns
        self.training_target = latest.data.target
        self.source_run_id = latest.id

        print(f"Loaded model from run: {self.source_run_id}")
        print(f"Model target: {self.training_target}")
        print(f"Features: {len(self.feature_columns)} columns")

        self.next(self.load_data)

    @step
    def load_data(self):
        """Load upcoming games and historical data for feature engineering."""
        print(f"Loading upcoming games from {self.input_path}")
        self.upcoming_df = pd.read_csv(self.input_path, parse_dates=["game_date"])
        print(f"  Found {len(self.upcoming_df)} upcoming games")

        print(f"Loading historical games from {self.historical_path}")
        self.historical_df = load_game_data(self.historical_path)
        print(f"  Found {len(self.historical_df)} historical games")

        self.next(self.prepare_features)

    @step
    def prepare_features(self):
        """
        Prepare features for upcoming games.

        Uses historical data to compute rolling averages and lag features
        for the teams playing in upcoming games.
        """
        print("Preparing features for prediction...")

        # Mark datasets
        upcoming = self.upcoming_df.copy()
        historical = self.historical_df.copy()

        upcoming["_is_upcoming"] = True
        historical["_is_upcoming"] = False

        # Add placeholder columns for upcoming games (needed for feature engineering)
        placeholder_cols = [
            "home_points", "away_points",
            "home_rebounds", "away_rebounds",
            "home_turnovers", "away_turnovers"
        ]
        for col in placeholder_cols:
            if col not in upcoming.columns:
                upcoming[col] = np.nan

        # Combine and sort by date
        combined = pd.concat([historical, upcoming], ignore_index=True)
        combined = combined.sort_values("game_date").reset_index(drop=True)

        # Build features for all games
        featured = build_enhanced_features(combined)

        # Extract only the upcoming games with features
        prediction_df = featured[featured["_is_upcoming"] == True].copy()

        # Check for missing features
        missing_features = prediction_df[self.feature_columns].isna().any(axis=1)
        if missing_features.any():
            n_missing = missing_features.sum()
            print(f"Warning: {n_missing} games have missing features (teams with no history)")
            prediction_df = prediction_df[~missing_features]

        self.prediction_df = prediction_df
        print(f"Prepared features for {len(self.prediction_df)} games")

        self.next(self.predict)

    @step
    def predict(self):
        """Generate predictions using the loaded model."""
        print("Generating predictions...")

        if len(self.prediction_df) == 0:
            print("Warning: No games to predict!")
            self.predictions = pd.DataFrame()
            self.next(self.end)
            return

        # Extract features
        X = self.prediction_df[self.feature_columns]

        # Generate probability predictions
        proba = self.model.predict_proba(X)[:, 1]

        # Build results dataframe
        results = self.prediction_df[[
            "game_id", "game_date", "home_team", "away_team", "playoff_game"
        ]].copy()

        results["pred_home_win_proba"] = proba
        results["pred_home_win"] = (proba >= 0.5).astype(int)
        results["model_run_id"] = self.source_run_id
        results["prediction_timestamp"] = datetime.now().isoformat()

        self.predictions = results

        # Save to file
        save_predictions(results, self.output_path)

        print(f"\nPrediction Summary:")
        print(f"  Total games: {len(results)}")
        print(f"  Predicted home wins: {results['pred_home_win'].sum()}")
        print(f"  Predicted away wins: {len(results) - results['pred_home_win'].sum()}")
        print(f"  Avg home win probability: {results['pred_home_win_proba'].mean():.3f}")

        self.next(self.end)

    @step
    def end(self):
        """Final step - print summary."""
        print("\n" + "=" * 60)
        print("Prediction Complete!")
        print("=" * 60)
        print(f"Model source: TrainNBAModelFlow run {self.source_run_id}")
        print(f"Games predicted: {len(self.predictions)}")
        print(f"Results saved to: {self.output_path}")
        print("=" * 60)

        if len(self.predictions) > 0:
            print("\nTop predictions (sorted by confidence):")
            top = self.predictions.sort_values("pred_home_win_proba", ascending=False).head(5)
            for _, row in top.iterrows():
                print(f"  {row['home_team']} vs {row['away_team']}: "
                      f"{row['pred_home_win_proba']:.1%} home win probability")


if __name__ == "__main__":
    PredictNBAModelFlow()
