"""
NBA Model Training Flow

Trains an ML model to predict NBA game outcomes using Metaflow.

Features:
- Rolling averages for box score stats
- Season win percentages and streaks
- Parallel hyperparameter tuning
- Multiple model types (XGBoost, GBM, RF, LogReg)
- Chronological train/validation split
- Evaluation card generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metaflow import FlowSpec, step, Parameter, card, current
from metaflow.cards import Markdown, Table, Image

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.io_utils import load_game_data, validate_raw_games
from src.feature_engineering import build_enhanced_features, get_enhanced_feature_columns


class TrainNBAModelFlow(FlowSpec):
    """Training flow for NBA game prediction model."""

    sport = Parameter("sport", default="nba")
    target = Parameter("target", default="home_win")
    start_date = Parameter("start-date", default="2022-10-01")
    end_date = Parameter("end-date", default="2025-12-31")
    input_path = Parameter("input-path", default="data/processed/nba_games_enhanced.csv")

    @step
    def start(self):
        """Load and validate enhanced game data."""
        print(f"Loading enhanced data from {self.input_path}")

        df = load_game_data(self.input_path)

        # Filter by sport and date
        df = df[
            (df["sport"] == self.sport) &
            (df["game_date"] >= self.start_date) &
            (df["game_date"] <= self.end_date)
        ].copy()

        self.raw_df = df
        self.n_rows = len(df)

        print(f"Loaded {self.n_rows} games")
        print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"Columns available: {len(df.columns)}")

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Build enhanced features."""
        print("Building enhanced features...")

        features_df = build_enhanced_features(self.raw_df)

        # Get available feature columns
        self.feature_columns = get_enhanced_feature_columns(features_df)
        print(f"Feature columns: {len(self.feature_columns)}")
        print(f"Features: {self.feature_columns}")

        # Drop rows with missing features
        before = len(features_df)
        features_df = features_df.dropna(subset=self.feature_columns).copy()
        after = len(features_df)
        print(f"Dropped {before - after} rows with missing features")

        X = features_df[self.feature_columns]
        y = features_df[self.target]

        # Use chronological split for more realistic evaluation
        # Last 20% of games as validation
        split_idx = int(len(features_df) * 0.8)
        features_df = features_df.sort_values("game_date")

        self.X_train = features_df.iloc[:split_idx][self.feature_columns]
        self.X_valid = features_df.iloc[split_idx:][self.feature_columns]
        self.y_train = features_df.iloc[:split_idx][self.target]
        self.y_valid = features_df.iloc[split_idx:][self.target]

        print(f"Training set: {len(self.X_train)} rows (through {features_df.iloc[split_idx-1]['game_date'].date()})")
        print(f"Validation set: {len(self.X_valid)} rows (from {features_df.iloc[split_idx]['game_date'].date()})")
        print(f"Target distribution - Train: {self.y_train.mean():.3f}, Valid: {self.y_valid.mean():.3f}")

        # Extended hyperparameter grid
        self.param_grid = [
            # XGBoost configurations
            {"model": "xgboost", "max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            {"model": "xgboost", "max_depth": 4, "learning_rate": 0.03, "n_estimators": 300},
            {"model": "xgboost", "max_depth": 5, "learning_rate": 0.02, "n_estimators": 400},
            {"model": "xgboost", "max_depth": 3, "learning_rate": 0.1, "n_estimators": 150},
            # Gradient Boosting
            {"model": "gbm", "max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
            {"model": "gbm", "max_depth": 4, "learning_rate": 0.03, "n_estimators": 300},
            # Random Forest
            {"model": "rf", "n_estimators": 200, "max_depth": 10},
            {"model": "rf", "n_estimators": 300, "max_depth": 15},
            # Logistic Regression
            {"model": "logreg", "C": 1.0},
            {"model": "logreg", "C": 0.1},
        ]

        self.next(self.tune_model, foreach="param_grid")

    @step
    def tune_model(self):
        """Train a model with specific configuration."""
        self.current_params = self.input
        model_type = self.current_params["model"]

        print(f"Training {model_type} with params: {self.current_params}")

        # Build model based on type
        if model_type == "xgboost":
            model = XGBClassifier(
                max_depth=self.current_params["max_depth"],
                learning_rate=self.current_params["learning_rate"],
                n_estimators=self.current_params["n_estimators"],
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        elif model_type == "gbm":
            model = GradientBoostingClassifier(
                max_depth=self.current_params["max_depth"],
                learning_rate=self.current_params["learning_rate"],
                n_estimators=self.current_params["n_estimators"],
                subsample=0.9,
                random_state=42,
            )
        elif model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=self.current_params["n_estimators"],
                max_depth=self.current_params["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "logreg":
            model = LogisticRegression(
                C=self.current_params["C"],
                max_iter=1000,
                random_state=42,
            )

        model.fit(self.X_train, self.y_train)

        # Evaluate
        preds_proba = model.predict_proba(self.X_valid)[:, 1]
        preds_label = (preds_proba >= 0.5).astype(int)

        self.valid_accuracy = accuracy_score(self.y_valid, preds_label)
        self.valid_auc = roc_auc_score(self.y_valid, preds_proba)
        self.valid_logloss = log_loss(self.y_valid, preds_proba)
        self.valid_brier = brier_score_loss(self.y_valid, preds_proba)

        print(f"  Accuracy: {self.valid_accuracy:.4f}")
        print(f"  AUC: {self.valid_auc:.4f}")
        print(f"  Log Loss: {self.valid_logloss:.4f}")
        print(f"  Brier Score: {self.valid_brier:.4f}")

        self.model = model

        self.next(self.join_tuning)

    @card(type="blank")
    @step
    def join_tuning(self, inputs):
        """Select best model and build evaluation card."""
        print("Collecting tuning results...")

        results = []
        for branch in inputs:
            results.append({
                "params": branch.current_params,
                "accuracy": branch.valid_accuracy,
                "auc": branch.valid_auc,
                "logloss": branch.valid_logloss,
                "brier": branch.valid_brier,
                "model": branch.model,
            })

        # Sort by AUC
        results.sort(key=lambda x: x["auc"], reverse=True)

        best = results[0]

        self.best_params = best["params"]
        self.best_accuracy = best["accuracy"]
        self.best_auc = best["auc"]
        self.best_logloss = best["logloss"]
        self.best_brier = best["brier"]
        self.best_model = best["model"]
        self.tuning_results = results

        self.X_valid = inputs[0].X_valid
        self.y_valid = inputs[0].y_valid
        self.feature_columns = inputs[0].feature_columns

        print(f"\nBest model: {self.best_params['model']}")
        print(f"  Params: {self.best_params}")
        print(f"  AUC: {self.best_auc:.4f}")

        self._build_evaluation_card()

        self.next(self.end)

    def _build_evaluation_card(self):
        """Generate evaluation card."""
        preds_proba = self.best_model.predict_proba(self.X_valid)[:, 1]
        preds_label = (preds_proba >= 0.5).astype(int)

        cm = confusion_matrix(self.y_valid, preds_label)
        cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])

        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": self.best_model.feature_importances_
            }).sort_values("importance", ascending=False).head(15)
        elif hasattr(self.best_model, 'coef_'):
            feat_imp = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": np.abs(self.best_model.coef_[0])
            }).sort_values("importance", ascending=False).head(15)
        else:
            feat_imp = pd.DataFrame({"feature": ["N/A"], "importance": [0]})

        # Calibration plot
        fig, ax = plt.subplots(figsize=(6, 4))
        CalibrationDisplay.from_predictions(self.y_valid, preds_proba, n_bins=10, ax=ax)
        ax.set_title("Calibration Plot")
        plt.tight_layout()

        # Build card
        current.card.append(Markdown("# NBA Model V2 - Evaluation Report"))
        current.card.append(Markdown(f"**Target:** `{self.target}`"))
        current.card.append(Markdown(f"**Best Model:** `{self.best_params['model']}`"))
        current.card.append(Markdown(f"**Params:** `{self.best_params}`"))

        current.card.append(Markdown("## Performance Metrics"))
        current.card.append(
            Table([
                ["Accuracy", f"{self.best_accuracy:.4f}"],
                ["AUC", f"{self.best_auc:.4f}"],
                ["Log Loss", f"{self.best_logloss:.4f}"],
                ["Brier Score", f"{self.best_brier:.4f}"],
            ], headers=["Metric", "Value"])
        )

        current.card.append(Markdown("## All Configurations (sorted by AUC)"))
        config_rows = []
        for r in self.tuning_results[:10]:
            config_rows.append([
                r["params"]["model"],
                str({k: v for k, v in r["params"].items() if k != "model"}),
                f"{r['accuracy']:.4f}",
                f"{r['auc']:.4f}",
            ])
        current.card.append(Table(config_rows, headers=["Model", "Params", "Accuracy", "AUC"]))

        current.card.append(Markdown("## Confusion Matrix"))
        current.card.append(Table.from_dataframe(cm_df))

        current.card.append(Markdown("## Top Feature Importance"))
        current.card.append(Table.from_dataframe(feat_imp))

        current.card.append(Markdown("## Calibration"))
        current.card.append(Image.from_matplotlib(fig))

        plt.close(fig)

    @step
    def end(self):
        """Final summary."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Target: {self.target}")
        print(f"Best model: {self.best_params['model']}")
        print(f"Best AUC: {self.best_auc:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Features used: {len(self.feature_columns)}")
        print("=" * 60)


if __name__ == "__main__":
    TrainNBAModelFlow()
