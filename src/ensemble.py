"""
Ensemble Model for NBA Prediction

Combines predictions from multiple model types for improved accuracy.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Tuple


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that combines multiple models.

    Supports:
    - Simple averaging
    - Weighted averaging (by AUC)
    - Stacking (meta-learner)
    """

    def __init__(self, models: List[Tuple], method: str = "weighted"):
        """
        Args:
            models: List of (model, weight) tuples
            method: 'simple', 'weighted', or 'voting'
        """
        self.models = models
        self.method = method
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """Models are pre-fitted, this is a no-op."""
        return self

    def predict_proba(self, X):
        """Combine predictions from all models."""
        probas = []
        weights = []

        for model, weight in self.models:
            prob = model.predict_proba(X)[:, 1]
            probas.append(prob)
            weights.append(weight)

        probas = np.array(probas)
        weights = np.array(weights)

        if self.method == "simple":
            combined = np.mean(probas, axis=0)
        elif self.method == "weighted":
            weights = weights / weights.sum()
            combined = np.average(probas, axis=0, weights=weights)
        elif self.method == "voting":
            votes = (probas > 0.5).astype(int)
            combined = np.mean(votes, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return np.column_stack([1 - combined, combined])

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def build_ensemble_from_tuning_results(tuning_results: List[dict],
                                        top_n_per_type: int = 1) -> EnsembleClassifier:
    """
    Build ensemble from Metaflow tuning results.

    Selects the best model of each type to ensure diversity.

    Args:
        tuning_results: List of dicts with 'params', 'model', 'auc' keys
        top_n_per_type: How many of each model type to include

    Returns:
        EnsembleClassifier with diverse models
    """
    # Group by model type
    by_type = {}
    for r in tuning_results:
        model_type = r['params']['model']
        if model_type not in by_type:
            by_type[model_type] = []
        by_type[model_type].append(r)

    # Select top N from each type
    selected = []
    for model_type, results in by_type.items():
        # Sort by AUC descending
        sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
        for r in sorted_results[:top_n_per_type]:
            selected.append((r['model'], r['auc']))
            print(f"  Including {model_type}: AUC={r['auc']:.4f}")

    return EnsembleClassifier(selected, method="weighted")


def calibrate_model(model, X_cal, y_cal, method='isotonic'):
    """
    Calibrate model probabilities.

    Args:
        model: Fitted classifier
        X_cal: Calibration features
        y_cal: Calibration labels
        method: 'isotonic' or 'sigmoid'

    Returns:
        CalibratedClassifierCV
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    return calibrated
