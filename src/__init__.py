"""
NBA Playoff Predictor - Core Library

Modules:
    data_acquisition: Fetch game data from NBA Stats API
    feature_engineering: Build ML features from game data
    io_utils: Data loading and saving utilities
    series_simulation: Monte Carlo playoff series simulation
"""

__version__ = "1.0.0"

from .data_acquisition import fetch_nba_games
from .feature_engineering import build_enhanced_features, get_enhanced_feature_columns
from .io_utils import load_game_data, save_predictions

__all__ = [
    "fetch_nba_games",
    "build_enhanced_features",
    "get_enhanced_feature_columns",
    "load_game_data",
    "save_predictions",
]
