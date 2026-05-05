"""
Advanced Features for NBA Prediction

Computes proxy features for:
- Player availability (from games played patterns)
- Implied team strength (from season performance)
- Situational factors (travel, rest, schedule)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def add_player_availability_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add proxy for player availability based on roster stability.

    Uses games played patterns to estimate if team is at full strength.
    Teams with consistent lineups tend to perform better.
    """
    df = df.copy()

    # This would require player-level data
    # For now, add placeholder that could be populated
    df['home_roster_stability'] = 1.0
    df['away_roster_stability'] = 1.0

    return df


def add_implied_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add implied team strength based on multiple factors.

    Combines:
    - Point differential (best predictor)
    - Win percentage
    - Strength of schedule adjustment
    """
    df = df.copy()

    # Calculate rolling point differential (last 10 games)
    if 'home_points_rolling_10' in df.columns and 'away_points_rolling_10' in df.columns:
        # Assume opponent allows ~110 points on average
        avg_allowed = 110

        df['home_net_rating_proxy'] = (df['home_points_rolling_10'] - avg_allowed) / 10
        df['away_net_rating_proxy'] = (df['away_points_rolling_10'] - avg_allowed) / 10
        df['net_rating_diff'] = df['home_net_rating_proxy'] - df['away_net_rating_proxy']

    return df


def add_schedule_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule difficulty features.

    Factors:
    - Back-to-back games
    - 3 games in 4 nights
    - Long road trips
    """
    df = df.copy()

    # Back-to-back detection (rest_days == 1)
    if 'home_rest_days' in df.columns:
        df['home_back_to_back'] = (df['home_rest_days'] == 1).astype(int)
        df['away_back_to_back'] = (df['away_rest_days'] == 1).astype(int)
        df['back_to_back_diff'] = df['away_back_to_back'] - df['home_back_to_back']

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and form features.

    - Win streak strength (weighted recent wins)
    - Scoring trend (increasing/decreasing)
    - Consistency (variance in recent scores)
    """
    df = df.copy()

    # Streak interaction with rest
    if 'home_streak' in df.columns and 'home_rest_days' in df.columns:
        # Rested team coming off wins is dangerous
        df['home_rested_winner'] = (df['home_streak'] > 0) & (df['home_rest_days'] >= 3)
        df['away_rested_winner'] = (df['away_streak'] > 0) & (df['away_rest_days'] >= 3)

        df['home_rested_winner'] = df['home_rested_winner'].astype(int)
        df['away_rested_winner'] = df['away_rested_winner'].astype(int)

    return df


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add head-to-head and stylistic matchup features.

    - Pace differential (fast vs slow teams)
    - Efficiency matchup
    """
    df = df.copy()

    # Pace proxy from points scored
    if 'home_points_rolling_5' in df.columns:
        league_avg_pts = 115  # Approximate

        df['home_pace_factor'] = df['home_points_rolling_5'] / league_avg_pts
        df['away_pace_factor'] = df['away_points_rolling_5'] / league_avg_pts
        df['pace_mismatch'] = abs(df['home_pace_factor'] - df['away_pace_factor'])

    return df


def add_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all advanced feature engineering."""

    df = add_implied_team_strength(df)
    df = add_schedule_difficulty(df)
    df = add_momentum_features(df)
    df = add_matchup_features(df)

    return df


def get_advanced_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of advanced feature columns."""

    advanced_cols = [
        'home_net_rating_proxy', 'away_net_rating_proxy', 'net_rating_diff',
        'home_back_to_back', 'away_back_to_back', 'back_to_back_diff',
        'home_rested_winner', 'away_rested_winner',
        'home_pace_factor', 'away_pace_factor', 'pace_mismatch',
    ]

    return [c for c in advanced_cols if c in df.columns]


def compute_implied_spread(home_win_prob: float) -> float:
    """
    Convert win probability to implied spread.

    Based on historical relationship between spread and win probability.
    Rough approximation: each 1 point of spread ≈ 3% win probability change
    """
    # Home team at 50% = spread of about -1 (slight home advantage)
    # Relationship is roughly linear in the middle
    base_prob = 0.50
    points_per_pct = 1 / 0.03  # ~33 points per 100%

    prob_diff = home_win_prob - base_prob
    implied_spread = -prob_diff * points_per_pct - 1  # -1 for home advantage

    return round(implied_spread, 1)


if __name__ == "__main__":
    # Test with sample data
    from io_utils import load_game_data
    from feature_engineering import build_enhanced_features

    print("Loading data...")
    df = load_game_data("data/processed/nba_games_enhanced.csv")
    df = build_enhanced_features(df)

    print("Adding advanced features...")
    df = add_all_advanced_features(df)

    new_cols = get_advanced_feature_columns(df)
    print(f"\nAdded {len(new_cols)} advanced features:")
    for col in new_cols:
        print(f"  - {col}")

    print(f"\nSample values:")
    print(df[new_cols].describe())
