#!/usr/bin/env python
"""
Quickstart Example - NBA Playoff Predictor

This script demonstrates the basic workflow:
1. Load a trained model
2. Get team win probabilities
3. Simulate a playoff series
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metaflow import Flow
import numpy as np

from src.io_utils import load_game_data
from src.feature_engineering import build_enhanced_features


def main():
    print("NBA Playoff Predictor - Quickstart")
    print("=" * 50)

    # Step 1: Load the trained model
    print("\n1. Loading trained model...")
    try:
        train_flow = Flow("TrainNBAModelFlow")
        run = train_flow.latest_successful_run
        model = run["join_tuning"].task.data.best_model
        feature_columns = run["feature_engineering"].task.data.feature_columns
        auc = run["join_tuning"].task.data.best_auc
        print(f"   Model loaded (AUC: {auc:.4f})")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Run 'python flows/train_model.py run' first to train a model.")
        return

    # Step 2: Load historical data
    print("\n2. Loading historical data...")
    try:
        df = load_game_data("data/processed/nba_games_enhanced.csv")
        print(f"   Loaded {len(df)} games")
    except FileNotFoundError:
        print("   Data not found. Run 'python src/data_acquisition.py' first.")
        return

    # Step 3: Get team win probability
    print("\n3. Getting team home win probability...")
    team = "OKC"
    featured = build_enhanced_features(df)
    team_games = featured[featured["home_team"] == team].sort_values("game_date", ascending=False)

    if len(team_games) > 0:
        features = team_games.iloc[0][feature_columns].values.reshape(1, -1)
        prob = model.predict_proba(features)[0, 1]
        print(f"   {team} home win probability: {prob:.1%}")

    # Step 4: Simulate a series
    print("\n4. Simulating a playoff series...")
    team_a, team_b = "OKC", "LAL"

    # Get both teams' home win probabilities
    team_a_games = featured[featured["home_team"] == team_a].sort_values("game_date", ascending=False)
    team_b_games = featured[featured["home_team"] == team_b].sort_values("game_date", ascending=False)

    prob_a = model.predict_proba(team_a_games.iloc[0][feature_columns].values.reshape(1, -1))[0, 1]
    prob_b = model.predict_proba(team_b_games.iloc[0][feature_columns].values.reshape(1, -1))[0, 1]

    print(f"   {team_a} home win prob: {prob_a:.1%}")
    print(f"   {team_b} home win prob: {prob_b:.1%}")

    # Monte Carlo simulation
    n_sims = 10000
    a_wins_series = 0
    home_schedule = ["A", "A", "B", "B", "A", "B", "A"]  # 2-2-1-1-1 format

    for _ in range(n_sims):
        a_wins, b_wins = 0, 0
        for home in home_schedule:
            if a_wins >= 4 or b_wins >= 4:
                break
            if home == "A":
                win_prob = prob_a
            else:
                win_prob = 1 - prob_b
            if np.random.random() < win_prob:
                a_wins += 1
            else:
                b_wins += 1
        if a_wins >= 4:
            a_wins_series += 1

    series_prob = a_wins_series / n_sims
    print(f"\n   Series prediction ({team_a} vs {team_b}):")
    print(f"   {team_a}: {series_prob:.1%}")
    print(f"   {team_b}: {1-series_prob:.1%}")

    winner = team_a if series_prob > 0.5 else team_b
    print(f"\n   Predicted winner: {winner}")

    print("\n" + "=" * 50)
    print("Done! See scripts/ for more detailed predictions.")


if __name__ == "__main__":
    main()
