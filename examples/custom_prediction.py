#!/usr/bin/env python
"""
Custom Prediction Example

Shows how to predict a specific matchup with custom parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from metaflow import Flow

from src.io_utils import load_game_data
from src.feature_engineering import build_enhanced_features


def simulate_series(prob_a_home: float, prob_b_home: float, a_wins: int = 0,
                    b_wins: int = 0, n_sims: int = 50000) -> dict:
    """Simulate a playoff series from current state."""
    home_schedule = ["A", "A", "B", "B", "A", "B", "A"]
    remaining = home_schedule[a_wins + b_wins:]

    a_series_wins = 0
    for _ in range(n_sims):
        a_w, b_w = a_wins, b_wins
        for home in remaining:
            if a_w >= 4 or b_w >= 4:
                break
            prob = prob_a_home if home == "A" else (1 - prob_b_home)
            if np.random.random() < prob:
                a_w += 1
            else:
                b_w += 1
        if a_w >= 4:
            a_series_wins += 1

    return {
        "prob_a": a_series_wins / n_sims,
        "prob_b": 1 - a_series_wins / n_sims,
    }


def get_team_home_prob(team: str, model, feature_columns, featured_df) -> float:
    """Get a team's predicted home win probability."""
    team_games = featured_df[featured_df["home_team"] == team].sort_values(
        "game_date", ascending=False
    )
    if len(team_games) == 0:
        return 0.55  # Default if no data
    features = team_games.iloc[0][feature_columns].values.reshape(1, -1)
    return model.predict_proba(features)[0, 1]


def main():
    parser = argparse.ArgumentParser(description="Predict a custom NBA matchup")
    parser.add_argument("team_a", help="Higher seed team (has home court)")
    parser.add_argument("team_b", help="Lower seed team")
    parser.add_argument("--a-wins", type=int, default=0, help="Team A current wins")
    parser.add_argument("--b-wins", type=int, default=0, help="Team B current wins")
    args = parser.parse_args()

    team_a = args.team_a.upper()
    team_b = args.team_b.upper()

    print(f"\nPredicting: {team_a} vs {team_b}")
    if args.a_wins or args.b_wins:
        print(f"Current series: {team_a} {args.a_wins}-{args.b_wins} {team_b}")
    print("-" * 40)

    # Load model
    try:
        train_flow = Flow("TrainNBAModelFlow")
        run = train_flow.latest_successful_run
        model = run["join_tuning"].task.data.best_model
        feature_columns = run["feature_engineering"].task.data.feature_columns
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Run 'python flows/train_model.py run' first.")
        return

    # Load and prepare data
    df = load_game_data("data/processed/nba_games_enhanced.csv")
    featured = build_enhanced_features(df)

    # Get probabilities
    prob_a = get_team_home_prob(team_a, model, feature_columns, featured)
    prob_b = get_team_home_prob(team_b, model, feature_columns, featured)

    print(f"\n{team_a} home win prob: {prob_a:.1%}")
    print(f"{team_b} home win prob: {prob_b:.1%}")

    # Simulate series
    result = simulate_series(prob_a, prob_b, args.a_wins, args.b_wins)

    print(f"\nSeries prediction:")
    print(f"  {team_a}: {result['prob_a']:.1%}")
    print(f"  {team_b}: {result['prob_b']:.1%}")

    winner = team_a if result["prob_a"] > 0.5 else team_b
    confidence = max(result["prob_a"], result["prob_b"])
    print(f"\nPredicted winner: {winner} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
