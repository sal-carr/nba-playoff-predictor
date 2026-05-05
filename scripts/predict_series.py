#!/usr/bin/env python
"""
Current 2025-26 NBA Playoff Predictions

Based on actual playoff data through May 4, 2026.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from metaflow import Flow

from src.io_utils import load_game_data
from src.feature_engineering import build_enhanced_features, get_enhanced_feature_columns


def load_model_and_data():
    """Load model and data."""
    print("Loading model...")
    train_flow = Flow("TrainNBAModelFlow")
    run = train_flow.latest_successful_run

    model = run["join_tuning"].task.data.best_model
    feature_columns = run["feature_engineering"].task.data.feature_columns

    print(f"  Run: {run.id}, AUC: {run['join_tuning'].task.data.best_auc:.4f}")

    historical_df = load_game_data("data/processed/nba_games_enhanced.csv")
    print(f"  Data: {len(historical_df)} games through {historical_df['game_date'].max().date()}")

    return model, feature_columns, historical_df


def get_team_home_win_prob(team: str, model, feature_columns, historical_df):
    """Get predicted home win probability for a team."""
    featured = build_enhanced_features(historical_df)
    team_games = featured[featured["home_team"] == team].sort_values("game_date", ascending=False)

    if len(team_games) == 0:
        return 0.5

    features = team_games.iloc[0][feature_columns].values.reshape(1, -1)
    return model.predict_proba(features)[0, 1]


def simulate_series_from_state(
    team_a: str,
    team_b: str,
    a_wins: int,
    b_wins: int,
    home_prob_a: float,
    home_prob_b: float,
    wins_needed: int = 4,
    n_sims: int = 100000
) -> dict:
    """
    Simulate remainder of series from current state.

    team_a has home court advantage.
    """
    # Remaining home schedule (2-2-1-1-1 format)
    # Games 1,2 at A, 3,4 at B, 5 at A, 6 at B, 7 at A
    full_schedule = ["A", "A", "B", "B", "A", "B", "A"]
    games_played = a_wins + b_wins
    remaining_schedule = full_schedule[games_played:]

    a_wins_series = 0
    outcomes = {}

    for _ in range(n_sims):
        a_w, b_w = a_wins, b_wins

        for home in remaining_schedule:
            if a_w >= wins_needed or b_w >= wins_needed:
                break

            if home == "A":
                prob_a_wins = home_prob_a
            else:
                prob_a_wins = 1 - home_prob_b

            if np.random.random() < prob_a_wins:
                a_w += 1
            else:
                b_w += 1

        if a_w >= wins_needed:
            a_wins_series += 1
            outcome = f"{team_a} in {a_w + b_w}"
        else:
            outcome = f"{team_b} in {a_w + b_w}"

        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    return {
        "team_a_prob": a_wins_series / n_sims,
        "team_b_prob": 1 - a_wins_series / n_sims,
        "outcomes": {k: v/n_sims for k, v in sorted(outcomes.items())}
    }


def main():
    model, feature_columns, historical_df = load_model_and_data()

    print("\n" + "="*70)
    print("2025-26 NBA PLAYOFF PREDICTIONS")
    print("As of May 5, 2026")
    print("="*70)

    # Current series states (from data analysis)
    # Second Round matchups:

    series = {
        "Western Conference Semifinals": [
            {"higher": "MIN", "lower": "SAS", "h_wins": 1, "l_wins": 0},  # MIN leads 1-0
            {"higher": "OKC", "lower": "LAL", "h_wins": 0, "l_wins": 0},  # Not started
        ],
        "Eastern Conference Semifinals": [
            {"higher": "NYK", "lower": "PHI", "h_wins": 1, "l_wins": 0},  # NYK leads 1-0
            {"higher": "CLE", "lower": "DET", "h_wins": 0, "l_wins": 0},  # Not started
        ]
    }

    results = []
    conf_winners = {"West": [], "East": []}

    for conf_name, matchups in series.items():
        print(f"\n{'='*70}")
        print(conf_name.upper())
        print("="*70)

        conf = "West" if "Western" in conf_name else "East"

        for s in matchups:
            higher, lower = s["higher"], s["lower"]
            h_wins, l_wins = s["h_wins"], s["l_wins"]

            print(f"\n{higher} vs {lower} (Current: {higher} {h_wins}-{l_wins} {lower})")

            # Get home win probabilities
            h_home_prob = get_team_home_win_prob(higher, model, feature_columns, historical_df)
            l_home_prob = get_team_home_win_prob(lower, model, feature_columns, historical_df)

            print(f"  {higher} home win prob: {h_home_prob:.1%}")
            print(f"  {lower} home win prob: {l_home_prob:.1%}")

            # Simulate from current state
            result = simulate_series_from_state(
                higher, lower, h_wins, l_wins,
                h_home_prob, l_home_prob
            )

            predicted_winner = higher if result["team_a_prob"] > 0.5 else lower
            confidence = max(result["team_a_prob"], result["team_b_prob"])

            print(f"\n  PREDICTION:")
            print(f"  {higher} wins series: {result['team_a_prob']:.1%}")
            print(f"  {lower} wins series: {result['team_b_prob']:.1%}")
            print(f"  → {predicted_winner} advances ({confidence:.1%})")

            results.append({
                "round": conf_name,
                "higher_seed": higher,
                "lower_seed": lower,
                "current_state": f"{h_wins}-{l_wins}",
                "higher_prob": result["team_a_prob"],
                "lower_prob": result["team_b_prob"],
                "predicted_winner": predicted_winner,
                "confidence": confidence
            })

            conf_winners[conf].append(predicted_winner)

    # Conference Finals predictions
    print(f"\n{'='*70}")
    print("CONFERENCE FINALS PREDICTIONS")
    print("="*70)

    for conf in ["West", "East"]:
        if len(conf_winners[conf]) == 2:
            t1, t2 = conf_winners[conf]
            print(f"\n{conf}ern Conference Finals: {t1} vs {t2}")

            p1 = get_team_home_win_prob(t1, model, feature_columns, historical_df)
            p2 = get_team_home_win_prob(t2, model, feature_columns, historical_df)

            result = simulate_series_from_state(t1, t2, 0, 0, p1, p2)

            winner = t1 if result["team_a_prob"] > 0.5 else t2
            conf_winners[conf] = [winner]

            print(f"  {t1}: {result['team_a_prob']:.1%}")
            print(f"  {t2}: {result['team_b_prob']:.1%}")
            print(f"  → {conf} Champion: {winner}")

            results.append({
                "round": f"{conf}ern Conference Finals",
                "higher_seed": t1,
                "lower_seed": t2,
                "current_state": "0-0",
                "higher_prob": result["team_a_prob"],
                "lower_prob": result["team_b_prob"],
                "predicted_winner": winner,
                "confidence": max(result["team_a_prob"], result["team_b_prob"])
            })

    # NBA Finals
    west_champ = conf_winners["West"][0]
    east_champ = conf_winners["East"][0]

    print(f"\n{'='*70}")
    print("NBA FINALS PREDICTION")
    print("="*70)
    print(f"\n{west_champ} (West) vs {east_champ} (East)")

    p_west = get_team_home_win_prob(west_champ, model, feature_columns, historical_df)
    p_east = get_team_home_win_prob(east_champ, model, feature_columns, historical_df)

    finals_result = simulate_series_from_state(west_champ, east_champ, 0, 0, p_west, p_east)

    champion = west_champ if finals_result["team_a_prob"] > 0.5 else east_champ
    champ_prob = max(finals_result["team_a_prob"], finals_result["team_b_prob"])

    print(f"  {west_champ}: {finals_result['team_a_prob']:.1%}")
    print(f"  {east_champ}: {finals_result['team_b_prob']:.1%}")

    print(f"\n{'='*70}")
    print(f"🏆 PREDICTED 2026 NBA CHAMPION: {champion}")
    print(f"   Confidence: {champ_prob:.1%}")
    print("="*70)

    results.append({
        "round": "NBA Finals",
        "higher_seed": west_champ,
        "lower_seed": east_champ,
        "current_state": "0-0",
        "higher_prob": finals_result["team_a_prob"],
        "lower_prob": finals_result["team_b_prob"],
        "predicted_winner": champion,
        "confidence": champ_prob
    })

    # Save
    df = pd.DataFrame(results)
    df.to_csv("results/playoff_predictions_current.csv", index=False)
    print(f"\nSaved to results/playoff_predictions_current.csv")

    return df


if __name__ == "__main__":
    main()
