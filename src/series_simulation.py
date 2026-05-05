"""
Series Simulation Module

Simulates playoff series outcomes using game-level win probabilities.
Supports best-of-5 and best-of-7 formats.
"""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass


@dataclass
class SeriesPrediction:
    """Result of a series simulation."""
    team_a: str
    team_b: str
    team_a_win_prob: float
    team_b_win_prob: float
    expected_games: float
    most_likely_outcome: str
    outcome_probabilities: dict


def simulate_series(
    home_win_prob_a: float,
    home_win_prob_b: float,
    series_format: Literal["bo5", "bo7"] = "bo7",
    home_schedule: list[str] = None,
    n_simulations: int = 100000
) -> dict:
    """
    Monte Carlo simulation of a playoff series.

    Args:
        home_win_prob_a: Probability Team A wins when hosting
        home_win_prob_b: Probability Team B wins when hosting
        series_format: "bo5" (best of 5) or "bo7" (best of 7)
        home_schedule: List of which team hosts each game ["A", "B", "A", "B", ...]
                      Defaults to standard 2-2-1-1-1 format
        n_simulations: Number of Monte Carlo iterations

    Returns:
        Dictionary with simulation results
    """
    wins_needed = 3 if series_format == "bo5" else 4
    max_games = 5 if series_format == "bo5" else 7

    if home_schedule is None:
        if series_format == "bo5":
            home_schedule = ["A", "A", "B", "B", "A"]
        else:
            home_schedule = ["A", "A", "B", "B", "A", "B", "A"]

    a_wins_series = 0
    total_games = []
    outcomes = {}

    for _ in range(n_simulations):
        a_wins = 0
        b_wins = 0
        game = 0

        while a_wins < wins_needed and b_wins < wins_needed:
            home_team = home_schedule[game]

            if home_team == "A":
                win_prob = home_win_prob_a
            else:
                win_prob = 1 - home_win_prob_b

            if np.random.random() < win_prob:
                a_wins += 1
            else:
                b_wins += 1

            game += 1

        if a_wins == wins_needed:
            a_wins_series += 1
            outcome = f"A in {game}"
        else:
            outcome = f"B in {game}"

        total_games.append(game)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    outcome_probs = {k: v / n_simulations for k, v in sorted(outcomes.items())}

    return {
        "team_a_series_prob": a_wins_series / n_simulations,
        "team_b_series_prob": 1 - (a_wins_series / n_simulations),
        "expected_games": np.mean(total_games),
        "outcome_probabilities": outcome_probs
    }


def predict_series(
    team_a: str,
    team_b: str,
    model,
    feature_columns: list[str],
    historical_df: pd.DataFrame,
    series_format: Literal["bo5", "bo7"] = "bo7",
    n_simulations: int = 100000
) -> SeriesPrediction:
    """
    Predict a playoff series outcome using the trained model.

    Args:
        team_a: Higher seed / first home team
        team_b: Lower seed / second home team
        model: Trained classifier with predict_proba method
        feature_columns: List of feature column names
        historical_df: Historical game data for feature calculation
        series_format: "bo5" or "bo7"
        n_simulations: Number of Monte Carlo iterations

    Returns:
        SeriesPrediction with results
    """
    from src.feature_engineering import build_game_level_features

    # Get the most recent features for each team
    featured = build_game_level_features(historical_df)

    # Get latest game features for team A as home
    a_home_games = featured[featured["home_team"] == team_a].sort_values("game_date", ascending=False)
    # Get latest game features for team B as home
    b_home_games = featured[featured["home_team"] == team_b].sort_values("game_date", ascending=False)

    if len(a_home_games) == 0 or len(b_home_games) == 0:
        raise ValueError(f"Insufficient data for {team_a} or {team_b}")

    # Use most recent home game features
    a_features = a_home_games.iloc[0][feature_columns].values.reshape(1, -1)
    b_features = b_home_games.iloc[0][feature_columns].values.reshape(1, -1)

    # Predict home win probabilities
    home_win_prob_a = model.predict_proba(a_features)[0, 1]
    home_win_prob_b = model.predict_proba(b_features)[0, 1]

    print(f"  {team_a} home win prob: {home_win_prob_a:.1%}")
    print(f"  {team_b} home win prob: {home_win_prob_b:.1%}")

    # Simulate series
    results = simulate_series(
        home_win_prob_a=home_win_prob_a,
        home_win_prob_b=home_win_prob_b,
        series_format=series_format,
        n_simulations=n_simulations
    )

    # Find most likely outcome
    most_likely = max(results["outcome_probabilities"].items(), key=lambda x: x[1])

    return SeriesPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_win_prob=results["team_a_series_prob"],
        team_b_win_prob=results["team_b_series_prob"],
        expected_games=results["expected_games"],
        most_likely_outcome=most_likely[0],
        outcome_probabilities=results["outcome_probabilities"]
    )


def predict_bracket(
    matchups: list[tuple[str, str]],
    model,
    feature_columns: list[str],
    historical_df: pd.DataFrame,
    round_names: list[str] = None,
    series_format: Literal["bo5", "bo7"] = "bo7"
) -> pd.DataFrame:
    """
    Predict a playoff bracket recursively.

    Args:
        matchups: List of (team_a, team_b) tuples for first round
        model: Trained classifier
        feature_columns: Feature column names
        historical_df: Historical data
        round_names: Names for each round (e.g., ["First Round", "Semis", "Finals"])
        series_format: Series format

    Returns:
        DataFrame with all predictions
    """
    if round_names is None:
        n_rounds = int(np.log2(len(matchups))) + 1
        round_names = [f"Round {i+1}" for i in range(n_rounds)]

    results = []
    current_matchups = matchups
    round_idx = 0

    while len(current_matchups) >= 1:
        round_name = round_names[round_idx] if round_idx < len(round_names) else f"Round {round_idx + 1}"
        print(f"\n=== {round_name} ===")

        winners = []
        for team_a, team_b in current_matchups:
            print(f"\n{team_a} vs {team_b}:")
            pred = predict_series(
                team_a, team_b, model, feature_columns, historical_df, series_format
            )

            results.append({
                "round": round_name,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_prob": pred.team_a_win_prob,
                "team_b_prob": pred.team_b_win_prob,
                "expected_games": pred.expected_games,
                "predicted_winner": team_a if pred.team_a_win_prob > 0.5 else team_b,
                "winner_confidence": max(pred.team_a_win_prob, pred.team_b_win_prob)
            })

            print(f"  → {team_a}: {pred.team_a_win_prob:.1%}")
            print(f"  → {team_b}: {pred.team_b_win_prob:.1%}")
            print(f"  Expected games: {pred.expected_games:.1f}")

            # Winner advances
            winner = team_a if pred.team_a_win_prob > 0.5 else team_b
            winners.append(winner)

        if len(winners) == 1:
            print(f"\n🏆 Predicted Champion: {winners[0]}")
            break

        # Create next round matchups
        current_matchups = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
        round_idx += 1

    return pd.DataFrame(results)
