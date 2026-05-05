#!/usr/bin/env python
"""
Next Game Predictor - Deep Analysis

Incorporates:
- Player performance trends (last 3 games)
- Rest days differential
- Travel fatigue factors
- Home court advantage
- Recent team form
- Star player momentum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from metaflow import Flow
from nba_api.stats.endpoints import teamgamelog, playergamelog
from nba_api.stats.static import teams, players
import time

from src.io_utils import load_game_data
from src.feature_engineering import build_enhanced_features


# Team IDs
TEAM_ABBRS = ['MIN', 'SAS', 'OKC', 'LAL', 'NYK', 'PHI', 'CLE', 'DET']
NBA_TEAMS = teams.get_teams()
TEAM_IDS = {t['abbreviation']: t['id'] for t in NBA_TEAMS if t['abbreviation'] in TEAM_ABBRS}
TEAM_CITIES = {t['abbreviation']: t['city'] for t in NBA_TEAMS if t['abbreviation'] in TEAM_ABBRS}

# Key players per team
KEY_PLAYERS = {
    'MIN': ['Julius Randle', 'Jaden McDaniels', 'Ayo Dosunmu'],
    'SAS': ['Victor Wembanyama', 'Stephon Castle', "De'Aaron Fox"],
    'OKC': ['Shai Gilgeous-Alexander', 'Chet Holmgren', 'Jalen Williams'],
    'LAL': ['LeBron James', 'Rui Hachimura', 'Marcus Smart'],
    'NYK': ['Jalen Brunson', 'Karl-Anthony Towns', 'OG Anunoby'],
    'PHI': ['Tyrese Maxey', 'Joel Embiid', 'Paul George'],
    'CLE': ['Donovan Mitchell', 'Evan Mobley', 'James Harden'],
    'DET': ['Cade Cunningham', 'Jalen Duren', 'Tobias Harris'],
}

# Travel distances (approximate miles between arenas)
TRAVEL_DISTANCES = {
    ('MIN', 'SAS'): 1200,
    ('SAS', 'MIN'): 1200,
    ('OKC', 'LAL'): 1300,
    ('LAL', 'OKC'): 1300,
    ('NYK', 'PHI'): 95,
    ('PHI', 'NYK'): 95,
    ('CLE', 'DET'): 170,
    ('DET', 'CLE'): 170,
}


def get_player_ids():
    """Get NBA API player IDs for key players."""
    all_players = players.get_players()
    player_ids = {}
    for name in sum(KEY_PLAYERS.values(), []):
        for p in all_players:
            if p['full_name'] == name:
                player_ids[name] = p['id']
                break
    return player_ids


def get_team_schedule(team: str) -> pd.DataFrame:
    """Get recent playoff games for team."""
    tid = TEAM_IDS[team]
    log = teamgamelog.TeamGameLog(team_id=tid, season='2025-26', season_type_all_star='Playoffs')
    return log.get_data_frames()[0]


def get_player_last3(player_name: str, player_ids: dict) -> dict:
    """Get player's last 3 playoff game averages."""
    if player_name not in player_ids:
        return None

    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_ids[player_name],
            season='2025-26',
            season_type_all_star='Playoffs'
        )
        df = log.get_data_frames()[0]
        if len(df) == 0:
            return None

        last3 = df.head(3)
        return {
            'pts': last3['PTS'].mean(),
            'reb': last3['REB'].mean(),
            'ast': last3['AST'].mean(),
            'fg_pct': last3['FG_PCT'].mean(),
            'plus_minus': last3['PLUS_MINUS'].mean() if 'PLUS_MINUS' in last3 else 0,
            'last_pts': df.iloc[0]['PTS'],
            'trend': 'hot' if df.iloc[0]['PTS'] > last3['PTS'].mean() else 'cold' if df.iloc[0]['PTS'] < last3['PTS'].mean() * 0.8 else 'steady'
        }
    except:
        return None


def analyze_team_factors(team: str, opponent: str, is_home: bool, player_ids: dict) -> dict:
    """Analyze all factors for a team in upcoming game."""

    sched = get_team_schedule(team)
    time.sleep(0.5)

    last_game = sched.iloc[0]
    last_date = datetime.strptime(last_game['GAME_DATE'], '%b %d, %Y')
    today = datetime(2026, 5, 5)
    rest_days = (today - last_date).days

    # Recent form
    last5 = sched.head(5)
    wins = sum(last5['WL'] == 'W')
    recent_form = wins / 5

    # Point differential last 5
    pts_for = last5['PTS'].mean()

    # Travel factor
    was_road = '@' in last_game['MATCHUP']
    travel_distance = TRAVEL_DISTANCES.get((team, opponent), 500)
    travel_fatigue = 0
    if was_road and not is_home:
        travel_fatigue = min(travel_distance / 1000, 0.5)

    # Star player analysis
    star_momentum = 0
    player_details = []
    for name in KEY_PLAYERS[team][:3]:
        stats = get_player_last3(name, player_ids)
        time.sleep(0.4)
        if stats:
            player_details.append({
                'name': name,
                'avg_pts': stats['pts'],
                'last_pts': stats['last_pts'],
                'fg_pct': stats['fg_pct'],
                'trend': stats['trend']
            })
            if stats['trend'] == 'hot':
                star_momentum += 0.05
            elif stats['trend'] == 'cold':
                star_momentum -= 0.05

    return {
        'team': team,
        'rest_days': rest_days,
        'recent_form': recent_form,
        'pts_avg': pts_for,
        'travel_fatigue': travel_fatigue,
        'star_momentum': star_momentum,
        'is_home': is_home,
        'players': player_details,
        'last_result': f"{last_game['WL']} {last_game['PTS']} pts"
    }


def calculate_win_probability(home_factors: dict, away_factors: dict, base_model_prob: float) -> float:
    """Calculate adjusted win probability incorporating all factors."""

    # Start with model baseline
    prob = base_model_prob

    # Rest advantage (max +/- 5%)
    rest_diff = home_factors['rest_days'] - away_factors['rest_days']
    rest_adj = np.clip(rest_diff * 0.01, -0.05, 0.05)
    prob += rest_adj

    # Recent form adjustment (max +/- 5%)
    form_diff = home_factors['recent_form'] - away_factors['recent_form']
    form_adj = form_diff * 0.1
    prob += np.clip(form_adj, -0.05, 0.05)

    # Travel fatigue (only affects away team, max -3%)
    travel_adj = -away_factors['travel_fatigue'] * 0.06
    prob += travel_adj

    # Star momentum (max +/- 3%)
    momentum_diff = home_factors['star_momentum'] - away_factors['star_momentum']
    prob += np.clip(momentum_diff, -0.03, 0.03)

    return np.clip(prob, 0.1, 0.9)


def load_base_model():
    """Load trained model for baseline probabilities."""
    train_flow = Flow("TrainNBAModelFlow")
    run = train_flow.latest_successful_run
    model = run["join_tuning"].task.data.best_model
    feature_columns = run["feature_engineering"].task.data.feature_columns
    historical_df = load_game_data("data/processed/nba_games_enhanced.csv")
    return model, feature_columns, historical_df


def get_base_home_prob(team: str, model, feature_columns, historical_df) -> float:
    """Get model's base home win probability for team."""
    featured = build_enhanced_features(historical_df)
    team_games = featured[featured["home_team"] == team].sort_values("game_date", ascending=False)
    if len(team_games) == 0:
        return 0.55
    features = team_games.iloc[0][feature_columns].values.reshape(1, -1)
    return model.predict_proba(features)[0, 1]


def main():
    print("Loading model and data...")
    model, feature_columns, historical_df = load_base_model()
    player_ids = get_player_ids()

    print("\n" + "="*80)
    print("NEXT GAME PREDICTIONS - MAY 5, 2026")
    print("="*80)

    # Current series matchups with next game location
    games = [
        {
            'series': 'Western Semis G2',
            'higher': 'MIN', 'lower': 'SAS',
            'state': 'MIN leads 1-0',
            'next_home': 'SAS',
            'game_num': 2
        },
        {
            'series': 'Western Semis G1',
            'higher': 'OKC', 'lower': 'LAL',
            'state': '0-0',
            'next_home': 'OKC',
            'game_num': 1
        },
        {
            'series': 'Eastern Semis G2',
            'higher': 'NYK', 'lower': 'PHI',
            'state': 'NYK leads 1-0',
            'next_home': 'NYK',
            'game_num': 2
        },
        {
            'series': 'Eastern Semis G1',
            'higher': 'CLE', 'lower': 'DET',
            'state': '0-0',
            'next_home': 'CLE',
            'game_num': 1
        },
    ]

    results = []

    for game in games:
        higher, lower = game['higher'], game['lower']
        home_team = game['next_home']
        away_team = lower if home_team == higher else higher

        print(f"\n{'='*80}")
        print(f"{game['series']}: {away_team} @ {home_team}")
        print(f"Series: {game['state']}")
        print("="*80)

        # Analyze both teams
        print(f"\nAnalyzing {home_team} (HOME)...")
        home_factors = analyze_team_factors(home_team, away_team, True, player_ids)

        print(f"Analyzing {away_team} (AWAY)...")
        away_factors = analyze_team_factors(away_team, home_team, False, player_ids)

        # Get baseline model probability
        base_prob = get_base_home_prob(home_team, model, feature_columns, historical_df)

        # Calculate adjusted probability
        final_prob = calculate_win_probability(home_factors, away_factors, base_prob)

        # Display analysis
        print(f"\n--- {home_team} (HOME) ---")
        print(f"  Rest: {home_factors['rest_days']} days")
        print(f"  Recent form: {home_factors['recent_form']:.0%} (last 5)")
        print(f"  Scoring avg: {home_factors['pts_avg']:.1f}")
        print(f"  Last game: {home_factors['last_result']}")
        print(f"  Key players:")
        for p in home_factors['players']:
            trend_icon = "🔥" if p['trend'] == 'hot' else "❄️" if p['trend'] == 'cold' else "➖"
            print(f"    {p['name']}: {p['avg_pts']:.1f} ppg ({p['fg_pct']:.1%} FG) {trend_icon}")

        print(f"\n--- {away_team} (AWAY) ---")
        print(f"  Rest: {away_factors['rest_days']} days")
        print(f"  Recent form: {away_factors['recent_form']:.0%} (last 5)")
        print(f"  Scoring avg: {away_factors['pts_avg']:.1f}")
        print(f"  Travel fatigue: {away_factors['travel_fatigue']:.2f}")
        print(f"  Last game: {away_factors['last_result']}")
        print(f"  Key players:")
        for p in away_factors['players']:
            trend_icon = "🔥" if p['trend'] == 'hot' else "❄️" if p['trend'] == 'cold' else "➖"
            print(f"    {p['name']}: {p['avg_pts']:.1f} ppg ({p['fg_pct']:.1%} FG) {trend_icon}")

        # Prediction
        print(f"\n{'='*40}")
        print("PREDICTION")
        print("="*40)
        print(f"  Model baseline (home): {base_prob:.1%}")
        print(f"  Adjusted probability:")
        print(f"    {home_team}: {final_prob:.1%}")
        print(f"    {away_team}: {1-final_prob:.1%}")

        predicted_winner = home_team if final_prob > 0.5 else away_team
        confidence = max(final_prob, 1-final_prob)

        print(f"\n  → PICK: {predicted_winner} ({confidence:.1%})")

        # Factors breakdown
        rest_diff = home_factors['rest_days'] - away_factors['rest_days']
        form_diff = home_factors['recent_form'] - away_factors['recent_form']

        print(f"\n  Key factors:")
        if abs(rest_diff) >= 2:
            adv = home_team if rest_diff > 0 else away_team
            print(f"    • Rest advantage: {adv} (+{abs(rest_diff)} days)")
        if abs(form_diff) >= 0.2:
            adv = home_team if form_diff > 0 else away_team
            print(f"    • Hot streak: {adv}")
        if away_factors['travel_fatigue'] > 0.2:
            print(f"    • Travel fatigue affects {away_team}")

        results.append({
            'game': f"{away_team} @ {home_team}",
            'series': game['series'],
            'home_team': home_team,
            'away_team': away_team,
            'home_prob': final_prob,
            'away_prob': 1 - final_prob,
            'pick': predicted_winner,
            'confidence': confidence,
            'home_rest': home_factors['rest_days'],
            'away_rest': away_factors['rest_days'],
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL PICKS")
    print("="*80)

    for r in results:
        print(f"  {r['series']}: {r['pick']} ({r['confidence']:.0%})")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results/next_game_predictions.csv", index=False)
    print(f"\nSaved to results/next_game_predictions.csv")

    return results


if __name__ == "__main__":
    main()
