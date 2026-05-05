"""
NBA Injury Data Acquisition

Fetches injury reports and player availability data.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


def fetch_espn_injuries() -> List[Dict]:
    """
    Fetch current NBA injuries from ESPN API.

    Returns:
        List of injury records with player, team, status, type
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Warning: Could not fetch ESPN injuries: {e}")
        return []

    injuries = []
    for team_data in data.get('teams', []):
        team_abbr = team_data.get('team', {}).get('abbreviation', '')
        team_name = team_data.get('team', {}).get('displayName', '')

        for injury in team_data.get('injuries', []):
            athlete = injury.get('athlete', {})
            injuries.append({
                'player_id': athlete.get('id', ''),
                'player_name': athlete.get('displayName', ''),
                'team': team_abbr,
                'team_name': team_name,
                'position': athlete.get('position', {}).get('abbreviation', ''),
                'status': injury.get('status', 'Unknown'),
                'injury_type': injury.get('type', {}).get('description', ''),
                'details': injury.get('details', {}).get('detail', ''),
                'fetch_date': datetime.now().date().isoformat()
            })

    return injuries


def get_team_injury_impact(team: str, injuries: List[Dict]) -> Dict:
    """
    Calculate injury impact score for a team.

    Args:
        team: Team abbreviation
        injuries: List of injury records

    Returns:
        Dict with injury metrics
    """
    team_injuries = [i for i in injuries if i['team'] == team]

    # Count by status
    out_count = sum(1 for i in team_injuries if i['status'].lower() == 'out')
    doubtful_count = sum(1 for i in team_injuries if i['status'].lower() == 'doubtful')
    questionable_count = sum(1 for i in team_injuries if i['status'].lower() == 'questionable')
    day_to_day = sum(1 for i in team_injuries if 'day-to-day' in i['status'].lower())

    # Weighted impact score (higher = more players out)
    impact_score = out_count * 1.0 + doubtful_count * 0.7 + questionable_count * 0.3 + day_to_day * 0.2

    return {
        'injury_count': len(team_injuries),
        'out_count': out_count,
        'doubtful_count': doubtful_count,
        'questionable_count': questionable_count,
        'injury_impact_score': impact_score,
        'injured_players': [i['player_name'] for i in team_injuries if i['status'].lower() == 'out']
    }


def estimate_star_availability(team: str, injuries: List[Dict], star_players: List[str]) -> float:
    """
    Estimate availability of star players (0-1 scale).

    Args:
        team: Team abbreviation
        injuries: List of injury records
        star_players: List of star player names for the team

    Returns:
        Float 0-1 representing star availability (1 = all healthy)
    """
    if not star_players:
        return 1.0

    team_injuries = {i['player_name'].lower(): i for i in injuries if i['team'] == team}

    available = 0
    for star in star_players:
        star_lower = star.lower()
        if star_lower not in team_injuries:
            available += 1
        else:
            status = team_injuries[star_lower]['status'].lower()
            if status == 'out':
                available += 0
            elif status == 'doubtful':
                available += 0.2
            elif status == 'questionable':
                available += 0.5
            else:
                available += 0.8

    return available / len(star_players)


def add_injury_features(df: pd.DataFrame, injuries: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Add injury-related features to game dataframe.

    Args:
        df: Game dataframe with home_team, away_team columns
        injuries: Pre-fetched injuries (will fetch if None)

    Returns:
        DataFrame with added injury features
    """
    if injuries is None:
        injuries = fetch_espn_injuries()

    if not injuries:
        print("Warning: No injury data available, using defaults")
        df['home_injury_impact'] = 0
        df['away_injury_impact'] = 0
        df['injury_impact_diff'] = 0
        return df

    # Calculate impact for each game
    home_impacts = []
    away_impacts = []

    for _, row in df.iterrows():
        home_impact = get_team_injury_impact(row['home_team'], injuries)
        away_impact = get_team_injury_impact(row['away_team'], injuries)
        home_impacts.append(home_impact['injury_impact_score'])
        away_impacts.append(away_impact['injury_impact_score'])

    df = df.copy()
    df['home_injury_impact'] = home_impacts
    df['away_injury_impact'] = away_impacts
    df['injury_impact_diff'] = df['away_injury_impact'] - df['home_injury_impact']

    return df


if __name__ == "__main__":
    print("Fetching current NBA injuries...")
    injuries = fetch_espn_injuries()

    print(f"\nFound {len(injuries)} injured players\n")

    # Group by team
    by_team = {}
    for inj in injuries:
        team = inj['team']
        if team not in by_team:
            by_team[team] = []
        by_team[team].append(inj)

    for team in sorted(by_team.keys()):
        players = by_team[team]
        out = [p for p in players if p['status'].lower() == 'out']
        if out:
            print(f"{team}: {len(out)} OUT")
            for p in out:
                print(f"  - {p['player_name']} ({p['injury_type']})")
