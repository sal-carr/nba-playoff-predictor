"""
NBA Betting Lines Data

Fetches historical betting lines from free sources.
Primary source: Sportsbook Reviews Online (SBRO)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests
from io import StringIO


# SBRO Excel file URLs by season
SBRO_URLS = {
    '2024-25': 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202024-25.xlsx',
    '2023-24': 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202023-24.xlsx',
    '2022-23': 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202022-23.xlsx',
    '2021-22': 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202021-22.xlsx',
}

# Team name mappings (SBRO uses different names)
SBRO_TEAM_MAP = {
    'Atlanta': 'ATL',
    'Boston': 'BOS',
    'Brooklyn': 'BKN',
    'Charlotte': 'CHA',
    'Chicago': 'CHI',
    'Cleveland': 'CLE',
    'Dallas': 'DAL',
    'Denver': 'DEN',
    'Detroit': 'DET',
    'GoldenState': 'GSW',
    'Golden State': 'GSW',
    'Houston': 'HOU',
    'Indiana': 'IND',
    'LAClippers': 'LAC',
    'LA Clippers': 'LAC',
    'LALakers': 'LAL',
    'LA Lakers': 'LAL',
    'Memphis': 'MEM',
    'Miami': 'MIA',
    'Milwaukee': 'MIL',
    'Minnesota': 'MIN',
    'NewOrleans': 'NOP',
    'New Orleans': 'NOP',
    'NewYork': 'NYK',
    'New York': 'NYK',
    'OklahomaCity': 'OKC',
    'Oklahoma City': 'OKC',
    'Orlando': 'ORL',
    'Philadelphia': 'PHI',
    'Phoenix': 'PHX',
    'Portland': 'POR',
    'Sacramento': 'SAC',
    'SanAntonio': 'SAS',
    'San Antonio': 'SAS',
    'Toronto': 'TOR',
    'Utah': 'UTA',
    'Washington': 'WAS',
}


def download_sbro_odds(season: str, cache_dir: str = 'data/odds') -> Optional[pd.DataFrame]:
    """
    Download betting odds from Sportsbook Reviews Online.

    Args:
        season: Season string like '2023-24'
        cache_dir: Directory to cache downloaded files

    Returns:
        DataFrame with betting lines or None if failed
    """
    if season not in SBRO_URLS:
        print(f"Warning: No SBRO data available for {season}")
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"sbro_odds_{season.replace('-', '_')}.csv"

    # Check cache first
    if cache_file.exists():
        print(f"Loading cached odds for {season}")
        return pd.read_csv(cache_file)

    url = SBRO_URLS[season]
    print(f"Downloading odds for {season} from SBRO...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Read Excel file
        df = pd.read_excel(response.content, engine='openpyxl')

        # Process the data
        df = _process_sbro_data(df)

        # Cache it
        df.to_csv(cache_file, index=False)
        print(f"  Cached {len(df)} games to {cache_file}")

        return df

    except Exception as e:
        print(f"Warning: Could not download SBRO odds for {season}: {e}")
        return None


def _process_sbro_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw SBRO data into standardized format."""

    # SBRO format typically has columns like:
    # Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H

    # Standardize column names (varies by file)
    df.columns = df.columns.str.strip().str.lower()

    # Map common column name variations
    col_map = {
        'date': 'date',
        'team': 'team',
        'vh': 'venue',  # V=visitor, H=home
        'final': 'score',
        'close': 'spread',
        'open': 'open_spread',
        'ml': 'moneyline',
        'ou': 'total',
        'over/under': 'total',
    }

    for old, new in col_map.items():
        matching = [c for c in df.columns if old in c.lower()]
        if matching and new not in df.columns:
            df = df.rename(columns={matching[0]: new})

    # Ensure required columns exist
    required = ['date', 'team']
    for col in required:
        if col not in df.columns:
            print(f"Warning: Missing required column '{col}'")
            return pd.DataFrame()

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Map team names
    if 'team' in df.columns:
        df['team_abbr'] = df['team'].map(lambda x: SBRO_TEAM_MAP.get(str(x).strip(), str(x)))

    return df


def merge_odds_with_games(games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge betting odds with game data.

    Args:
        games_df: Game data with game_date, home_team, away_team
        odds_df: Odds data from SBRO

    Returns:
        Games dataframe with added odds columns
    """
    if odds_df is None or len(odds_df) == 0:
        return games_df

    # This is complex because SBRO has one row per team
    # We need to match on date + teams

    games_df = games_df.copy()
    games_df['closing_spread'] = np.nan
    games_df['closing_total'] = np.nan
    games_df['home_moneyline'] = np.nan
    games_df['away_moneyline'] = np.nan

    # TODO: Implement proper matching logic
    # For now, return with NaN odds
    return games_df


def fetch_odds_api_live(api_key: str, sport: str = 'basketball_nba') -> Optional[pd.DataFrame]:
    """
    Fetch live odds from The Odds API.

    Args:
        api_key: API key from the-odds-api.com
        sport: Sport key

    Returns:
        DataFrame with current odds
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads,totals,h2h',
        'oddsFormat': 'american'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        games = []
        for game in data:
            game_info = {
                'game_id': game['id'],
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
            }

            # Extract best odds from bookmakers
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                game_info['home_spread'] = outcome.get('point', 0)
                            else:
                                game_info['away_spread'] = outcome.get('point', 0)
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                game_info['total'] = outcome.get('point', 0)
                                break
                break  # Just use first bookmaker for now

            games.append(game_info)

        return pd.DataFrame(games)

    except Exception as e:
        print(f"Warning: Could not fetch live odds: {e}")
        return None


def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add betting-line derived features.

    Features:
    - spread_normalized: Spread as fraction of typical game margin
    - implied_prob: Implied win probability from moneyline
    - line_value: Difference between model prob and implied prob
    """
    df = df.copy()

    # Normalized spread (typical NBA game margin is ~10 points)
    if 'closing_spread' in df.columns:
        df['spread_normalized'] = df['closing_spread'] / 10.0

    # Implied probability from moneyline
    if 'home_moneyline' in df.columns:
        df['home_implied_prob'] = df['home_moneyline'].apply(_ml_to_prob)

    return df


def _ml_to_prob(ml: float) -> float:
    """Convert American moneyline to implied probability."""
    if pd.isna(ml):
        return 0.5
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


if __name__ == "__main__":
    # Test downloading odds
    for season in ['2023-24', '2024-25']:
        df = download_sbro_odds(season)
        if df is not None:
            print(f"\n{season}: {len(df)} records")
            print(df.head())
