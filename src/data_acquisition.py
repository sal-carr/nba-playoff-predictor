"""
Enhanced NBA Data Acquisition Module

Fetches comprehensive NBA game data including:
- Box score statistics (FG%, 3P%, FT%, AST, STL, BLK)
- Advanced team stats
- Home/road splits
- Recent form (win streaks)
"""

import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    boxscoretraditionalv2,
    leaguestandings
)
from nba_api.stats.static import teams


def get_all_nba_teams() -> pd.DataFrame:
    """Get all NBA teams with their IDs."""
    return pd.DataFrame(teams.get_teams())


def fetch_enhanced_games_for_seasons(
    seasons: list[str],
    output_path: str = "data/raw/nba_games_enhanced.csv",
    delay_seconds: float = 0.6
) -> pd.DataFrame:
    """
    Fetch NBA games with enhanced statistics.
    """
    all_games = []

    for season in seasons:
        print(f"Fetching games for season {season}...")

        for season_type in ["Regular Season", "Playoffs"]:
            try:
                game_finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable="00",
                    season_type_nullable=season_type
                )
                time.sleep(delay_seconds)

                df = game_finder.get_data_frames()[0]
                df["season_type"] = season_type
                all_games.append(df)
                print(f"  Found {len(df)} {season_type.lower()} game-team rows")

            except Exception as e:
                print(f"  Error fetching {season_type}: {e}")
                continue

    if not all_games:
        raise ValueError("No game data retrieved from API")

    raw_df = pd.concat(all_games, ignore_index=True)
    games_df = _transform_to_enhanced_game_rows(raw_df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(output_file, index=False)
    print(f"Saved {len(games_df)} games to {output_path}")

    return games_df


def _transform_to_enhanced_game_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform team-game rows into game rows with enhanced stats.
    """
    raw_df["is_home"] = raw_df["MATCHUP"].str.contains(" vs. ")

    home_games = raw_df[raw_df["is_home"]].copy()
    away_games = raw_df[~raw_df["is_home"]].copy()

    # Enhanced home columns
    home_cols = {
        "TEAM_ID": "home_team_id",
        "TEAM_ABBREVIATION": "home_team",
        "TEAM_NAME": "home_team_name",
        "PTS": "home_points",
        "REB": "home_rebounds",
        "TOV": "home_turnovers",
        "AST": "home_assists",
        "STL": "home_steals",
        "BLK": "home_blocks",
        "FG_PCT": "home_fg_pct",
        "FG3_PCT": "home_fg3_pct",
        "FT_PCT": "home_ft_pct",
        "OREB": "home_off_rebounds",
        "DREB": "home_def_rebounds",
        "PF": "home_fouls",
        "PLUS_MINUS": "home_plus_minus",
    }
    home_games = home_games.rename(columns=home_cols)

    # Enhanced away columns
    away_cols = {
        "TEAM_ID": "away_team_id",
        "TEAM_ABBREVIATION": "away_team",
        "TEAM_NAME": "away_team_name",
        "PTS": "away_points",
        "REB": "away_rebounds",
        "TOV": "away_turnovers",
        "AST": "away_assists",
        "STL": "away_steals",
        "BLK": "away_blocks",
        "FG_PCT": "away_fg_pct",
        "FG3_PCT": "away_fg3_pct",
        "FT_PCT": "away_ft_pct",
        "OREB": "away_off_rebounds",
        "DREB": "away_def_rebounds",
        "PF": "away_fouls",
        "PLUS_MINUS": "away_plus_minus",
    }
    away_games = away_games.rename(columns=away_cols)

    # Select columns
    home_keep = [
        "GAME_ID", "GAME_DATE", "SEASON_ID", "season_type",
        "home_team_id", "home_team", "home_team_name",
        "home_points", "home_rebounds", "home_turnovers",
        "home_assists", "home_steals", "home_blocks",
        "home_fg_pct", "home_fg3_pct", "home_ft_pct",
        "home_off_rebounds", "home_def_rebounds", "home_fouls", "home_plus_minus"
    ]
    away_keep = [
        "GAME_ID",
        "away_team_id", "away_team", "away_team_name",
        "away_points", "away_rebounds", "away_turnovers",
        "away_assists", "away_steals", "away_blocks",
        "away_fg_pct", "away_fg3_pct", "away_ft_pct",
        "away_off_rebounds", "away_def_rebounds", "away_fouls", "away_plus_minus"
    ]

    # Filter to existing columns
    home_keep = [c for c in home_keep if c in home_games.columns]
    away_keep = [c for c in away_keep if c in away_games.columns]

    home_games = home_games[home_keep]
    away_games = away_games[away_keep]

    # Merge
    games_df = home_games.merge(away_games, on="GAME_ID", how="inner")

    games_df = games_df.rename(columns={
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "SEASON_ID": "season",
    })

    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df["sport"] = "nba"
    games_df["playoff_game"] = (games_df["season_type"] == "Playoffs").astype(int)

    games_df = games_df.sort_values("game_date").reset_index(drop=True)

    return games_df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rest days for each team."""
    df = df.sort_values("game_date").copy()

    home_dates = df[["game_date", "home_team"]].copy()
    home_dates.columns = ["game_date", "team"]

    away_dates = df[["game_date", "away_team"]].copy()
    away_dates.columns = ["game_date", "team"]

    all_team_dates = pd.concat([home_dates, away_dates]).sort_values("game_date")
    all_team_dates["prev_game_date"] = all_team_dates.groupby("team")["game_date"].shift(1)

    home_rest = all_team_dates.rename(columns={"team": "home_team", "prev_game_date": "home_prev_game"})
    df = df.merge(
        home_rest[["game_date", "home_team", "home_prev_game"]].drop_duplicates(),
        on=["game_date", "home_team"],
        how="left"
    )

    away_rest = all_team_dates.rename(columns={"team": "away_team", "prev_game_date": "away_prev_game"})
    df = df.merge(
        away_rest[["game_date", "away_team", "away_prev_game"]].drop_duplicates(),
        on=["game_date", "away_team"],
        how="left"
    )

    df["home_rest_days"] = (df["game_date"] - df["home_prev_game"]).dt.days
    df["away_rest_days"] = (df["game_date"] - df["away_prev_game"]).dt.days

    df["home_rest_days"] = df["home_rest_days"].fillna(7).clip(upper=14)
    df["away_rest_days"] = df["away_rest_days"].fillna(7).clip(upper=14)

    df = df.drop(columns=["home_prev_game", "away_prev_game"])

    return df


def add_win_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate current win/loss streak for each team."""
    df = df.sort_values("game_date").copy()

    # Create team game log with wins
    home_log = df[["game_date", "home_team", "home_points", "away_points"]].copy()
    home_log["team"] = home_log["home_team"]
    home_log["won"] = (home_log["home_points"] > home_log["away_points"]).astype(int)

    away_log = df[["game_date", "away_team", "home_points", "away_points"]].copy()
    away_log["team"] = away_log["away_team"]
    away_log["won"] = (away_log["home_points"] < away_log["away_points"]).astype(int)

    all_log = pd.concat([
        home_log[["game_date", "team", "won"]],
        away_log[["game_date", "team", "won"]]
    ]).sort_values(["team", "game_date"])

    # Calculate streak (positive = wins, negative = losses)
    def calc_streak(group):
        streaks = []
        current_streak = 0
        for won in group["won"]:
            if won == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streaks.append(current_streak)
        # Shift to get streak BEFORE current game
        return pd.Series([0] + streaks[:-1], index=group.index)

    all_log["streak"] = all_log.groupby("team", group_keys=False).apply(calc_streak)

    # Merge back
    home_streak = all_log[["game_date", "team", "streak"]].rename(
        columns={"team": "home_team", "streak": "home_streak"}
    ).drop_duplicates(["game_date", "home_team"])

    away_streak = all_log[["game_date", "team", "streak"]].rename(
        columns={"team": "away_team", "streak": "away_streak"}
    ).drop_duplicates(["game_date", "away_team"])

    df = df.merge(home_streak, on=["game_date", "home_team"], how="left")
    df = df.merge(away_streak, on=["game_date", "away_team"], how="left")

    df["home_streak"] = df["home_streak"].fillna(0)
    df["away_streak"] = df["away_streak"].fillna(0)

    return df


def add_season_records(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate running season win percentage for each team."""
    df = df.sort_values("game_date").copy()

    # Extract season start
    df["season_start"] = df["game_date"].dt.year.where(
        df["game_date"].dt.month >= 10,
        df["game_date"].dt.year - 1
    )

    # Build team game log
    home_log = df[["game_date", "season_start", "home_team", "home_points", "away_points"]].copy()
    home_log["team"] = home_log["home_team"]
    home_log["won"] = (home_log["home_points"] > home_log["away_points"]).astype(int)
    home_log["is_home"] = 1

    away_log = df[["game_date", "season_start", "away_team", "home_points", "away_points"]].copy()
    away_log["team"] = away_log["away_team"]
    away_log["won"] = (away_log["home_points"] < away_log["away_points"]).astype(int)
    away_log["is_home"] = 0

    all_log = pd.concat([
        home_log[["game_date", "season_start", "team", "won", "is_home"]],
        away_log[["game_date", "season_start", "team", "won", "is_home"]]
    ]).sort_values(["team", "season_start", "game_date"])

    # Calculate cumulative wins and games (shifted to exclude current game)
    all_log["cum_wins"] = all_log.groupby(["team", "season_start"])["won"].cumsum().shift(1).fillna(0)
    all_log["cum_games"] = all_log.groupby(["team", "season_start"]).cumcount()
    all_log["win_pct"] = (all_log["cum_wins"] / all_log["cum_games"].clip(lower=1)).fillna(0.5)

    # Home and road records
    all_log["cum_home_wins"] = all_log.groupby(["team", "season_start"]).apply(
        lambda g: (g["won"] * g["is_home"]).cumsum().shift(1).fillna(0)
    ).reset_index(level=[0,1], drop=True)
    all_log["cum_home_games"] = all_log.groupby(["team", "season_start"]).apply(
        lambda g: g["is_home"].cumsum().shift(1).fillna(0)
    ).reset_index(level=[0,1], drop=True)
    all_log["home_win_pct"] = (all_log["cum_home_wins"] / all_log["cum_home_games"].clip(lower=1)).fillna(0.5)

    all_log["cum_road_wins"] = all_log.groupby(["team", "season_start"]).apply(
        lambda g: (g["won"] * (1 - g["is_home"])).cumsum().shift(1).fillna(0)
    ).reset_index(level=[0,1], drop=True)
    all_log["cum_road_games"] = all_log.groupby(["team", "season_start"]).apply(
        lambda g: (1 - g["is_home"]).cumsum().shift(1).fillna(0)
    ).reset_index(level=[0,1], drop=True)
    all_log["road_win_pct"] = (all_log["cum_road_wins"] / all_log["cum_road_games"].clip(lower=1)).fillna(0.5)

    # Merge back
    home_records = all_log[["game_date", "team", "win_pct", "home_win_pct", "road_win_pct"]].rename(columns={
        "team": "home_team",
        "win_pct": "home_season_win_pct",
        "home_win_pct": "home_home_win_pct",
        "road_win_pct": "home_road_win_pct"
    }).drop_duplicates(["game_date", "home_team"])

    away_records = all_log[["game_date", "team", "win_pct", "home_win_pct", "road_win_pct"]].rename(columns={
        "team": "away_team",
        "win_pct": "away_season_win_pct",
        "home_win_pct": "away_home_win_pct",
        "road_win_pct": "away_road_win_pct"
    }).drop_duplicates(["game_date", "away_team"])

    df = df.merge(home_records, on=["game_date", "home_team"], how="left")
    df = df.merge(away_records, on=["game_date", "away_team"], how="left")

    # Fill missing with 0.5
    for col in ["home_season_win_pct", "home_home_win_pct", "home_road_win_pct",
                "away_season_win_pct", "away_home_win_pct", "away_road_win_pct"]:
        df[col] = df[col].fillna(0.5)

    df = df.drop(columns=["season_start"])

    return df


def add_synthetic_betting_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic betting lines."""
    df = df.copy()

    actual_margin = df["home_points"] - df["away_points"]

    np.random.seed(42)
    noise = np.random.normal(0, 5, len(df))
    df["closing_spread"] = -actual_margin + noise
    df["closing_spread"] = df["closing_spread"].round(1)

    actual_total = df["home_points"] + df["away_points"]
    total_noise = np.random.normal(0, 8, len(df))
    df["closing_total"] = actual_total + total_noise
    df["closing_total"] = df["closing_total"].round(1)

    return df


def run_enhanced_acquisition(
    seasons: list[str] = None,
    output_dir: str = "data"
) -> pd.DataFrame:
    """Run full enhanced data acquisition."""
    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25"]

    raw_path = f"{output_dir}/raw/nba_games_enhanced.csv"
    processed_path = f"{output_dir}/processed/nba_games_enhanced.csv"

    print("=" * 60)
    print("Enhanced NBA Data Acquisition")
    print("=" * 60)

    games_df = fetch_enhanced_games_for_seasons(seasons, raw_path)

    print("\nCalculating rest days...")
    games_df = add_rest_days(games_df)

    print("Calculating win streaks...")
    games_df = add_win_streaks(games_df)

    print("Calculating season records...")
    games_df = add_season_records(games_df)

    print("Adding synthetic betting lines...")
    games_df = add_synthetic_betting_lines(games_df)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    print("\n" + "=" * 60)
    print("Acquisition complete!")
    print(f"  Total games: {len(games_df)}")
    print(f"  Columns: {len(games_df.columns)}")
    print(f"  Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
    print("=" * 60)

    return games_df


if __name__ == "__main__":
    run_enhanced_acquisition()
