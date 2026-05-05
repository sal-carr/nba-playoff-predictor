"""
I/O utilities for NBA game data loading and validation.
"""

from pathlib import Path
import pandas as pd


def load_game_data(csv_path: str) -> pd.DataFrame:
    """
    Load game data from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with game data, dates parsed
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    df = pd.read_csv(path, parse_dates=["game_date"])

    return df


def validate_raw_games(df: pd.DataFrame) -> None:
    """
    Validate that the game DataFrame has required columns and data quality.

    Raises:
        ValueError: If validation fails
    """
    required = {
        "game_id", "game_date", "sport", "playoff_game",
        "home_team", "away_team",
        "home_points", "away_points",
        "home_rebounds", "away_rebounds",
        "home_turnovers", "away_turnovers",
        "home_rest_days", "away_rest_days",
        "closing_spread", "closing_total"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["game_date"].isna().any():
        raise ValueError("Some game_date values could not be parsed.")

    if (df["home_team"] == df["away_team"]).any():
        raise ValueError("A game row has the same team on both sides.")


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    """
    Save predictions DataFrame to CSV.

    Args:
        df: DataFrame with predictions
        output_path: Where to save
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved predictions to {output_path}")
