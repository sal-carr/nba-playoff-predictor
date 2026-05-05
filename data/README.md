# Data Directory

This directory contains NBA game data for the prediction pipeline.

## Directory Structure

```
data/
├── sample/                 # Sample data (committed to git)
│   └── nba_games_sample.csv
├── raw/                    # Original data from NBA API (gitignored)
│   └── nba_games.csv
├── processed/              # Cleaned and enriched data (gitignored)
│   └── nba_games_enhanced.csv
└── prediction/             # Input data for prediction flow
    └── upcoming_games.csv
```

## Quick Start

To fetch full data, run:
```bash
python src/data_acquisition.py
```

Or use the sample data for testing:
```bash
cp data/sample/nba_games_sample.csv data/processed/nba_games_enhanced.csv
```

## Data Schema

### nba_games_enhanced.csv

| Column | Type | Description |
|--------|------|-------------|
| game_id | string | Unique NBA game identifier |
| game_date | datetime | Date of the game |
| season | string | Season identifier (e.g., "2024-25") |
| season_type | string | "Regular Season" or "Playoffs" |
| sport | string | Always "nba" |
| playoff_game | int | 1 if playoff game, 0 otherwise |
| home_team | string | Home team abbreviation (e.g., "LAL") |
| away_team | string | Away team abbreviation |
| home_points | int | Points scored by home team |
| away_points | int | Points scored by away team |
| home_rebounds | int | Rebounds by home team |
| away_rebounds | int | Rebounds by away team |
| home_assists | int | Assists by home team |
| away_assists | int | Assists by away team |
| home_turnovers | int | Turnovers by home team |
| away_turnovers | int | Turnovers by away team |
| home_steals | int | Steals by home team |
| away_steals | int | Steals by away team |
| home_blocks | int | Blocks by home team |
| away_blocks | int | Blocks by away team |
| home_fg_pct | float | Home team field goal percentage |
| away_fg_pct | float | Away team field goal percentage |
| home_fg3_pct | float | Home team 3-point percentage |
| away_fg3_pct | float | Away team 3-point percentage |
| home_rest_days | float | Days since home team's last game |
| away_rest_days | float | Days since away team's last game |
| home_season_win_pct | float | Home team's season win percentage |
| away_season_win_pct | float | Away team's season win percentage |
| home_home_win_pct | float | Home team's home win percentage |
| away_road_win_pct | float | Away team's road win percentage |
| home_streak | int | Home team's win/loss streak (+/- games) |
| away_streak | int | Away team's win/loss streak |

## Data Sources

- **Primary**: NBA Stats API via `nba_api` Python package
- **Seasons available**: 2022-23 through 2025-26
- **Game types**: Regular season and playoffs

## Notes

1. **Rest days**: Calculated from the previous game date for each team. Season openers default to 7 days.

2. **Streaks**: Positive values indicate wins, negative indicate losses (e.g., +3 = 3 game win streak).

3. **Feature engineering**: The training flow computes additional features (rolling averages, differentials) from this base data. See `src/feature_engineering.py`.

4. **Full data not committed**: The full dataset (~5000+ games) is gitignored to keep the repository lightweight. Use `data_acquisition.py` to fetch it.
