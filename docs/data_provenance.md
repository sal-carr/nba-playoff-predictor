# Data Provenance

## Overview

This document describes the source, acquisition method, and processing of data used in the NBA prediction pipeline.

## Data Source

**Primary Source**: NBA Official Statistics API  
**Access Method**: `nba_api` Python package (version 1.11.4)  
**API Endpoint**: `leaguegamefinder.LeagueGameFinder`

## Acquisition Details

| Field | Value |
|-------|-------|
| Acquisition Date | 2026-05-05 |
| Seasons Acquired | 2022-23, 2023-24, 2024-25 |
| Game Types | Regular Season, Playoffs |
| Total Games | 3,935 |
| Total Teams | 30 |
| Date Range | 2022-10-18 to 2025-06-22 |

## Acquisition Process

1. **Raw Fetch**: Used `LeagueGameFinder` to retrieve game-team rows for each season
2. **Transformation**: Pivoted from team-game rows to game rows (one row per game)
3. **Enrichment**: Added rest days calculation based on previous game dates
4. **Synthetic Data**: Generated synthetic betting lines (spread, total) for training
5. **Storage**: Saved to `data/raw/` (original) and `data/processed/` (enriched)

## Rate Limiting

- Delay of 0.6 seconds between API calls
- No aggressive scraping or bulk crawling
- Compliant with NBA API terms of service

## Data Transformations

### Rest Days Calculation
- Computed from difference between current game date and team's previous game
- Default value for season openers: 7 days
- Maximum cap: 14 days

### Synthetic Betting Lines
Since real betting data is not available:
- `closing_spread`: Generated from actual margin with Gaussian noise (σ=5)
- `closing_total`: Generated from actual total with Gaussian noise (σ=8)
- Random seed: 42 (reproducible)

**Caveat**: Synthetic betting lines are for demonstration purposes only. The `home_cover` and `total_over` targets are derived from these synthetic values and should not be used for actual betting decisions.

## Data Quality

### Validation Checks Applied
- No missing game dates
- No games with same team on both sides
- Required columns present
- Non-empty dataset

### Known Limitations
1. Play-by-play data not included
2. Player-level statistics not included
3. Injury data not included
4. Real betting lines not available
5. Some early-season games dropped during feature engineering (insufficient rolling history)

## Reproducibility

To regenerate the data:
```bash
source .venv/bin/activate
python src/data_acquisition.py
```

This will:
1. Fetch fresh data from NBA API
2. Process and enrich it
3. Save to `data/raw/` and `data/processed/`
4. Create sample upcoming games in `data/prediction/`
