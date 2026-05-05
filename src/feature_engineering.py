"""
Enhanced Feature Engineering for NBA Game Prediction

Includes:
- Rolling averages for all box score stats
- Season win percentages (overall, home, road)
- Win/loss streaks
- Rest day differentials
- Efficiency metrics
"""

import pandas as pd
import numpy as np


def _to_long_team_view(df: pd.DataFrame) -> pd.DataFrame:
    """Convert game-level data to team-game-log format."""

    # Core stats that should exist
    core_stats = ["points", "rebounds", "turnovers", "assists"]

    # Extended stats that may exist
    extended_stats = ["steals", "blocks", "fg_pct", "fg3_pct", "ft_pct",
                      "off_rebounds", "def_rebounds", "fouls"]

    # Build home dataframe
    home_cols = ["game_id", "game_date", "home_team", "away_team", "playoff_game"]
    home_rename = {"home_team": "team", "away_team": "opponent"}

    for stat in core_stats + extended_stats:
        col = f"home_{stat}"
        if col in df.columns:
            home_cols.append(col)
            home_rename[col] = stat

    if "home_rest_days" in df.columns:
        home_cols.append("home_rest_days")
        home_rename["home_rest_days"] = "rest_days"

    home = df[home_cols].copy().rename(columns=home_rename)
    home["is_home"] = 1

    # Build away dataframe
    away_cols = ["game_id", "game_date", "away_team", "home_team", "playoff_game"]
    away_rename = {"away_team": "team", "home_team": "opponent"}

    for stat in core_stats + extended_stats:
        col = f"away_{stat}"
        if col in df.columns:
            away_cols.append(col)
            away_rename[col] = stat

    if "away_rest_days" in df.columns:
        away_cols.append("away_rest_days")
        away_rename["away_rest_days"] = "rest_days"

    away = df[away_cols].copy().rename(columns=away_rename)
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    return long_df


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive model-ready features."""

    long_df = _to_long_team_view(df)

    # Stats to compute rolling averages for
    rolling_stats = ["points", "rebounds", "turnovers", "assists"]

    # Add extended stats if available
    for stat in ["steals", "blocks", "fg_pct", "fg3_pct", "ft_pct"]:
        if stat in long_df.columns:
            rolling_stats.append(stat)

    # Compute lag and rolling features
    for col in rolling_stats:
        if col not in long_df.columns:
            continue

        # Previous game value
        long_df[f"{col}_lag_1"] = long_df.groupby("team")[col].shift(1)

        # Rolling 5-game average
        long_df[f"{col}_rolling_5"] = (
            long_df.groupby("team")[col]
            .shift(1)
            .rolling(5, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling 10-game average (more stable)
        long_df[f"{col}_rolling_10"] = (
            long_df.groupby("team")[col]
            .shift(1)
            .rolling(10, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Keep feature columns
    keep = ["game_id", "team", "is_home", "playoff_game"]
    if "rest_days" in long_df.columns:
        keep.append("rest_days")

    for col in rolling_stats:
        for suffix in ["_lag_1", "_rolling_5", "_rolling_10"]:
            feat_col = f"{col}{suffix}"
            if feat_col in long_df.columns:
                keep.append(feat_col)

    long_df = long_df[keep]

    # Split into home and away
    home_feats = long_df[long_df["is_home"] == 1].copy()
    away_feats = long_df[long_df["is_home"] == 0].copy()

    # Rename with home/away prefix
    home_rename = {"team": "home_team"}
    away_rename = {"team": "away_team"}

    for col in home_feats.columns:
        if col not in ["game_id", "home_team", "team", "is_home", "playoff_game"]:
            home_rename[col] = f"home_{col}"
            away_rename[col] = f"away_{col}"

    home_feats = home_feats.rename(columns=home_rename)
    away_feats = away_feats.rename(columns=away_rename)

    # Drop helper columns
    home_feats = home_feats.drop(columns=["is_home", "playoff_game"], errors="ignore")
    away_feats = away_feats.drop(columns=["is_home", "playoff_game"], errors="ignore")

    # Merge back to game level
    out = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    out = out.merge(away_feats, on=["game_id", "away_team"], how="left")

    # Add pre-computed season records if they exist
    season_cols = ["home_season_win_pct", "away_season_win_pct",
                   "home_home_win_pct", "away_road_win_pct",
                   "home_streak", "away_streak"]

    # Create target variables
    out["home_win"] = (out["home_points"] > out["away_points"]).astype(int)

    if "closing_spread" in out.columns:
        out["home_cover"] = (
            (out["home_points"] - out["away_points"]) > out["closing_spread"]
        ).astype(int)

    if "closing_total" in out.columns:
        out["total_over"] = (
            (out["home_points"] + out["away_points"]) > out["closing_total"]
        ).astype(int)

    # Create differential features
    diff_pairs = [
        ("home_rest_days", "away_rest_days", "rest_diff"),
        ("home_points_rolling_5", "away_points_rolling_5", "rolling_points_diff"),
        ("home_points_rolling_10", "away_points_rolling_10", "rolling_points_diff_10"),
        ("home_rebounds_rolling_5", "away_rebounds_rolling_5", "rolling_rebounds_diff"),
        ("home_turnovers_rolling_5", "away_turnovers_rolling_5", "rolling_turnovers_diff"),
        ("home_assists_rolling_5", "away_assists_rolling_5", "rolling_assists_diff"),
        ("home_season_win_pct", "away_season_win_pct", "season_win_pct_diff"),
        ("home_home_win_pct", "away_road_win_pct", "home_away_win_pct_diff"),
        ("home_streak", "away_streak", "streak_diff"),
    ]

    for home_col, away_col, diff_col in diff_pairs:
        if home_col in out.columns and away_col in out.columns:
            out[diff_col] = out[home_col] - out[away_col]

    # Efficiency metrics if shooting stats available
    if "home_fg_pct_rolling_5" in out.columns:
        out["home_shooting_eff"] = (
            out["home_fg_pct_rolling_5"] * 0.5 +
            out.get("home_fg3_pct_rolling_5", out["home_fg_pct_rolling_5"]) * 0.3 +
            out.get("home_ft_pct_rolling_5", 0.75) * 0.2
        )
        out["away_shooting_eff"] = (
            out["away_fg_pct_rolling_5"] * 0.5 +
            out.get("away_fg3_pct_rolling_5", out["away_fg_pct_rolling_5"]) * 0.3 +
            out.get("away_ft_pct_rolling_5", 0.75) * 0.2
        )
        out["shooting_eff_diff"] = out["home_shooting_eff"] - out["away_shooting_eff"]

    return out


def get_enhanced_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature columns available in the dataframe."""

    # Core rolling features
    core_features = [
        "home_points_lag_1", "home_points_rolling_5", "home_points_rolling_10",
        "away_points_lag_1", "away_points_rolling_5", "away_points_rolling_10",
        "home_rebounds_lag_1", "home_rebounds_rolling_5",
        "away_rebounds_lag_1", "away_rebounds_rolling_5",
        "home_turnovers_lag_1", "home_turnovers_rolling_5",
        "away_turnovers_lag_1", "away_turnovers_rolling_5",
        "home_assists_lag_1", "home_assists_rolling_5",
        "away_assists_lag_1", "away_assists_rolling_5",
    ]

    # Extended features
    extended_features = [
        "home_steals_rolling_5", "away_steals_rolling_5",
        "home_blocks_rolling_5", "away_blocks_rolling_5",
        "home_fg_pct_rolling_5", "away_fg_pct_rolling_5",
        "home_fg3_pct_rolling_5", "away_fg3_pct_rolling_5",
    ]

    # Season/streak features
    season_features = [
        "home_season_win_pct", "away_season_win_pct",
        "home_home_win_pct", "away_road_win_pct",
        "home_streak", "away_streak",
    ]

    # Differential features
    diff_features = [
        "rest_diff", "rolling_points_diff", "rolling_points_diff_10",
        "rolling_rebounds_diff", "rolling_turnovers_diff", "rolling_assists_diff",
        "season_win_pct_diff", "home_away_win_pct_diff", "streak_diff",
        "shooting_eff_diff",
    ]

    # Other
    other_features = ["playoff_game"]

    # Filter to columns that exist
    all_candidates = core_features + extended_features + season_features + diff_features + other_features
    available = [c for c in all_candidates if c in df.columns]

    return available
