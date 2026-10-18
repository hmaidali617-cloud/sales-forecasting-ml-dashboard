"""
feature_engineering.py
======================
Generate temporal, lag, rolling, expanding, holiday features.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

from utils import get_logger

logger = get_logger("features")

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14]
# feature_engineering.py - ADD THESE NEW FUNCTIONS

def add_peak_features(df: pd.DataFrame, target: str = "revenue", group_col: str = "store_id") -> pd.DataFrame:
    """
    Add features that help models predict peak days.
    This is THE KEY to predicting values > 1000 DT.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])
    
    # 1. Is this day historically a peak day? (Top 20% of that day-of-week)
    def is_peak_day(g):
        if len(g) < 4:
            return pd.Series(0, index=g.index)
        threshold = g[target].quantile(0.80)
        return (g[target] >= threshold).astype(int)
    
    df["hist_peak_dow"] = df.groupby([group_col, df["date"].dt.dayofweek], group_keys=False).apply(is_peak_day).values
    
    # 2. Historical max for this day-of-week (what's the ceiling?)
    df["dow_hist_max"] = df.groupby([group_col, df["date"].dt.dayofweek])[target].transform("max")
    df["dow_hist_p90"] = df.groupby([group_col, df["date"].dt.dayofweek])[target].transform(
        lambda x: x.quantile(0.90) if len(x) > 3 else x.max()
    )
    df["dow_hist_mean"] = df.groupby([group_col, df["date"].dt.dayofweek])[target].transform("mean")
    
    # 3. How far below the historical max are we? (upside potential)
    df["upside_to_max"] = (df["dow_hist_max"] - df[target]) / (df["dow_hist_max"] + 1e-6)
    df["upside_to_p90"] = (df["dow_hist_p90"] - df[target]) / (df["dow_hist_p90"] + 1e-6)
    
    # 4. Peak ratio: current vs historical average for this day-of-week
    df["dow_ratio"] = df[target] / (df["dow_hist_mean"] + 1e-6)
    
    # 5. Days since last peak (momentum indicator)
    def days_since_peak(g):
        peaks = g[g[target] >= g[target].quantile(0.80)].index
        if len(peaks) == 0:
            return pd.Series(999, index=g.index)
        last_peak_pos = g.index.get_loc(peaks[-1])
        return pd.Series(range(len(g)) - last_peak_pos, index=g.index)
    
    df["days_since_peak"] = df.groupby(group_col, group_keys=False).apply(days_since_peak).values
    
    # 6. Consecutive low days (higher chance of rebound)
    def consecutive_low(g):
        threshold = g[target].quantile(0.40)
        is_low = (g[target] <= threshold).astype(int)
        groups = (is_low != is_low.shift()).cumsum()
        return is_low.groupby(groups).cumsum() * is_low
    
    df["consecutive_low_days"] = df.groupby(group_col, group_keys=False).apply(consecutive_low).values
    
    # 7. Month-end effect (common in retail - salary days)
    df["days_to_month_end"] = (df["date"].dt.days_in_month - df["date"].dt.day)
    df["is_month_end"] = (df["days_to_month_end"] <= 5).astype(int)
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
    
    # 8. Week-of-month (1-5) - some weeks have higher sales
    df["week_of_month"] = ((df["date"].dt.day - 1) // 7) + 1
    df["week_of_month_sin"] = np.sin(2 * np.pi * df["week_of_month"] / 5)
    df["week_of_month_cos"] = np.cos(2 * np.pi * df["week_of_month"] / 5)
    
    return df


def add_volatility_features(df: pd.DataFrame, target: str = "revenue", group_col: str = "store_id") -> pd.DataFrame:
    """
    Add features that capture volatility - high volatility = potential for peaks.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])
    
    # 1. Coefficient of variation (volatility relative to mean)
    df["cv_14"] = df.groupby(group_col)[target].transform(
        lambda x: x.rolling(14, min_periods=3).std() / (x.rolling(14, min_periods=3).mean() + 1e-6)
    ).fillna(0)
    
    # 2. Range (max - min) in recent window
    df["range_14"] = df.groupby(group_col)[target].transform(
        lambda x: x.rolling(14, min_periods=3).max() - x.rolling(14, min_periods=3).min()
    ).fillna(0)
    
    # 3. Gap from rolling max (are we due for a spike?)
    df["gap_from_14d_max"] = df.groupby(group_col)[target].transform(
        lambda x: x.rolling(14, min_periods=3).max() - x
    ).fillna(0)
    
    # 4. Recent acceleration (is revenue trending up?)
    df["accel_3d"] = df.groupby(group_col)[target].transform(
        lambda x: x - x.shift(3)
    ).fillna(0)
    
    df["accel_7d"] = df.groupby(group_col)[target].transform(
        lambda x: x - x.shift(7)
    ).fillna(0)
    
    return df


def add_momentum_features(df: pd.DataFrame, target: str = "revenue", group_col: str = "store_id") -> pd.DataFrame:
    """
    Add momentum features - if trending up, likely to continue.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])
    
    # 1. Short-term momentum (3-day)
    df["mom_3d"] = df.groupby(group_col)[target].transform(
        lambda x: x.pct_change(3)
    ).fillna(0)
    
    # 2. Medium-term momentum (7-day)
    df["mom_7d"] = df.groupby(group_col)[target].transform(
        lambda x: x.pct_change(7)
    ).fillna(0)
    
    # 3. Rate of change acceleration
    df["mom_accel"] = df["mom_3d"] - df["mom_7d"]
    
    # 4. Position in recent range (0 = at min, 1 = at max)
    df["position_in_range"] = df.groupby(group_col).apply(
        lambda g: (g[target] - g[target].rolling(14, min_periods=3).min()) / 
                 (g[target].rolling(14, min_periods=3).max() - g[target].rolling(14, min_periods=3).min() + 1e-6)
    ).reset_index(level=0, drop=True).fillna(0.5)
    
    return df

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["quarter"] = d.dt.quarter
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    # Cyclic encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


def add_holiday_features(df: pd.DataFrame, date_col: str = "date", country: str = "TN") -> pd.DataFrame:
    df = df.copy()
    if not _HAS_HOLIDAYS:
        df["is_holiday"] = 0
        return df
    years = sorted(pd.to_datetime(df[date_col]).dt.year.unique())
    try:
        hol = _holidays.CountryHoliday(country, years=years)
    except Exception:
        try:
            hol = _holidays.CountryHoliday("FR", years=years)
        except Exception:
            df["is_holiday"] = 0
            return df
    df["is_holiday"] = pd.to_datetime(df[date_col]).dt.date.map(lambda x: 1 if x in hol else 0).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "revenue", group_col: Optional[str] = "store_id") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([group_col, "date"] if group_col else ["date"])
    grp = df.groupby(group_col)[target] if group_col else df[target]
    for lag in LAGS:
        df[f"lag_{lag}"] = (grp.shift(lag) if group_col else df[target].shift(lag))
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "revenue", group_col: Optional[str] = "store_id") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([group_col, "date"] if group_col else ["date"])
    for w in ROLLING_WINDOWS:
        if group_col:
            shifted = df.groupby(group_col)[target].shift(1)
        else:
            shifted = df[target].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True) if group_col else shifted.rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True) if group_col else shifted.rolling(window=w, min_periods=1).std()
    return df


def add_expanding_features(df: pd.DataFrame, target: str = "revenue", group_col: Optional[str] = "store_id") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([group_col, "date"] if group_col else ["date"])
    if group_col:
        shifted = df.groupby(group_col)[target].shift(1)
        df["expanding_mean"] = shifted.rolling(window=len(df), min_periods=1).mean().reset_index(level=0, drop=True)
    else:
        shifted = df[target].shift(1)
        df["expanding_mean"] = shifted.expanding(min_periods=1).mean()
    return df


def add_trend_features(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    df = df.copy()
    df["trend"] = np.arange(len(df))
    df["monthly_trend"] = df["trend"] * df["month"] / 12.0
    df["ma_7"] = df[target].rolling(7, min_periods=1).mean()
    df["ma_14"] = df[target].rolling(14, min_periods=1).mean()
    df["ma_diff"] = df["ma_7"] - df["ma_14"]
    return df


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame, group_col: str = "store_id") -> pd.DataFrame:
    if weather is None or weather.empty:
        for c in ["temp_avg", "temp_max", "temp_min", "rain", "wind_speed", "humidity"]:
            df[c] = 0.0
        return df
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.merge(w, on=["date", group_col], how="left")


def add_store_popularity(df: pd.DataFrame, group_col: str = "store_id") -> pd.DataFrame:
    df = df.copy()
    pop = df.groupby(group_col)["revenue"].transform("mean")
    df["store_popularity"] = pop
    df["store_share"] = df["revenue"] / df["revenue"].sum() if df["revenue"].sum() > 0 else 0
    return df


# feature_engineering.py - UPDATE build_features function

def build_features(
    daily_df: pd.DataFrame,
    weather: Optional[pd.DataFrame] = None,
    target: str = "revenue",
    group_col: Optional[str] = "store_id",
) -> pd.DataFrame:
    """End-to-end feature engineering pipeline WITH PEAK DETECTION."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if group_col and group_col in df.columns:
        df[group_col] = df[group_col].astype(str)
        df = df.sort_values([group_col, "date"])
    else:
        df = df.sort_values("date")
        group_col = None

    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)

    # Calendar
    df = add_calendar_features(df, "date")
    df = add_holiday_features(df, "date")

    # Lags + rolling + expanding
    df = add_lag_features(df, target, group_col)
    df = add_rolling_features(df, target, group_col)
    df = add_expanding_features(df, target, group_col)

    # Weather
    if group_col:
        df = add_weather_features(df, weather, group_col)

    # Store popularity
    if group_col:
        df = add_store_popularity(df, group_col)

    # ═══════════════════════════════════════════════════════════
    # NEW: PEAK, VOLATILITY, MOMENTUM FEATURES
    # These are KEY for predicting high-revenue days!
    # ═══════════════════════════════════════════════════════════
    if group_col:
        df = add_peak_features(df, target, group_col)
        df = add_volatility_features(df, target, group_col)
        df = add_momentum_features(df, target, group_col)
    else:
        # Global version (no group_col)
        df = add_peak_features_global(df, target)
        df = add_volatility_features_global(df, target)
        df = add_momentum_features_global(df, target)

    # Trend
    df = df.sort_values(["date"] + ([group_col] if group_col else []))
    if group_col:
        df["trend"] = df.groupby(group_col).cumcount()
        df["ma_7"] = df.groupby(group_col)[target].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df["ma_14"] = df.groupby(group_col)[target].transform(lambda x: x.rolling(14, min_periods=1).mean())
    else:
        df["trend"] = np.arange(len(df))
        df["ma_7"] = df[target].rolling(7, min_periods=1).mean()
        df["ma_14"] = df[target].rolling(14, min_periods=1).mean()
    df["ma_diff"] = df["ma_7"] - df["ma_14"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target:  # Don't modify target
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    df = df.fillna(method="bfill").fillna(0)
    return df


# Global versions (when no group_col)
def add_peak_features_global(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    """Global version of peak features (no grouping)."""
    df = df.copy()
    df = df.sort_values("date")
    
    dow = df["date"].dt.dayofweek
    df["dow_hist_max"] = df.groupby(dow)[target].transform("max")
    df["dow_hist_p90"] = df.groupby(dow)[target].transform(lambda x: x.quantile(0.90) if len(x) > 3 else x.max())
    df["dow_hist_mean"] = df.groupby(dow)[target].transform("mean")
    df["upside_to_max"] = (df["dow_hist_max"] - df[target]) / (df["dow_hist_max"] + 1e-6)
    df["upside_to_p90"] = (df["dow_hist_p90"] - df[target]) / (df["dow_hist_p90"] + 1e-6)
    df["dow_ratio"] = df[target] / (df["dow_hist_mean"] + 1e-6)
    
    threshold = df[target].quantile(0.80)
    df["hist_peak_dow"] = (df[target] >= threshold).astype(int)
    
    df["days_to_month_end"] = (df["date"].dt.days_in_month - df["date"].dt.day)
    df["is_month_end"] = (df["days_to_month_end"] <= 5).astype(int)
    df["week_of_month"] = ((df["date"].dt.day - 1) // 7) + 1
    
    return df

def add_volatility_features_global(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    """Global version of volatility features."""
    df = df.copy()
    df["cv_14"] = df[target].rolling(14, min_periods=3).std() / (df[target].rolling(14, min_periods=3).mean() + 1e-6)
    df["range_14"] = df[target].rolling(14, min_periods=3).max() - df[target].rolling(14, min_periods=3).min()
    df["gap_from_14d_max"] = df[target].rolling(14, min_periods=3).max() - df[target]
    df["accel_3d"] = df[target] - df[target].shift(3)
    df["accel_7d"] = df[target] - df[target].shift(7)
    return df.fillna(0)

def add_momentum_features_global(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    """Global version of momentum features."""
    df = df.copy()
    df["mom_3d"] = df[target].pct_change(3).fillna(0)
    df["mom_7d"] = df[target].pct_change(7).fillna(0)
    df["mom_accel"] = df["mom_3d"] - df["mom_7d"]
    return df


def feature_columns(df: pd.DataFrame, target: str = "revenue", exclude=("date", "store_id", "store_name", "address", "coord_x", "coord_y")) -> List[str]:
    excl = set(exclude) | {target}
    return [c for c in df.columns if c not in excl and df[c].dtype != object]