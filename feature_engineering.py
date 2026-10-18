"""
Feature engineering module.
Generates temporal, lag, rolling, expanding, trend, and
popularity features for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from utils import logger
from data_loader import TUNISIAN_HOLIDAYS


class FeatureEngineer:
    """Creates features for sales forecasting models."""

    def __init__(self):
        self.feature_columns: List[str] = []

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based temporal features."""
        df = df.copy()
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for day_of_week and month
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Holiday feature
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        df["is_holiday"] = df["date_str"].map(TUNISIAN_HOLIDAYS).notna().astype(int)
        df = df.drop(columns=["date_str"], errors="ignore")

        # Day of year for seasonal patterns
        df["day_of_year"] = df["date"].dt.dayofyear
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        logger.info("Added temporal features")
        return df

    def add_lag_features(
        self, df: pd.DataFrame, target_col: str = "total_revenue"
    ) -> pd.DataFrame:
        """Add lag features for the target variable."""
        df = df.sort_values("date").copy()
        lags = [1, 2, 3, 7, 14, 21, 28]

        for lag in lags:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        logger.info(f"Added {len(lags)} lag features")
        return df

    def add_rolling_features(
        self, df: pd.DataFrame, target_col: str = "total_revenue"
    ) -> pd.DataFrame:
        """Add rolling mean and std features."""
        df = df.sort_values("date").copy()

        for window in [7, 14]:
            df[f"rolling_mean_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
            )

        logger.info("Added rolling features")
        return df

    def add_expanding_features(
        self, df: pd.DataFrame, target_col: str = "total_revenue"
    ) -> pd.DataFrame:
        """Add expanding (cumulative) mean feature."""
        df = df.sort_values("date").copy()
        df["expanding_mean"] = df[target_col].expanding(min_periods=1).mean()
        logger.info("Added expanding features")
        return df

    def add_trend_features(
        self, df: pd.DataFrame, target_col: str = "total_revenue"
    ) -> pd.DataFrame:
        """Add trend and momentum features."""
        df = df.sort_values("date").copy()

        # 7-day trend: percentage change
        df["trend_7d"] = df[target_col].pct_change(periods=7).fillna(0)

        # 14-day trend
        df["trend_14d"] = df[target_col].pct_change(periods=14).fillna(0)

        # Difference from rolling mean
        if "rolling_mean_7" in df.columns:
            df["diff_from_mean_7"] = df[target_col] - df["rolling_mean_7"]
        if "rolling_mean_14" in df.columns:
            df["diff_from_mean_14"] = df[target_col] - df["rolling_mean_14"]

        # Momentum: current vs previous week same day
        df["momentum_7d"] = df[target_col] - df[target_col].shift(7)

        # Acceleration: change in momentum
        df["acceleration"] = df["momentum_7d"].diff(7).fillna(0)

        logger.info("Added trend features")
        return df

    def add_moving_average_features(
        self, df: pd.DataFrame, target_col: str = "total_revenue"
    ) -> pd.DataFrame:
        """Add various moving averages."""
        df = df.sort_values("date").copy()

        # Exponential moving averages
        for span in [7, 14, 28]:
            df[f"ema_{span}"] = df[target_col].ewm(span=span, adjust=False).mean()

        # Weighted moving average (7-day, linearly decreasing weights)
        weights = np.arange(1, 8)
        df["wma_7"] = (
            df[target_col]
            .rolling(window=7, min_periods=1)
            .apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        )

        logger.info("Added moving average features")
        return df

    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and add weather-derived features."""
        df = df.copy()

        weather_cols = ["temp_avg", "temp_max", "temp_min", "rain", "wind_speed", "humidity"]
        existing_weather = [c for c in weather_cols if c in df.columns]

        if not existing_weather:
            logger.info("No weather columns found, skipping weather features")
            return df

        # Temperature range
        if "temp_max" in df.columns and "temp_min" in df.columns:
            df["temp_range"] = df["temp_max"] - df["temp_min"]

        # Temperature bins
        if "temp_avg" in df.columns:
            df["temp_bin"] = pd.cut(
                df["temp_avg"], bins=[-np.inf, 10, 20, 30, np.inf], labels=[0, 1, 2, 3]
            ).astype(float)

        # Rain indicator
        if "rain" in df.columns:
            df["is_rainy"] = (df["rain"] > 0).astype(int)
            df["heavy_rain"] = (df["rain"] > 5).astype(int)

        # Interaction: weekend * weather
        if "weekend" in df.columns:
            if "temp_avg" in df.columns:
                df["weekend_temp"] = df["weekend"] * df["temp_avg"]
            if "rain" in df.columns:
                df["weekend_rain"] = df["weekend"] * df["rain"]

        logger.info(f"Added weather features from {len(existing_weather)} weather columns")
        return df

    def add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier seasonality features."""
        df = df.copy()
        df["t"] = (df["date"] - df["date"].min()).dt.days

        # Weekly seasonality (period = 7)
        for k in range(1, 4):
            df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * df["t"] / 7)
            df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * df["t"] / 7)

        # Monthly seasonality (period ≈ 30.44)
        for k in range(1, 3):
            df[f"monthly_sin_{k}"] = np.sin(2 * np.pi * k * df["t"] / 30.44)
            df[f"monthly_cos_{k}"] = np.cos(2 * np.pi * k * df["t"] / 30.44)

        df = df.drop(columns=["t"], errors="ignore")
        logger.info("Added Fourier seasonality features")
        return df

    def add_popularity_features(
        self,
        df: pd.DataFrame,
        stores_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Add store and product popularity features."""
        df = df.copy()

        if "store_id" in df.columns and df["store_id"].nunique() > 1:
            store_pop = df.groupby("store_id")["total_revenue"].transform("mean")
            df["store_popularity"] = store_pop

        if "order_count" in df.columns:
            df["order_count_lag_7"] = df["order_count"].shift(7)
            df["order_count_rolling_7"] = (
                df["order_count"].rolling(7, min_periods=1).mean()
            )

        logger.info("Added popularity features")
        return df

    def build_all_features(
        self,
        df: pd.DataFrame,
        target_col: str = "total_revenue",
        include_weather: bool = True,
    ) -> pd.DataFrame:
        """
        Build the complete feature set.
        Returns dataframe with all features, ready for modeling.
        """
        logger.info("Building all features...")
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        df = self.add_temporal_features(df)
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_features(df, target_col)
        df = self.add_expanding_features(df, target_col)
        df = self.add_trend_features(df, target_col)
        df = self.add_moving_average_features(df, target_col)
        df = self.add_seasonality_features(df)

        if include_weather:
            df = self.add_weather_features(df)

        df = self.add_popularity_features(df)

        # Drop rows with NaN from lag/rolling features
        initial_len = len(df)
        df = df.dropna(subset=[f"lag_{l}" for l in [1, 2, 3, 7, 14, 21, 28]], how="all")
        df = df.fillna(0)

        # Identify feature columns (exclude date, target, metadata)
        exclude_cols = {
            "date", "total_revenue", "daily_revenue", "order_count",
            "avg_order_value", "store_id", "store_name", "coord_x", "coord_y",
            "address", "temp_bin",
        }
        self.feature_columns = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, float, int]
        ]

        logger.info(
            f"Feature engineering complete. {len(self.feature_columns)} features, "
            f"{initial_len - len(df)} rows dropped due to NaN lags"
        )
        return df

    def prepare_future_features(
        self,
        future_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        target_col: str = "total_revenue",
    ) -> pd.DataFrame:
        """
        Prepare features for future dates using the last available
        historical data for lag/rolling computations.
        """
        # Extract and clean historical data
        hist_cols = ["date", target_col]
        if "order_count" in historical_df.columns:
            hist_cols.append("order_count")
            
        hist_tail = historical_df[hist_cols].copy()
        for col in hist_cols:
            if col in hist_tail.columns and col != "date":
                hist_tail[col] = hist_tail[col].fillna(0)

        # Clean future data - strip timezone to prevent concat type errors
        future_input = future_df[["date"]].copy()
        future_input["date"] = pd.to_datetime(future_input["date"]).dt.tz_localize(None)
        hist_tail["date"] = pd.to_datetime(hist_tail["date"]).dt.tz_localize(None)

        # Combine historical tail with future for proper lag/rolling computation
        combined = pd.concat(
            [hist_tail, future_input],
            ignore_index=True,
        )
        combined[target_col] = combined[target_col].fillna(0)

        # Add lag and rolling features on the combined data
        combined = self.add_lag_features(combined, target_col)
        combined = self.add_rolling_features(combined, target_col)
        combined = self.add_expanding_features(combined, target_col)

        # FIX: Extract future rows simply by taking the last N rows.
        # This completely avoids .isin() which fails on timezone/type mismatches.
        n_future = len(future_df)
        result = combined.tail(n_future).copy()

        # Add temporal, seasonality, and weather features
        result = self.add_temporal_features(result)
        result = self.add_seasonality_features(result)

        # Add weather features directly from future_df
        weather_cols = ["temp_avg", "temp_max", "temp_min", "rain", "wind_speed", "humidity"]
        for col in weather_cols:
            if col in future_df.columns:
                result[col] = future_df[col].values[:n_future]

        if not any(c in result.columns for c in weather_cols):
            result = self.add_weather_features(result)

        # Fill all NaN with 0
        for col in result.columns:
            if col != "date":
                result[col] = result[col].fillna(0)

        return result