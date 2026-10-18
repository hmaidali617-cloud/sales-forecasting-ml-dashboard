"""
Weather data module.
Downloads historical weather data from NASA POWER API
using store coordinates. Implements local file caching.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from utils import logger

CACHE_DIR = "weather_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "weather_cache.json")
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

WEATHER_PARAMS = {
    "T2M": "temp_avg",
    "T2M_MAX": "temp_max",
    "T2M_MIN": "temp_min",
    "PRECTOTCORR": "rain",
    "WS2M": "wind_speed",
    "RH2M": "humidity",
}


class WeatherClient:
    """Fetches and caches weather data from NASA POWER API."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "weather_cache.json")
        self._cache: Dict = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load weather cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded weather cache with {len(self._cache)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save weather cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            logger.info("Weather cache saved to disk")
        except IOError as e:
            logger.warning(f"Could not save cache: {e}")

    def _make_cache_key(self, lat: float, lon: float, start: str, end: str) -> str:
        """Generate a unique cache key."""
        return f"{lat:.4f}_{lon:.4f}_{start}_{end}"

    def fetch_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily weather data from NASA POWER API.
        Uses caching to avoid redundant API calls.
        """
        cache_key = self._make_cache_key(lat, lon, start_date, end_date)

        if cache_key in self._cache:
            logger.info(f"Using cached weather data for ({lat:.4f}, {lon:.4f})")
            return pd.DataFrame(self._cache[cache_key])

        params = {
            "start": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
            "latitude": lat,
            "longitude": lon,
            "community": "AG",
            "parameters": ",".join(WEATHER_PARAMS.keys()),
            "format": "JSON",
            "time-standard": "lst",
        }

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Fetching weather for ({lat:.4f}, {lon:.4f}) "
                    f"[{start_date} to {end_date}] (attempt {attempt + 1})"
                )

                response = requests.get(
                    NASA_POWER_URL, params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if "properties" not in data or "parameter" not in data["properties"]:
                    logger.warning("Unexpected API response format")
                    return None

                records = []
                raw_params = data["properties"]["parameter"]

                dates = sorted(
                    raw_params.get("T2M", {}).keys()
                )
                for date_str in dates:
                    record = {"date": pd.Timestamp(date_str)}
                    for api_name, col_name in WEATHER_PARAMS.items():
                        val = raw_params.get(api_name, {}).get(date_str)
                        record[col_name] = float(val) if val is not None else np.nan
                    records.append(record)

                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])

                # Cache the result
                self._cache[cache_key] = df.to_dict(orient="records")
                self._save_cache()

                logger.info(f"Downloaded {len(df)} days of weather data")
                return df

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error("All API retry attempts exhausted")
                    return None
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse API response: {e}")
                return None

        return None

    def fetch_for_stores(
        self,
        stores_df: pd.DataFrame,
        coord_x_col: str,
        coord_y_col: str,
        store_id_col: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch weather for all unique store locations.
        Handles multiple stores at the same location efficiently.
        """
        all_weather = []

        # Group by approximate location to avoid duplicate API calls
        stores_df = stores_df.copy()
        stores_df["_lat_round"] = stores_df[coord_y_col].round(3)
        stores_df["_lon_round"] = stores_df[coord_x_col].round(3)

        unique_locations = stores_df[["_lat_round", "_lon_round"]].drop_duplicates()

        for _, loc_row in unique_locations.iterrows():
            lat = loc_row["_lat_round"]
            lon = loc_row["_lon_round"]

            if pd.isna(lat) or pd.isna(lon):
                continue

            weather_df = self.fetch_weather(lat, lon, start_date, end_date)
            if weather_df is not None:
                weather_df["coord_x"] = lon
                weather_df["coord_y"] = lat
                all_weather.append(weather_df)

        if not all_weather:
            logger.warning("No weather data retrieved for any store")
            return pd.DataFrame()

        result = pd.concat(all_weather, ignore_index=True)
        logger.info(f"Total weather records: {len(result)}")
        return result

    def merge_weather_with_sales(
        self,
        sales_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        coord_x_col: str = "coord_x",
        coord_y_col: str = "coord_y",
    ) -> pd.DataFrame:
        """Merge weather data with daily sales using coordinates and date."""
        if weather_df.empty or sales_df.empty:
            logger.warning("Cannot merge: empty dataframe")
            return sales_df

        # Round coordinates for matching
        merge_weather = weather_df.copy()
        merge_weather["_match_x"] = merge_weather["coord_x"].round(3)
        merge_weather["_match_y"] = merge_weather["coord_y"].round(3)

        merge_sales = sales_df.copy()
        if coord_x_col in merge_sales.columns and coord_y_col in merge_sales.columns:
            merge_sales["_match_x"] = merge_sales[coord_x_col].round(3)
            merge_sales["_match_y"] = merge_sales[coord_y_col].round(3)
            merge_keys = ["date", "_match_x", "_match_y"]
        else:
            # No coordinates in sales: use average weather per day
            daily_weather = (
                merge_weather.groupby("date")
                .agg(
                    temp_avg=("temp_avg", "mean"),
                    temp_max=("temp_max", "mean"),
                    temp_min=("temp_min", "mean"),
                    rain=("rain", "mean"),
                    wind_speed=("wind_speed", "mean"),
                    humidity=("humidity", "mean"),
                )
                .reset_index()
            )
            result = merge_sales.merge(daily_weather, on="date", how="left")
            logger.info("Merged average weather with sales (no store coords)")
            return result

        # Drop weather coord columns before merge
        weather_cols = [
            "date", "_match_x", "_match_y",
            "temp_avg", "temp_max", "temp_min",
            "rain", "wind_speed", "humidity",
        ]
        weather_cols = [c for c in weather_cols if c in merge_weather.columns]
        merge_weather = merge_weather[weather_cols]

        result = merge_sales.merge(merge_weather, on=merge_keys, how="left")
        result = result.drop(columns=["_match_x", "_match_y"], errors="ignore")

        # Fill missing weather with column means
        weather_fill_cols = ["temp_avg", "temp_max", "temp_min", "rain", "wind_speed", "humidity"]
        for col in weather_fill_cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].mean())

        logger.info(f"Merged weather data. Shape: {result.shape}")
        return result


def generate_future_weather(
    historical_weather: pd.DataFrame, future_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Generate plausible weather data for future dates
    using seasonal averages from historical data.
    """
    if historical_weather.empty:
        return pd.DataFrame()

    hw = historical_weather.copy()
    hw["month"] = hw["date"].dt.month
    hw["day_of_year"] = hw["date"].dt.dayofyear

    # Compute seasonal averages with a 15-day window around each day of year
    seasonal_stats = []
    for doy in range(1, 367):
        window = hw[abs(hw["day_of_year"] - doy) <= 15]
        if len(window) > 0:
            seasonal_stats.append(
                {
                    "day_of_year": doy,
                    "temp_avg_mean": window["temp_avg"].mean(),
                    "temp_avg_std": window["temp_avg"].std(),
                    "temp_max_mean": window["temp_max"].mean(),
                    "temp_min_mean": window["temp_min"].mean(),
                    "rain_mean": window["rain"].mean(),
                    "wind_speed_mean": window["wind_speed"].mean(),
                    "humidity_mean": window["humidity"].mean(),
                }
            )

    seasonal_df = pd.DataFrame(seasonal_stats)

    records = []
    np.random.seed(42)
    for date in future_dates:
        doy = date.dayofyear
        # Handle day 366
        doy_idx = min(doy, 366) - 1
        if doy_idx >= len(seasonal_df):
            doy_idx = len(seasonal_df) - 1

        row = seasonal_df.iloc[doy_idx]
        records.append(
            {
                "date": date,
                "temp_avg": max(
                    0, np.random.normal(row["temp_avg_mean"], max(row["temp_avg_std"], 1))
                ),
                "temp_max": row["temp_max_mean"] + np.random.uniform(0, 3),
                "temp_min": row["temp_min_mean"] - np.random.uniform(0, 3),
                "rain": max(0, row["rain_mean"]),
                "wind_speed": max(0, row["wind_speed_mean"]),
                "humidity": np.clip(row["humidity_mean"] + np.random.uniform(-5, 5), 10, 100),
            }
        )

    return pd.DataFrame(records)