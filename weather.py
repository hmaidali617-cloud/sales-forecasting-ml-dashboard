"""
weather.py
==========
NASA POWER API integration with caching.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

from utils import CACHE_DIR, get_logger

logger = get_logger("weather")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

WEATHER_CACHE = os.path.join(CACHE_DIR, "weather_cache.json")


def _load_cache() -> Dict:
    if os.path.exists(WEATHER_CACHE):
        try:
            with open(WEATHER_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict) -> None:
    try:
        with open(WEATHER_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, default=str)
    except Exception as e:
        logger.warning(f"Could not save weather cache: {e}")


def fetch_weather_for_point(lat: float, lon: float, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch daily weather for a single point."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    key = f"{round(float(lat),4)}_{round(float(lon),4)}_{start}_{end}"
    cache = _load_cache()
    if key in cache:
        return pd.DataFrame(cache[key])

    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS10M,RH2M",
        "community": "AG",
        "longitude": round(float(lon), 4),
        "latitude": round(float(lat), 4),
        "start": start,
        "end": end,
        "format": "JSON",
    }
    for attempt in range(3):
        try:
            logger.info(f"NASA POWER request: lat={lat}, lon={lon}, {start}->{end}")
            r = requests.get(POWER_URL, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                props = data.get("properties", {}).get("parameter", {})
                if not props:
                    logger.warning("No weather parameters returned")
                    return None
                df = pd.DataFrame(props)
                if df.empty:
                    return None
                df.index.name = "date"
                df = df.reset_index()
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
                df = df.dropna(subset=["date"])
                rename = {
                    "T2M": "temp_avg",
                    "T2M_MAX": "temp_max",
                    "T2M_MIN": "temp_min",
                    "PRECTOTCORR": "rain",
                    "WS10M": "wind_speed",
                    "RH2M": "humidity",
                }
                df = df.rename(columns=rename)
                for c in ["temp_avg", "temp_max", "temp_min", "rain", "wind_speed", "humidity"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                cache[key] = df.to_dict(orient="records")
                _save_cache(cache)
                return df
            elif r.status_code == 422:
                logger.warning(f"NASA POWER 422 (date range too far in future?) lat={lat}, lon={lon}")
                return None
            else:
                logger.warning(f"NASA POWER status {r.status_code}")
                time.sleep(2)
        except Exception as e:
            logger.warning(f"NASA POWER error attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


# weather.py - Replace fetch_weather_for_stores function
def fetch_weather_for_stores(
    stores: pd.DataFrame,
    start_date,
    end_date
) -> pd.DataFrame:
    """Fetch weather for each unique store location."""

    if (
        stores.empty
        or "coord_x" not in stores.columns
        or "coord_y" not in stores.columns
    ):
        return pd.DataFrame()

    # ================================================================
    # Normalize requested dates to timezone-naive timestamps
    # ================================================================
    start = (
        pd.to_datetime(start_date, errors="coerce", utc=True)
        .tz_localize(None)
        .normalize()
    )

    end = (
        pd.to_datetime(end_date, errors="coerce", utc=True)
        .tz_localize(None)
        .normalize()
    )

    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    # ================================================================
    # NASA POWER requires end <= today
    # ================================================================
    today = pd.Timestamp.now().normalize()
    end = min(end, today)

    if end < start:
        return pd.DataFrame()

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    frames = []

    for _, row in stores.iterrows():

        lat = row.get("coord_x")
        lon = row.get("coord_y")

        if pd.isna(lat) or pd.isna(lon):
            continue

        w = fetch_weather_for_point(
            lat,
            lon,
            start_str,
            end_str
        )

        if w is not None and not w.empty:

            # ========================================================
            # CRITICAL FIX:
            # Force weather date to datetime64[ns] WITHOUT timezone
            # ========================================================
            if "date" in w.columns:
                w["date"] = (
                    pd.to_datetime(
                        w["date"],
                        errors="coerce",
                        utc=True
                    )
                    .dt.tz_localize(None)
                    .dt.normalize()
                )

                w = w.dropna(subset=["date"])

            w["store_id"] = row["store_id"]

            frames.append(w)

    if not frames:
        return pd.DataFrame()

    # ================================================================
    # Combine all store weather data
    # ================================================================
    out = pd.concat(
        frames,
        ignore_index=True
    )

    # ================================================================
    # FINAL SAFETY CHECK
    # ================================================================
    if "date" in out.columns:
        out["date"] = (
            pd.to_datetime(
                out["date"],
                errors="coerce",
                utc=True
            )
            .dt.tz_localize(None)
            .dt.normalize()
        )

        out = out.dropna(subset=["date"])

    return out