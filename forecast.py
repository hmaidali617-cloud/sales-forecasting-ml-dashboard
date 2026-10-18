"""
forecast.py
===========
7-day revenue forecast with prediction intervals using top-3 models.
With BOLD prediction adjustments for high-revenue days.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engineering import build_features, feature_columns
from models import lstm_predict_recursive
from utils import get_logger

logger = get_logger("forecast")


def apply_peak_boost(
    prediction: float, 
    history_df: pd.DataFrame, 
    target: str = "revenue",
    date: pd.Timestamp = None
) -> float:
    """
    Final post-processing: boost prediction if conditions suggest a peak day.
    This is a safety net to ensure predictions can reach historical highs.
    """
    if len(history_df) < 30:
        return prediction
    
    day_of_week = date.dayofweek if date else history_df["date"].iloc[-1].dayofweek
    
    # Get historical stats for this day of week
    hist_dow = history_df[history_df["date"].dt.dayofweek == day_of_week][target]
    
    if len(hist_dow) < 3:
        return prediction
    
    hist_max = hist_dow.max()
    hist_p90 = hist_dow.quantile(0.90)
    hist_mean = hist_dow.mean()
    hist_std = hist_dow.std()
    
    # Conditions that suggest a peak day:
    boost = 1.0
    reasons = []
    
    # 1. Recent trend is upward
    recent_3d = history_df[target].tail(3).mean()
    recent_7d = history_df[target].tail(7).mean()
    if recent_3d > recent_7d * 1.05:
        boost += 0.05
        reasons.append("upward_trend")
    
    # 2. We're due for a peak (gap from max is large)
    current = history_df[target].iloc[-1]
    gap_ratio = (hist_max - current) / (hist_max + 1e-6)
    if gap_ratio > 0.3:
        boost += 0.08
        reasons.append("due_for_peak")
    
    # 3. Low volatility recently but historically high volatility
    recent_cv = history_df[target].tail(7).std() / (history_df[target].tail(7).mean() + 1e-6)
    hist_cv = hist_dow.std() / (hist_dow.mean() + 1e-6)
    if recent_cv < hist_cv * 0.7 and hist_cv > 0.15:
        boost += 0.05
        reasons.append("volatility_squeeze")
    
    # 4. Month-end effect
    if date and date.day >= 25:
        month_end_days = history_df[history_df["date"].dt.day >= 25][target]
        if len(month_end_days) > 3 and month_end_days.mean() > hist_mean * 1.1:
            boost += 0.07
            reasons.append("month_end_effect")
    
    # 5. Hard floor: prediction should not be below 60% of historical mean for this DOW
    min_prediction = hist_mean * 0.60
    if prediction < min_prediction:
        prediction = min_prediction
    
    # 6. Hard ceiling: prediction CAN reach up to 95% of historical max if conditions are right
    max_prediction = hist_max * 0.95
    boosted = prediction * boost
    if boosted > max_prediction and len(reasons) >= 2:
        boosted = max_prediction
    
    if boost > 1.0:
        logger.debug(f"Peak boost: {boost:.3f} (reasons: {reasons})")
    
    return boosted


def calculate_boldness_factor(history_df: pd.DataFrame, target: str = "revenue", horizon: int = 7) -> float:
    """
    Calculate a boldness factor based on recent trend and volatility.
    Returns a multiplier > 1.0 when trend is upward, < 1.0 when downward.
    """
    if len(history_df) < 14:
        return 1.0
    
    recent_7 = history_df[target].tail(7).mean()
    prev_7 = history_df[target].iloc[-14:-7].mean()
    
    if prev_7 > 0:
        trend_ratio = recent_7 / prev_7
    else:
        trend_ratio = 1.0
    
    # Calculate volatility (coefficient of variation)
    recent_std = history_df[target].tail(7).std()
    recent_mean = history_df[target].tail(7).mean()
    cv = recent_std / recent_mean if recent_mean > 0 else 0
    
    # Boldness factor: amplify upward trends, especially for volatile series
    if trend_ratio > 1.0:
        boldness = 1.0 + (trend_ratio - 1.0) * 1.5
        if cv > 0.15:
            boldness += 0.05 * cv
    elif trend_ratio < 1.0:
        boldness = 1.0 + (trend_ratio - 1.0) * 0.7
    else:
        boldness = 1.0
    
    # Clamp to reasonable range
    boldness = np.clip(boldness, 0.85, 1.25)
    
    return boldness


def adjust_for_high_values(prediction: float, history_df: pd.DataFrame, target: str = "revenue") -> float:
    """
    Boost predictions that are below recent high values.
    Prevents under-prediction of peak days.
    """
    if len(history_df) < 7:
        return prediction
    
    recent_max = history_df[target].tail(14).max()
    recent_mean = history_df[target].tail(14).mean()
    recent_p90 = history_df[target].tail(30).quantile(0.90) if len(history_df) >= 30 else recent_max
    
    # If prediction is significantly below the 90th percentile, boost it
    if prediction < recent_p90 * 0.85 and recent_p90 > recent_mean * 1.1:
        gap = recent_p90 - prediction
        boost = gap * 0.4
        prediction = prediction + boost
    
    # Never predict below 70% of recent mean
    min_prediction = recent_mean * 0.70
    if prediction < min_prediction:
        prediction = min_prediction
    
    return prediction


def predict_next_days(
    full_featured: pd.DataFrame,
    fitted_models: Dict,
    feature_cols: List[str],
    target: str = "revenue",
    horizon: int = 7,
    top_model_names: Optional[List[str]] = None,
    boldness_override: Optional[float] = None,
) -> pd.DataFrame:
    """Predict next `horizon` days using top models (ensemble average) with BOLD adjustments."""
    
    DL_MODELS = {"LSTM", "Bi-LSTM", "GRU", "CNN-LSTM", "Attention-LSTM"}
    
    if full_featured.empty:
        raise RuntimeError("No data for forecasting")

    if top_model_names is None:
        top_model_names = list(fitted_models.keys())[:3]

    history_df = full_featured.copy()
    history_df["date"] = pd.to_datetime(history_df["date"])
    last_date = history_df["date"].max()
    
    # Ensure target exists in history
    if target not in history_df.columns:
        history_df[target] = 0.0
        
    # Calculate boldness factor from historical trend
    if boldness_override is not None:
        boldness_factor = boldness_override
    else:
        boldness_factor = calculate_boldness_factor(history_df, target, horizon)
    
    logger.info(f"Boldness factor applied: {boldness_factor:.3f}")
    
    # Get the last historical row to use as a template for future rows
    last_row = history_df.iloc[-1:].copy()
    
    # Store predictions
    future_predictions = []

    for h in range(1, horizon + 1):
        new_date = last_date + pd.Timedelta(days=h)
        new_row = last_row.copy()
        new_row["date"] = new_date
        
        # 1. Update Calendar Features
        d = pd.Timestamp(new_date)
        new_row["year"] = d.year
        new_row["month"] = d.month
        new_row["day"] = d.day
        new_row["quarter"] = d.quarter
        new_row["week"] = d.isocalendar().week
        new_row["dayofweek"] = d.dayofweek
        new_row["dayofyear"] = d.dayofyear
        new_row["weekend"] = int(d.dayofweek >= 5)
        new_row["is_month_start"] = int(d.is_month_start)
        new_row["is_month_end"] = int(d.is_month_end)
        new_row["month_sin"] = np.sin(2 * np.pi * new_row["month"] / 12)
        new_row["month_cos"] = np.cos(2 * np.pi * new_row["month"] / 12)
        new_row["dow_sin"] = np.sin(2 * np.pi * new_row["dayofweek"] / 7)
        new_row["dow_cos"] = np.cos(2 * np.pi * new_row["dayofweek"] / 7)
        new_row["doy_sin"] = np.sin(2 * np.pi * new_row["dayofyear"] / 365)
        new_row["doy_cos"] = np.cos(2 * np.pi * new_row["dayofyear"] / 365)
        
        # Peak features calendar updates
        new_row["days_to_month_end"] = (d.days_in_month - d.day)
        new_row["is_month_end"] = int(d.days_in_month - d.day <= 5)
        new_row["is_month_start"] = int(d.day <= 5)
        new_row["week_of_month"] = ((d.day - 1) // 7) + 1
        new_row["week_of_month_sin"] = np.sin(2 * np.pi * new_row["week_of_month"] / 5)
        new_row["week_of_month_cos"] = np.cos(2 * np.pi * new_row["week_of_month"] / 5)
        
        # Holiday
        if "is_holiday" in new_row.columns:
            new_row["is_holiday"] = 0
        
        # 2. Update Lag Features
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            col_name = f"lag_{lag}"
            if col_name in new_row.columns:
                if len(history_df) >= lag:
                    new_row[col_name] = history_df[target].iloc[-lag]
                else:
                    new_row[col_name] = history_df[target].iloc[-1] if len(history_df) > 0 else 0.0
                    
        # 3. Update Rolling Features
        for w in [7, 14]:
            mean_col = f"rolling_mean_{w}"
            std_col = f"rolling_std_{w}"
            if mean_col in new_row.columns:
                vals = history_df[target].tail(w).values
                new_row[mean_col] = np.mean(vals) if len(vals) > 0 else 0.0
            if std_col in new_row.columns:
                vals = history_df[target].tail(w).values
                new_row[std_col] = np.std(vals) if len(vals) > 1 else 0.0
                
        # 4. Update Expanding Mean
        if "expanding_mean" in new_row.columns:
            new_row["expanding_mean"] = history_df[target].mean()
            
        # 5. Update Trend & Moving Averages
        if "trend" in new_row.columns:
            new_row["trend"] = history_df["trend"].iloc[-1] + 1 if "trend" in history_df.columns else h
        if "ma_7" in new_row.columns:
            new_row["ma_7"] = history_df[target].tail(7).mean()
        if "ma_14" in new_row.columns:
            new_row["ma_14"] = history_df[target].tail(14).mean()
        if "ma_diff" in new_row.columns:
            new_row["ma_diff"] = new_row["ma_7"].iloc[0] - new_row["ma_14"].iloc[0]
            
        # 6. Update Peak/Volatility/Momentum Features
        dow = d.dayofweek
        hist_dow_vals = history_df[history_df["date"].dt.dayofweek == dow][target]
        
        if "dow_hist_max" in new_row.columns:
            new_row["dow_hist_max"] = hist_dow_vals.max() if len(hist_dow_vals) > 0 else 0.0
        if "dow_hist_p90" in new_row.columns:
            new_row["dow_hist_p90"] = hist_dow_vals.quantile(0.90) if len(hist_dow_vals) > 3 else (hist_dow_vals.max() if len(hist_dow_vals) > 0 else 0.0)
        if "dow_hist_mean" in new_row.columns:
            new_row["dow_hist_mean"] = hist_dow_vals.mean() if len(hist_dow_vals) > 0 else 0.0
        if "upside_to_max" in new_row.columns and "dow_hist_max" in new_row.columns:
            new_row["upside_to_max"] = (new_row["dow_hist_max"] - history_df[target].iloc[-1]) / (new_row["dow_hist_max"] + 1e-6)
        if "upside_to_p90" in new_row.columns and "dow_hist_p90" in new_row.columns:
            new_row["upside_to_p90"] = (new_row["dow_hist_p90"] - history_df[target].iloc[-1]) / (new_row["dow_hist_p90"] + 1e-6)
        if "dow_ratio" in new_row.columns and "dow_hist_mean" in new_row.columns:
            new_row["dow_ratio"] = history_df[target].iloc[-1] / (new_row["dow_hist_mean"] + 1e-6)
            
        if "cv_14" in new_row.columns:
            vals = history_df[target].tail(14).values
            new_row["cv_14"] = np.std(vals) / (np.mean(vals) + 1e-6) if len(vals) > 1 else 0.0
        if "range_14" in new_row.columns:
            vals = history_df[target].tail(14).values
            new_row["range_14"] = (np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
        if "gap_from_14d_max" in new_row.columns:
            vals = history_df[target].tail(14).values
            new_row["gap_from_14d_max"] = (np.max(vals) - history_df[target].iloc[-1]) if len(vals) > 1 else 0.0
        if "accel_3d" in new_row.columns:
            new_row["accel_3d"] = history_df[target].iloc[-1] - (history_df[target].iloc[-4] if len(history_df) > 3 else history_df[target].iloc[-1])
        if "accel_7d" in new_row.columns:
            new_row["accel_7d"] = history_df[target].iloc[-1] - (history_df[target].iloc[-8] if len(history_df) > 7 else history_df[target].iloc[-1])
        if "mom_3d" in new_row.columns:
            prev = history_df[target].iloc[-4] if len(history_df) > 3 else history_df[target].iloc[-1]
            new_row["mom_3d"] = (history_df[target].iloc[-1] - prev) / (prev + 1e-6)
        if "mom_7d" in new_row.columns:
            prev = history_df[target].iloc[-8] if len(history_df) > 7 else history_df[target].iloc[-1]
            new_row["mom_7d"] = (history_df[target].iloc[-1] - prev) / (prev + 1e-6)
        if "mom_accel" in new_row.columns:
            new_row["mom_accel"] = new_row.get("mom_3d", 0) - new_row.get("mom_7d", 0)
        if "position_in_range" in new_row.columns:
            vals = history_df[target].tail(14).values
            rng = np.max(vals) - np.min(vals)
            new_row["position_in_range"] = (history_df[target].iloc[-1] - np.min(vals)) / (rng + 1e-6) if rng > 0 else 0.5
        if "hist_peak_dow" in new_row.columns:
            threshold = hist_dow_vals.quantile(0.80) if len(hist_dow_vals) > 3 else hist_dow_vals.max() if len(hist_dow_vals) > 0 else 1e10
            new_row["hist_peak_dow"] = int(history_df[target].iloc[-1] >= threshold)
        if "days_since_peak" in new_row.columns:
            peaks = history_df[history_df[target] >= history_df[target].quantile(0.80)].index
            if len(peaks) > 0:
                last_peak_pos = history_df.index.get_loc(peaks[-1])
                new_row["days_since_peak"] = len(history_df) - 1 - last_peak_pos
            else:
                new_row["days_since_peak"] = 999
        if "consecutive_low_days" in new_row.columns:
            threshold_low = history_df[target].quantile(0.40)
            count = 0
            for i in range(len(history_df) - 1, -1, -1):
                if history_df[target].iloc[i] <= threshold_low:
                    count += 1
                else:
                    break
            new_row["consecutive_low_days"] = count
            
        # 7. Ensure all feature columns exist
        for col in feature_cols:
            if col not in new_row.columns:
                new_row[col] = 0.0
                
        # 8. Predict with Top Models
        X_future = new_row[feature_cols].values.astype(np.float32)
        
        # ═══════════════════════════════════════════════════════════
        # CLEAN: Remove infinite/NaN values before prediction
        # ═══════════════════════════════════════════════════════════
        X_future = np.where(np.isinf(X_future), 0, X_future)
        X_future = np.where(np.isnan(X_future), 0, X_future)
        X_future = np.clip(X_future, -1e6, 1e6)
        
        per_model_preds = {}
        
        for name in top_model_names:
            if name not in fitted_models:
                continue
            m = fitted_models[name]
            try:
                if name in DL_MODELS:
                    dl_model, scaler, lookback = m
                    hist = history_df[target].tail(lookback).values.astype(np.float32)
                    hist_scaled = scaler.transform(hist.reshape(-1, 1)).flatten()
                    if len(hist_scaled) < lookback:
                        pad_value = hist_scaled.mean()
                        hist_scaled = np.pad(hist_scaled, (lookback - len(hist_scaled), 0), mode='constant', constant_values=pad_value)
                    x_input = hist_scaled[-lookback:].reshape(1, lookback, 1)
                    p = dl_model.predict(x_input, verbose=0)[0, 0]
                    p = scaler.inverse_transform([[p]])[0, 0]
                else:
                    p = m.predict(X_future)[0]
                p = max(0.0, float(p))
                per_model_preds[name] = p
            except Exception as e:
                logger.warning(f"Forecast with {name} failed at step {h}: {e}")
                
        if not per_model_preds:
            raise RuntimeError(f"No forecast could be produced at step {h}")
            
        # 9. Calculate Ensemble - Use weighted average favoring higher predictions
        pred_values = list(per_model_preds.values())
        
        weights = np.array(pred_values)
        weights = weights / (weights.sum() + 1e-6)
        ensemble = float(np.sum(weights * np.array(pred_values)))
        
        # Apply boldness factor
        ensemble_before_bold = ensemble
        ensemble = ensemble * boldness_factor
        
        # Adjust for high-value days (prevent under-prediction)
        ensemble = adjust_for_high_values(ensemble, history_df, target)
        
        # Apply Peak Boost
        ensemble = apply_peak_boost(ensemble, history_df, target, new_date)
        
        logger.info(f"Day {h}: Raw={ensemble_before_bold:.2f}, After Boldness={ensemble:.2f}")
        
        # 10. Append prediction to history so next iteration uses it for lags
        new_row[target] = ensemble
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        future_predictions.append({
            "date": new_date,
            "predicted_revenue": ensemble,
            "raw_ensemble": ensemble_before_bold,
            **{f"pred_{k}": v for k, v in per_model_preds.items()}
        })
        
    # Create final dataframe
    out = pd.DataFrame(future_predictions)
    
    # Calculate prediction intervals across models
    pred_arrays = np.vstack([out[f"pred_{name}"].values for name in top_model_names if f"pred_{name}" in out.columns])
    std = np.std(pred_arrays, axis=0) if len(pred_arrays) > 1 else out["predicted_revenue"].values * 0.1
    
    # Widen intervals for bold predictions
    std = std * 1.2
    
    out["prediction_lower"] = np.clip(out["predicted_revenue"] - 1.96 * std, 0, None)
    out["prediction_upper"] = out["predicted_revenue"] + 1.96 * std
    
    # Trend
    out["trend"] = np.where(out["predicted_revenue"] > out["predicted_revenue"].shift(1), "↑",
                            np.where(out["predicted_revenue"] < out["predicted_revenue"].shift(1), "↓", "→"))
    out.loc[out.index[0], "trend"] = "→"
    
    # Confidence based on std relative to mean
    mean_pred = out["predicted_revenue"].mean()
    out["confidence"] = np.clip(1 - (std / (mean_pred + 1e-6)), 0.5, 0.99)
    
    # Add boldness info
    out["boldness_factor"] = boldness_factor
    
    return out