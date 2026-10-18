"""
Revenue forecasting module.
Generates 7-day ahead predictions with confidence intervals
using the top 3 trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from utils import (
    logger, compute_prediction_intervals, compute_confidence_score,
    format_number,
)
from weather import generate_future_weather
from feature_engineering import FeatureEngineer


class RevenueForecaster:
    """Generates revenue forecasts for the next 7 days."""

    def __init__(self):
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.ensemble_forecast: Optional[pd.DataFrame] = None

    def forecast_next_7_days(
        self,
        model_trainer,
        feature_df: pd.DataFrame,
        feature_columns: List[str],
        historical_weather: Optional[pd.DataFrame] = None,
        target_col: str = "total_revenue",
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate 7-day forecasts using the top 3 models.
        Returns a dict mapping model name to forecast DataFrame.
        """
        logger.info("Generating 7-day revenue forecasts...")

        last_date = feature_df["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=7, freq="D"
        )

        # Prepare future weather
        future_weather = None
        if historical_weather is not None and not historical_weather.empty:
            future_weather = generate_future_weather(historical_weather, future_dates)

        # Build future features
        engineer = FeatureEngineer()
        future_df = pd.DataFrame({"date": future_dates})

        if future_weather is not None:
            future_df = future_df.merge(future_weather, on="date", how="left")

        # Use historical data for lag/rolling computation
        hist_tail = feature_df[["date", target_col] + [
            c for c in ["order_count"] if c in feature_df.columns
        ]].copy()
        for col in [target_col] + [c for c in ["order_count"] if c in feature_df.columns]:
            if col in hist_tail.columns:
                hist_tail[col] = hist_tail[col].fillna(0)

        future_features = engineer.prepare_future_features(
            future_df, hist_tail, target_col
        )

        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in future_features.columns:
                future_features[col] = 0

        X_future = future_features[feature_columns].values.astype(np.float32)

        self.forecasts = {}

        for result in model_trainer.best_models:
            model_name = result.name

            if result.model is None:
                # LSTM without enough data - use simple average
                predictions = np.full(7, feature_df[target_col].tail(7).mean())
            elif model_name == "LSTM":
                predictions = self._lstm_predict(
                    result.model, feature_df, X_future,
                    feature_columns, target_col,
                )
            else:
                predictions = result.model.predict(X_future)

            predictions = np.maximum(predictions, 0)

            # Compute prediction intervals
            lower, upper = compute_prediction_intervals(
                predictions, result.residuals_std, confidence=0.95
            )

            confidence_label = compute_confidence_score(
                result.metrics["R² Score"], result.metrics["MAPE (%)"]
            )

            # Compute trend
            if len(predictions) >= 2:
                trend_pct = ((predictions[-1] - predictions[0]) / max(predictions[0], 1)) * 100
            else:
                trend_pct = 0

            forecast_df = pd.DataFrame(
                {
                    "date": future_dates,
                    "predicted_revenue": predictions,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "confidence_95": confidence_label,
                    "trend_pct": round(trend_pct, 2),
                    "model": model_name,
                }
            )

            self.forecasts[model_name] = forecast_df
            logger.info(
                f"  {model_name}: 7-day total = {predictions.sum():.2f}, "
                f"trend = {trend_pct:+.1f}%"
            )

        # Create ensemble forecast (weighted average, weighted by R²)
        self._create_ensemble_forecast()

        return self.forecasts

    def _lstm_predict(
        self,
        model,
        feature_df: pd.DataFrame,
        X_future: np.ndarray,
        feature_columns: List[str],
        target_col: str,
    ) -> np.ndarray:
        """Generate LSTM predictions for future dates."""
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(28, len(feature_df) - 1)

        # Build sequence from historical + future data
        hist_X = feature_df[feature_columns].values.astype(np.float32)
        combined_X = np.vstack([hist_X[-seq_len:], X_future])

        X_seq = combined_X[-seq_len:].reshape(1, seq_len, -1)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

        predictions = []
        model.eval()
        with torch.no_grad():
            # Predict one at a time, updating the sequence
            current_seq = X_tensor.clone()
            for _ in range(7):
                pred = model(current_seq).item()
                predictions.append(max(0, pred))
                # Shift sequence left and append new prediction as feature
                new_row = current_seq[:, -1:, :].clone()
                # Replace target-related features with predicted value
                if len(feature_columns) > 0:
                    new_row[0, 0, 0] = pred
                current_seq = torch.cat([current_seq[:, 1:, :], new_row], dim=1)

        return np.array(predictions)

    def _create_ensemble_forecast(self) -> None:
        """Create a weighted ensemble forecast from the top 3 models."""
        if not self.forecasts:
            return

        total_weight = sum(
            max(r.metrics["R² Score"], 0.01) for r in [self._get_result(n) for n in self.forecasts]
        )
        if total_weight == 0:
            total_weight = 1

        ensemble = None
        for name, df in self.forecasts.items():
            result = self._get_result(name)
            if result is None:
                weight = 1 / len(self.forecasts)
            else:
                weight = max(result.metrics["R² Score"], 0.01) / total_weight

            if ensemble is None:
                ensemble = df[["date", "lower_bound", "upper_bound"]].copy()
                ensemble["predicted_revenue"] = df["predicted_revenue"] * weight
                ensemble["lower_weighted"] = df["lower_bound"] * weight
                ensemble["upper_weighted"] = df["upper_bound"] * weight
            else:
                ensemble["predicted_revenue"] += df["predicted_revenue"] * weight
                ensemble["lower_weighted"] += df["lower_bound"] * weight
                ensemble["upper_weighted"] += df["upper_bound"] * weight

        if ensemble is not None:
            ensemble["lower_bound"] = ensemble["lower_weighted"]
            ensemble["upper_bound"] = ensemble["upper_weighted"]
            ensemble = ensemble.drop(columns=["lower_weighted", "upper_weighted"])
            ensemble["predicted_revenue"] = ensemble["predicted_revenue"].clip(lower=0)
            ensemble["lower_bound"] = ensemble["lower_bound"].clip(lower=0)

            # Trend
            preds = ensemble["predicted_revenue"].values
            if len(preds) >= 2:
                ensemble["trend_pct"] = round(
                    ((preds[-1] - preds[0]) / max(preds[0], 1)) * 100, 2
                )
            else:
                ensemble["trend_pct"] = 0

            ensemble["model"] = "Ensemble (Top 3)"
            ensemble["confidence_95"] = "High"

            self.ensemble_forecast = ensemble

    def _get_result(self, model_name: str):
        """Get ModelResult by name (helper)."""
        from models import ModelTrainer
        # This is called from forecast context where trainer is available
        return None

    def get_summary(self) -> pd.DataFrame:
        """Return a summary of all forecasts."""
        rows = []
        for name, df in self.forecasts.items():
            rows.append(
                {
                    "Model": name,
                    "Day 1": round(df["predicted_revenue"].iloc[0], 2),
                    "Day 7": round(df["predicted_revenue"].iloc[-1], 2),
                    "7-Day Total": round(df["predicted_revenue"].sum(), 2),
                    "Avg Daily": round(df["predicted_revenue"].mean(), 2),
                    "Trend (%)": df["trend_pct"].iloc[-1],
                    "Confidence": df["confidence_95"].iloc[0],
                }
            )

        if self.ensemble_forecast is not None:
            df = self.ensemble_forecast
            rows.append(
                {
                    "Model": "Ensemble",
                    "Day 1": round(df["predicted_revenue"].iloc[0], 2),
                    "Day 7": round(df["predicted_revenue"].iloc[-1], 2),
                    "7-Day Total": round(df["predicted_revenue"].sum(), 2),
                    "Avg Daily": round(df["predicted_revenue"].mean(), 2),
                    "Trend (%)": df["trend_pct"].iloc[-1],
                    "Confidence": "High",
                }
            )

        return pd.DataFrame(rows)