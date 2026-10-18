"""
Product-level forecasting module.
Predicts top-selling products for the next 7 days
using the best trained models and historical product distribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from utils import logger, compute_confidence_score, format_number


class ProductForecaster:
    """Forecasts product-level sales for the next 7 days."""

    def __init__(self):
        self.product_forecasts: Dict[str, pd.DataFrame] = {}

    def forecast_products(
        self,
        product_daily: pd.DataFrame,
        revenue_forecasts: Dict[str, pd.DataFrame],
        top_n: int = 20,
        days: int = 7,
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecast product sales by distributing predicted revenue
        across products based on historical proportions.
        """
        if product_daily is None or product_daily.empty:
            logger.warning("No product daily data available for product forecasting")
            return {}

        logger.info("Generating product-level forecasts...")

        # Compute historical product proportions
        product_stats = (
            product_daily.groupby(["product_name", "category"])
            .agg(
                total_qty=("quantity_sold", "sum"),
                total_revenue=("revenue", "sum"),
                avg_unit_price=("unit_price", "first"),
                avg_daily_qty=("quantity_sold", "mean"),
                days_active=("date", "nunique"),
            )
            .reset_index()
        )

        if product_stats.empty:
            return {}

        total_revenue = product_stats["total_revenue"].sum()
        total_qty = product_stats["total_qty"].sum()

        if total_revenue == 0 or total_qty == 0:
            return {}

        product_stats["revenue_share"] = product_stats["total_revenue"] / total_revenue
        product_stats["qty_share"] = product_stats["total_qty"] / total_qty
        product_stats["avg_price"] = product_stats["total_revenue"] / product_stats["total_qty"]
        product_stats["avg_price"] = product_stats["avg_price"].replace([np.inf, 0], np.nan)
        overall_avg_price = product_stats["avg_price"].mean()
        product_stats["avg_price"] = product_stats["avg_price"].fillna(overall_avg_price)

        # Get top N products by revenue
        top_products = product_stats.nlargest(top_n, "total_revenue")

        self.product_forecasts = {}

        for model_name, forecast_df in revenue_forecasts.items():
            daily_predicted_revenue = forecast_df["predicted_revenue"].values
            future_dates = forecast_df["date"].values

            records = []
            for _, prod in top_products.iterrows():
                for day_idx in range(min(days, len(daily_predicted_revenue))):
                    pred_rev = daily_predicted_revenue[day_idx] * prod["revenue_share"]
                    pred_qty = max(1, round(pred_rev / prod["avg_price"]))

                    records.append(
                        {
                            "date": future_dates[day_idx],
                            "product": prod["product_name"],
                            "category": prod["category"],
                            "predicted_quantity": pred_qty,
                            "predicted_revenue": round(pred_rev, 2),
                            "unit_price": round(prod["avg_price"], 2),
                            "revenue_share": round(prod["revenue_share"] * 100, 2),
                            "model": model_name,
                        }
                    )

            if records:
                prod_df = pd.DataFrame(records)
                self.product_forecasts[model_name] = prod_df

        logger.info(
            f"Generated forecasts for {len(top_products)} products "
            f"across {len(self.product_forecasts)} models"
        )
        return self.product_forecasts

    def get_top_products_summary(
        self, model_name: str = "Ensemble", days: int = 7, top_n: int = 20
    ) -> pd.DataFrame:
        """Get summary of top predicted products for a given model."""
        # Try ensemble first, then fall back to best available
        df = self.product_forecasts.get(model_name)
        if df is None:
            for name in self.product_forecasts:
                df = self.product_forecasts[name]
                break

        if df is None:
            return pd.DataFrame()

        summary = (
            df.groupby(["product", "category"])
            .agg(
                total_predicted_qty=("predicted_quantity", "sum"),
                total_predicted_revenue=("predicted_revenue", "sum"),
                avg_daily_qty=("predicted_quantity", "mean"),
                avg_unit_price=("unit_price", "first"),
            )
            .reset_index()
            .nlargest(top_n, "total_predicted_revenue")
        )

        # Add confidence based on revenue share
        max_rev = summary["total_predicted_revenue"].max()
        if max_rev > 0:
            summary["confidence"] = summary["total_predicted_revenue"].apply(
                lambda x: compute_confidence_score(
                    min(x / max_rev + 0.3, 1.0) * 0.85,
                    max(5, 30 - 25 * x / max_rev),
                )
            )
        else:
            summary["confidence"] = "Low"

        return summary

    def get_product_daily_forecast(
        self, model_name: str = "Ensemble", top_n: int = 10
    ) -> pd.DataFrame:
        """Get daily forecast breakdown for top products."""
        df = self.product_forecasts.get(model_name)
        if df is None:
            for name in self.product_forecasts:
                df = self.product_forecasts[name]
                break

        if df is None:
            return pd.DataFrame()

        # Get top products
        top = df.groupby("product")["predicted_revenue"].sum().nlargest(top_n).index
        return df[df["product"].isin(top)].copy()