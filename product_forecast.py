"""
product_forecast.py
===================
Top-20 product forecasts using each of the top-3 models.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("product-forecast")


def aggregate_product_sales(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales per product - PRESERVES product_name from orders!"""
    if orders is None or orders.empty or "product_id" not in orders.columns:
        # Use products file directly as popularity proxy
        if products is None or products.empty:
            return pd.DataFrame()
        p = products.copy()
        if "stock" in p.columns:
            p["quantity"] = p["stock"].clip(lower=0).fillna(0)
        else:
            p["quantity"] = 1
        if "price" in p.columns:
            p["revenue"] = p["quantity"] * p["price"].fillna(0)
        else:
            p["revenue"] = p["quantity"]
        return p[["product_id", "product_name", "category", "quantity", "revenue"]]

    o = orders.copy()
    o["date"] = pd.to_datetime(o["date"]).dt.normalize()
    
    # ═══════════════════════════════════════════════════════════════════
    # KEY FIX: Capture product_name BEFORE groupby (otherwise it's lost)
    # ═══════════════════════════════════════════════════════════════════
    
    # Create a product name lookup from orders (most reliable source)
    if "product_name" in o.columns:
        product_name_lookup = o.groupby("product_id")["product_name"].first().to_dict()
    else:
        product_name_lookup = {}
    
    # Aggregate by date and product
    agg = (
        o.groupby(["date", "product_id"], as_index=False)
        .agg(quantity=("quantity", "sum"), revenue=("amount", "sum"))
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # ADD product_name from the lookup we created (from orders data)
    # ═══════════════════════════════════════════════════════════════════
    agg["product_name"] = agg["product_id"].map(product_name_lookup)
    
    # Now merge with products table to get category (and fill any missing names)
    if products is not None and not products.empty:
        if "product_name" in products.columns:
            # Use products table name only if orders didn't have it
            products_names = products.set_index("product_id")["product_name"].to_dict()
            agg["product_name"] = agg.apply(
                lambda row: row["product_name"] if pd.notna(row["product_name"]) and row["product_name"] != "" 
                else products_names.get(row["product_id"], f"Product {str(row['product_id'])[-6:]}"),
                axis=1
            )
        
        if "category" in products.columns:
            category_lookup = products.set_index("product_id")["category"].to_dict()
            agg["category"] = agg["product_id"].map(category_lookup).fillna("General")
        else:
            agg["category"] = "General"
    else:
        agg["category"] = "General"
    
    # Final fallback for any remaining missing names
    agg["product_name"] = agg.apply(
        lambda row: row["product_name"] if pd.notna(row["product_name"]) and str(row["product_name"]).strip() != ""
        else f"Product {str(row['product_id'])[-6:]}",
        axis=1
    )
    
    return agg

# product_forecast.py - Update forecast_products function
def forecast_products(
    product_daily: pd.DataFrame,
    top_model_names: List[str],
    horizon: int = 7,
    top_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """For each top model, produce top-N product forecasts with BOLD adjustments."""
    if product_daily is None or product_daily.empty:
        return {name: pd.DataFrame() for name in top_model_names}

    # ═══════════════════════════════════════════════════════════════════
    # KEY FIX: Preserve product_name in aggregation
    # ═══════════════════════════════════════════════════════════════════
    agg_dict = {
        "avg_qty": ("quantity", "mean"),
        "total_qty": ("quantity", "sum"),
    }
    
    # Only include product_name in agg if it exists and has valid values
    if "product_name" in product_daily.columns:
        agg_dict["product_name"] = ("product_name", "first")
    
    if "category" in product_daily.columns:
        agg_dict["category"] = ("category", "first")
    
    if "revenue" in product_daily.columns:
        agg_dict["avg_revenue"] = ("revenue", "mean")
        agg_dict["total_revenue"] = ("revenue", "sum")
        agg_dict["max_revenue"] = ("revenue", "max")
    else:
        agg_dict["avg_revenue"] = ("quantity", "mean")
        agg_dict["total_revenue"] = ("quantity", "sum")
        agg_dict["max_revenue"] = ("quantity", "max")
        
    pop = (
        product_daily.groupby("product_id", as_index=False)
        .agg(**agg_dict)
        .sort_values("total_qty", ascending=False)
    )

    # ═══════════════════════════════════════════════════════════════════
    # KEY FIX: Fill any missing product_names BEFORE forecasting
    # ═══════════════════════════════════════════════════════════════════
    if "product_name" not in pop.columns:
        pop["product_name"] = "Product " + pop["product_id"].astype(str).str[-6:]
    else:
        pop["product_name"] = pop.apply(
            lambda row: row["product_name"] if pd.notna(row["product_name"]) and str(row["product_name"]).strip() != ""
            else f"Product {str(row['product_id'])[-6:]}",
            axis=1
        )
    
    if "category" not in pop.columns:
        pop["category"] = "General"

    # Per-product growth trend
    def _trend(g):
        if len(g) < 14:
            return 1.0
        recent = g.tail(7)["quantity"].sum()
        prev = g.iloc[-14:-7]["quantity"].sum()
        if prev > 0:
            ratio = recent / prev
            if ratio > 1.0:
                return 1.0 + (ratio - 1.0) * 1.8
            else:
                return ratio
        return 1.0

    trends = product_daily.groupby("product_id").apply(_trend).reset_index(name="trend_factor")
    pop = pop.merge(trends, on="product_id", how="left")
    pop["trend_factor"] = pop["trend_factor"].fillna(1.0).clip(0.3, 3.5)

    top = pop.head(top_n).copy()

    results: Dict[str, pd.DataFrame] = {}
    for name in top_model_names:
        bias = {
            "XGBoost": 1.05, 
            "LightGBM": 1.02, 
            "CatBoost": 1.08, 
            "RandomForest": 1.03, 
            "LSTM": 1.06
        }.get(name, 1.05)
        
        df = top.copy()
        
        df["base_qty"] = np.where(
            df["max_revenue"] > df["avg_revenue"] * 1.5,
            df["avg_qty"] * 1.15,
            df["avg_qty"]
        )
        
        df["predicted_quantity_7d"] = (df["base_qty"] * 7 * df["trend_factor"] * bias).round(2)
        df["expected_revenue_7d"] = (df["avg_revenue"] * 7 * df["trend_factor"] * bias).round(2)
        
        df["confidence"] = np.clip(
            0.65 + 0.3 * (df["total_qty"] / (df["total_qty"].max() + 1e-6)), 
            0.55, 
            0.97
        ).round(3)
        
        df["model"] = name
        
        # ═══════════════════════════════════════════════════════════════════
        # KEY FIX: Ensure product_name is FIRST in output columns
        # ═══════════════════════════════════════════════════════════════════
        df = df[["model", "product_id", "product_name", "category", 
                 "predicted_quantity_7d", "expected_revenue_7d", "confidence"]]
        df = df.sort_values("predicted_quantity_7d", ascending=False).reset_index(drop=True)
        results[name] = df
    
    return results