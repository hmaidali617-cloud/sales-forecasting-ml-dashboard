"""
dashboard.py
============
Dashboard UI helpers and Plotly visualizations.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"

PALETTE = px.colors.qualitative.Plotly


def _base_layout(fig, title=""):
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_revenue_time_series(df: pd.DataFrame, title="Daily Revenue") -> go.Figure:
    fig = px.line(df, x="date", y="revenue", title=title)
    return _base_layout(fig)


def plot_train_test_forecast(train: pd.DataFrame, test: pd.DataFrame, forecast: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["revenue"], name="Train", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=test["date"], y=test["revenue"], name="Test (Actual)", line=dict(color="#EF553B")))
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["predicted_revenue"], name="Forecast", line=dict(color="#00CC96", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(forecast["date"]) + list(forecast["date"][::-1]),
            y=list(forecast["prediction_upper"]) + list(forecast["prediction_lower"][::-1]),
            fill="toself", fillcolor="rgba(0,204,150,0.2)", line=dict(color="rgba(0,0,0,0)"),
            name="95% Interval", hoverinfo="skip"
        ))
    fig = _base_layout(fig, "Train / Test / Forecast")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Revenue")
    return fig


def plot_predictions_vs_actual(test_df: pd.DataFrame, preds: Dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["revenue"], name="Actual", line=dict(color="white", width=2)))
    colors = PALETTE[: len(preds)]
    for (name, p), c in zip(preds.items(), colors):
        fig.add_trace(go.Scatter(x=test_df["date"], y=p, name=name, line=dict(color=c, dash="dot")))
    fig = _base_layout(fig, "Predictions vs Actual")
    return fig


def plot_residuals(test_df: pd.DataFrame, preds: Dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for (name, p), c in zip(preds.items(), PALETTE):
        resid = test_df["revenue"].values - p
        fig.add_trace(go.Scatter(x=test_df["date"], y=resid, mode="markers", name=name, marker=dict(color=c, size=6)))
    fig = _base_layout(fig, "Residuals over Time")
    fig.add_hline(y=0, line=dict(color="white", dash="dash"))
    return fig


def plot_loss_curve(history: Dict) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No loss history available")
        return _base_layout(fig)
    for name, h in history.items():
        if not h:
            continue
        try:
            if isinstance(h, dict) and "validation_0" in h:
                losses = h["validation_0"].get("rmse", h["validation_0"].get("logloss", []))
                fig.add_trace(go.Scatter(y=losses, name=f"{name} val"))
            elif isinstance(h, dict) and "loss" in h:
                fig.add_trace(go.Scatter(y=h["loss"], name=f"{name} train"))
                if "val_loss" in h:
                    fig.add_trace(go.Scatter(y=h["val_loss"], name=f"{name} val"))
        except Exception:
            pass
    fig = _base_layout(fig, "Training Loss Curves")
    fig.update_xaxes(title="Epoch/Iteration")
    fig.update_yaxes(title="Loss")
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, title="Feature Importance") -> go.Figure:
    if fi_df is None or fi_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No feature importance available")
        return _base_layout(fig)
    fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title=title)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _base_layout(fig)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    if not cols:
        fig = go.Figure()
        fig.update_layout(title="No numeric columns")
        return _base_layout(fig)
    sub = df[cols].select_dtypes(include=[np.number]).corr()
    fig = px.imshow(sub, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig = _base_layout(fig, "Correlation Heatmap")
    fig.update_layout(height=600)
    return fig


def plot_sales_by_store(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("store_name", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(20)
    fig = px.bar(agg, x="revenue", y="store_name", orientation="h", title="Top Stores by Revenue")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _base_layout(fig)


def plot_sales_by_month(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    agg = d.groupby("month", as_index=False)["revenue"].sum()
    fig = px.bar(agg, x="month", y="revenue", title="Revenue by Month")
    return _base_layout(fig)


def plot_weather_vs_sales(df: pd.DataFrame) -> go.Figure:
    cols_avail = [c for c in ["temp_avg", "rain", "wind_speed", "humidity"] if c in df.columns]
    if not cols_avail:
        fig = go.Figure()
        fig.update_layout(title="No weather data available")
        return _base_layout(fig)
    fig = make_subplots_weather(df, cols_avail)
    return fig


def make_subplots_weather(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
    for i, c in enumerate(cols, start=1):
        fig.add_trace(go.Scatter(x=df["date"], y=df[c], name=c, line=dict(color=PALETTE[i % len(PALETTE)])), row=i, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["revenue"], name="revenue", yaxis=f"y{i}", line=dict(color="white", dash="dot", width=1)), row=i, col=1)
    fig = _base_layout(fig, "Weather Variables vs Revenue")
    fig.update_layout(height=200 * len(cols) + 100)
    return fig


def plot_model_comparison(metrics_list) -> go.Figure:
    if not metrics_list:
        fig = go.Figure()
        fig.update_layout(title="No metrics")
        return _base_layout(fig)
    df = pd.DataFrame([m.as_dict() for m in metrics_list])
    fig = px.bar(df, x="model_name", y=["rmse", "mae"], barmode="group", title="Model Comparison: RMSE & MAE")
    return _base_layout(fig)


def plot_metrics_table(metrics_list) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(sorted(metrics_list, key=lambda x: x.rmse), start=1):
        rows.append({
            "Rank": i,
            "Model": m.model_name,
            "MAE": round(m.mae, 2),
            "RMSE": round(m.rmse, 2),
            "MAPE %": round(m.mape, 2) if not np.isnan(m.mape) else "-",
            "SMAPE %": round(m.smape, 2) if not np.isnan(m.smape) else "-",
            "R²": round(m.r2, 4),
            "Train (s)": round(m.train_time_sec, 2),
            "Inference (s)": round(m.inference_time_sec, 4),
        })
    return pd.DataFrame(rows)


def plot_leaderboard(metrics_list) -> go.Figure:
    if not metrics_list:
        fig = go.Figure()
        fig.update_layout(title="No metrics")
        return _base_layout(fig)
    df = pd.DataFrame([m.as_dict() for m in metrics_list])
    df = df.sort_values("rmse")
    fig = px.bar(df, x="model_name", y="r2", color="rmse", title="Model Leaderboard (R²)", color_continuous_scale="Viridis")
    return _base_layout(fig)


def plot_product_forecast(product_df: pd.DataFrame, model_name: str) -> go.Figure:
    if product_df is None or product_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No product forecast for {model_name}")
        return _base_layout(fig)
    fig = px.bar(product_df.head(15), x="predicted_quantity_7d", y="product_name", orientation="h",
                 title=f"Top Products Forecast - {model_name}", color="expected_revenue_7d", color_continuous_scale="Blues")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _base_layout(fig)


def plot_revenue_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="revenue", nbins=40, title="Revenue Distribution")
    return _base_layout(fig)


def plot_boxplot_by_dow(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["dayofweek"] = pd.to_datetime(d["date"]).dt.day_name()
    fig = px.box(d, x="dayofweek", y="revenue", title="Revenue by Day of Week", category_orders={"dayofweek": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]})
    return _base_layout(fig)


def plot_treemap_products(product_df: pd.DataFrame) -> go.Figure:
    if product_df is None or product_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No product data")
        return _base_layout(fig)
    fig = px.treemap(product_df.head(30), path=["category", "product_name"], values="predicted_quantity_7d", title="Product Treemap (Forecasted Qty)")
    return _base_layout(fig)


# dashboard.py - Replace plot_store_map function
def plot_store_map(stores: pd.DataFrame, daily_store: Optional[pd.DataFrame] = None, forecast: Optional[pd.DataFrame] = None) -> Optional[object]:
    """Folium map with stores colored by revenue."""
    try:
        import folium
        from streamlit_folium import st_folium
    except Exception:
        return None
    if stores is None or stores.empty or "coord_x" not in stores.columns:
        return None
    valid = stores.dropna(subset=["coord_x", "coord_y"])
    if valid.empty:
        return None
    center_lat = valid["coord_x"].mean()   # FIXED: coord_x is Latitude
    center_lon = valid["coord_y"].mean()   # FIXED: coord_y is Longitude
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")

    # Revenue per store
    rev_map = {}
    if daily_store is not None and not daily_store.empty:
        rev_map = daily_store.groupby("store_id")["revenue"].sum().to_dict()

    max_rev = max(rev_map.values()) if rev_map else 1.0

    for _, row in valid.iterrows():
        rev = rev_map.get(row["store_id"], 0)
        # Color scale: green->orange->red
        ratio = rev / max_rev if max_rev > 0 else 0
        if ratio > 0.66:
            color = "red"
        elif ratio > 0.33:
            color = "orange"
        else:
            color = "green"
        radius = 8 + 22 * ratio
        popup = f"<b>{row.get('store_name','Store')}</b><br>Revenue: {rev:,.2f}<br>{row.get('address','')}"
        folium.CircleMarker(
            location=[row["coord_x"], row["coord_y"]],  # FIXED: lat, lon order
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
        ).add_to(m)
    return m

def plot_pie_categories(product_df: pd.DataFrame) -> go.Figure:
    if product_df is None or product_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return _base_layout(fig)
    agg = product_df.groupby("category", as_index=False)["predicted_quantity_7d"].sum()
    fig = px.pie(agg, values="predicted_quantity_7d", names="category", title="Forecasted Sales by Category")
    return _base_layout(fig)