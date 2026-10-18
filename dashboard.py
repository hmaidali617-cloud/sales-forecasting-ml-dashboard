"""
Dashboard rendering module.
Contains all page rendering functions with Plotly visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging

import streamlit as st

from utils import logger, get_dark_theme_css, format_number, compute_confidence_score

# Plotly dark template
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#8B85FF",
    "success": "#3FB950",
    "warning": "#D29922",
    "danger": "#F85149",
    "info": "#58A6FF",
    "gradient": ["#6C63FF", "#3FB950", "#D29922", "#F85149", "#58A6FF"],
}
MODEL_COLORS = {
    "XGBoost": "#6C63FF",
    "LightGBM": "#3FB950",
    "CatBoost": "#D29922",
    "Random Forest": "#F85149",
    "LSTM": "#58A6FF",
    "Ensemble (Top 3)": "#FF6B9D",
    "Ensemble": "#FF6B9D",
}


def _apply_plotly_style(fig: go.Figure, height: int = 450) -> go.Figure:
    """Apply consistent dark styling to a Plotly figure."""
    # Use update_layout instead of update_template for broader compatibility
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="Inter, sans-serif", size=12, color="#8B949E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(28,35,51,0.8)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="#21262D", zerolinecolor="#30363D",
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            gridcolor="#21262D", zerolinecolor="#30363D",
            showgrid=True, gridwidth=1,
        ),
    )
    return fig


def render_home(data_loader) -> None:
    """Render the Home/Overview page."""
    st.markdown('<div class="section-header"><h2>📊 Dashboard Overview</h2></div>', unsafe_allow_html=True)

    if data_loader.df_commands is None:
        st.info("Please upload your CSV files to see the overview.")
        return

    stats = data_loader.get_summary_stats()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div style="color:#8B949E;font-size:0.85rem">Total Revenue</div>'
            f'<div style="color:#3FB950;font-size:1.8rem;font-weight:700">'
            f'{format_number(stats["total_revenue"])}</div>'
            f'<div style="color:#8B949E;font-size:0.75rem">TND</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div style="color:#8B949E;font-size:0.85rem">Total Orders</div>'
            f'<div style="color:#58A6FF;font-size:1.8rem;font-weight:700">'
            f'{format_number(stats["total_commands"], 0)}</div>'
            f'<div style="color:#8B949E;font-size:0.75rem">commands</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div style="color:#8B949E;font-size:0.85rem">Avg Daily Revenue</div>'
            f'<div style="color:#D29922;font-size:1.8rem;font-weight:700">'
            f'{format_number(stats["avg_daily_revenue"])}</div>'
            f'<div style="color:#8B949E;font-size:0.75rem">TND / day</div>'
            f'</div>', unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div style="color:#8B949E;font-size:0.85rem">Date Range</div>'
            f'<div style="color:#F85149;font-size:1.2rem;font-weight:700">'
            f'{stats["date_range"][0]}<br>→ {stats["date_range"][1]}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 📈 Daily Revenue Trend")
        if data_loader.df_daily is not None:
            df = data_loader.df_daily.copy()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["date"], y=df["total_revenue"],
                    mode="lines+markers", name="Daily Revenue",
                    line=dict(color=COLORS["primary"], width=2.5),
                    marker=dict(size=4),
                    fill="tozeroy",
                    fillcolor="rgba(108,99,255,0.1)",
                )
            )
            # 7-day moving average
            if len(df) >= 7:
                df["ma_7"] = df["total_revenue"].rolling(7, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], y=df["ma_7"],
                        mode="lines", name="7-Day MA",
                        line=dict(color=COLORS["warning"], width=2, dash="dash"),
                    )
                )
            fig = _apply_plotly_style(fig, 380)
            fig.update_yaxes(title_text="Revenue (TND)")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 📋 Dataset Summary")
        summary_data = {
            "Metric": [
                "Total Orders", "Stores", "Products",
                "Daily Records", "Missing Values",
                "Avg Order Value",
            ],
            "Value": [
                f"{stats['total_commands']:,}",
                str(stats["total_stores"]),
                str(stats["total_products"]),
                str(stats.get("daily_records", "N/A")),
                str(stats["missing_values"]),
                f"{format_number(stats['total_revenue'] / max(stats['total_commands'], 1))} TND",
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data), hide_index=True,
            use_container_width=True, height=300,
        )

    st.markdown("---")
    st.markdown("#### 🏪 Stores Overview")
    if data_loader.df_stores is not None and data_loader.df_daily is not None:
        store_data = data_loader.get_store_coordinates()
        if not store_data.empty:
            col_map_s, col_name_s = data_loader._col_map.get("store"), data_loader._col_map.get("store_name")
            if col_map_s and data_loader.df_daily is not None:
                store_rev = (
                    data_loader.df_daily.groupby(col_map_s)["daily_revenue"]
                    .sum()
                    .reset_index()
                    .rename(columns={col_map_s: "store_id", "daily_revenue": "total_revenue"})
                )
                store_data = store_data.merge(store_rev, left_on="store_id", right_on="store_id", how="left")
                store_data["total_revenue"] = store_data["total_revenue"].fillna(0)

            for _, row in store_data.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{row.get('store_name', 'Unknown')}**")
                    st.caption(f"📍 {row.get('lat', 'N/A'):.4f}, {row.get('lon', 'N/A'):.4f}")
                with col_b:
                    st.markdown(
                        f'<div style="text-align:right;color:#3FB950;font-weight:600">'
                        f'{format_number(row.get("total_revenue", 0))} TND</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("---")


def render_data_explorer(data_loader) -> None:
    """Render the Data Explorer page."""
    st.markdown('<div class="section-header"><h2>🔍 Data Explorer</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Commands", "Stores", "Products"])

    with tab1:
        if data_loader.df_commands is not None:
            st.markdown(f"**{len(data_loader.df_commands)}** records loaded")
            st.dataframe(data_loader.df_commands.head(500), use_container_width=True, height=400)
        else:
            st.info("No command data loaded")

    with tab2:
        if data_loader.df_stores is not None:
            st.markdown(f"**{len(data_loader.df_stores)}** stores")
            st.dataframe(data_loader.df_stores, use_container_width=True, height=400)
        else:
            st.info("No store data loaded")

    with tab3:
        if data_loader.df_products is not None:
            st.markdown(f"**{len(data_loader.df_products)}** product records")
            st.dataframe(data_loader.df_products.head(500), use_container_width=True, height=400)
        else:
            st.info("No product data loaded")

    # Distribution plots
    if data_loader.df_commands is not None:
        st.markdown("---")
        st.markdown("#### 📊 Order Amount Distribution")
        amounts = data_loader.df_commands[data_loader._col_map["amount"]]
        fig = px.histogram(
            amounts, nbins=50, title="Order Amount Distribution",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig = _apply_plotly_style(fig, 350)
        fig.update_xaxes(title_text="Amount (TND)")
        fig.update_yaxes(title_text="Frequency")
        st.plotly_chart(fig, use_container_width=True)


def render_weather(data_loader, weather_df: Optional[pd.DataFrame]) -> None:
    """Render the Weather page."""
    st.markdown('<div class="section-header"><h2>🌤️ Weather Data</h2></div>', unsafe_allow_html=True)

    if weather_df is None or weather_df.empty:
        st.info("No weather data available. Train models to download weather data.")
        return

    st.markdown(f"**{len(weather_df)}** daily weather records")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Temperature")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["temp_max"],
            name="Max Temp", line=dict(color=COLORS["danger"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["temp_avg"],
            name="Avg Temp", line=dict(color=COLORS["warning"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["temp_min"],
            name="Min Temp", line=dict(color=COLORS["info"], width=1.5),
            fill="tonexty", fillcolor="rgba(88,166,255,0.1)",
        ))
        fig = _apply_plotly_style(fig, 350)
        fig.update_yaxes(title_text="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🌧️ Precipitation")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weather_df["date"], y=weather_df["rain"],
            name="Rain", marker_color=COLORS["info"],
        ))
        fig = _apply_plotly_style(fig, 350)
        fig.update_yaxes(title_text="Rain (mm/day)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 💨 Wind Speed")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["wind_speed"],
            name="Wind Speed", line=dict(color=COLORS["success"], width=2),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.1)",
        ))
        fig = _apply_plotly_style(fig, 350)
        fig.update_yaxes(title_text="Wind Speed (m/s)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### 💧 Humidity")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weather_df["date"], y=weather_df["humidity"],
            name="Humidity", line=dict(color="#A78BFA", width=2),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
        ))
        fig = _apply_plotly_style(fig, 350)
        fig.update_yaxes(title_text="Humidity (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Weather vs Sales correlation
    if data_loader.df_daily is not None:
        st.markdown("---")
        st.markdown("#### 🔗 Weather vs Sales Correlation")
        merged = data_loader.df_merged if data_loader.df_merged is not None else data_loader.df_daily
        if "temp_avg" in merged.columns:
            corr_cols = [c for c in ["temp_avg", "temp_max", "rain", "wind_speed", "humidity", "total_revenue"] if c in merged.columns]
            if len(corr_cols) >= 2:
                corr_matrix = merged[corr_cols].corr()
                fig = px.imshow(
                    corr_matrix, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                )
                fig = _apply_plotly_style(fig, 400)
                st.plotly_chart(fig, use_container_width=True)


def render_model_training(model_trainer, feature_columns: List[str]) -> None:
    """Render the Model Training page."""
    st.markdown('<div class="section-header"><h2>🤖 Model Training</h2></div>', unsafe_allow_html=True)

    if not model_trainer.results:
        st.info("Models have not been trained yet. Click 'Train Models' in the sidebar.")
        return

    # Leaderboard
    st.markdown("#### 🏆 Model Leaderboard")
    metrics_df = model_trainer.get_metrics_dataframe()

    # Style the dataframe
    def _color_rank(val):
        if val == 1:
            return "background: linear-gradient(90deg, rgba(255,215,0,0.2), transparent); font-weight:700; color:#FFD700"
        elif val == 2:
            return "background: linear-gradient(90deg, rgba(192,192,192,0.2), transparent); font-weight:600; color:#C0C0C0"
        elif val == 3:
            return "background: linear-gradient(90deg, rgba(205,127,50,0.2), transparent); font-weight:600; color:#CD7F32"
        return ""

    styled_df = metrics_df.style.map(_color_rank, subset=["Rank"])
    st.dataframe(styled_df, use_container_width=True, height=350)

    # Metrics comparison bar charts
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### RMSE Comparison")
        fig = px.bar(
            metrics_df, x="Model", y="RMSE",
            color="Model", color_discrete_map=MODEL_COLORS,
        )
        fig = _apply_plotly_style(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### R² Score Comparison")
        fig = px.bar(
            metrics_df, x="Model", y="R² Score",
            color="Model", color_discrete_map=MODEL_COLORS,
        )
        fig = _apply_plotly_style(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### MAPE Comparison")
        fig = px.bar(
            metrics_df, x="Model", y="MAPE (%)",
            color="Model", color_discrete_map=MODEL_COLORS,
        )
        fig = _apply_plotly_style(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Training Time")
        fig = px.bar(
            metrics_df, x="Model", y="Training Time (s)",
            color="Model", color_discrete_map=MODEL_COLORS,
        )
        fig = _apply_plotly_style(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # Best model highlight
    best = model_trainer.results[0]
    st.markdown("---")
    st.markdown(
        f'#### 🥇 Best Model: **{best.name}** '
        f'(RMSE: {best.metrics["RMSE"]:.4f}, R²: {best.metrics["R² Score"]:.4f})'
    )


def render_model_comparison(
    model_trainer, feature_df: pd.DataFrame, feature_columns: List[str]
) -> None:
    """Render the Model Comparison page with detailed analysis."""
    st.markdown('<div class="section-header"><h2>⚖️ Model Comparison</h2></div>', unsafe_allow_html=True)

    if not model_trainer.results:
        st.info("Train models first to see comparison.")
        return

    # Feature Importance (from best model)
    st.markdown("#### 📊 Feature Importance (Best Model)")
    fi_df = model_trainer.get_feature_importance_dataframe(feature_columns)
    if not fi_df.empty:
        top_fi = fi_df.head(20)
        fig = px.bar(
            top_fi, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig = _apply_plotly_style(fig, 500)
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance not available for the best model.")

    # Prediction vs Actual
    st.markdown("---")
    st.markdown("#### 📈 Prediction vs Actual (Test Set)")

    # Get test data dates
    test_ratio = 0.2
    split_idx = max(int(len(feature_df) * (1 - test_ratio)), len(feature_df) - max(7, int(len(feature_df) * 0.2)))
    test_dates = feature_df["date"].iloc[split_idx:].values
    y_actual = feature_df["total_revenue"].iloc[split_idx:].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_actual, mode="lines+markers",
        name="Actual", line=dict(color="white", width=2.5), marker=dict(size=5),
    ))

    for result in model_trainer.results[:3]:
        color = MODEL_COLORS.get(result.name, "#888")
        fig.add_trace(go.Scatter(
            x=test_dates, y=result.predictions, mode="lines",
            name=result.name, line=dict(color=color, width=1.5, dash="dash"),
        ))

    fig = _apply_plotly_style(fig, 400)
    fig.update_yaxes(title_text="Revenue (TND)")
    st.plotly_chart(fig, use_container_width=True)

    # Residual plots
    st.markdown("---")
    st.markdown("#### 📉 Residual Analysis")
    cols = st.columns(min(3, len(model_trainer.results[:3])))
    for idx, (result, col) in enumerate(zip(model_trainer.results[:3], cols)):
        with col:
            residuals = y_actual - result.predictions
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals, nbinsx=20,
                marker_color=MODEL_COLORS.get(result.name, "#888"),
                name=result.name,
            ))
            fig = _apply_plotly_style(fig, 300)
            fig.update_xaxes(title_text="Residual")
            fig.update_layout(title_text=result.name, title_font_size=13)
            st.plotly_chart(fig, use_container_width=True)


def render_revenue_forecast(forecaster) -> None:
    """Render the Revenue Forecast page."""
    st.markdown('<div class="section-header"><h2>💰 Revenue Forecast</h2></div>', unsafe_allow_html=True)

    if not forecaster.forecasts:
        st.info("Generate predictions to see the revenue forecast.")
        return

    # Summary table
    st.markdown("#### 📋 7-Day Forecast Summary")
    summary_df = forecaster.get_summary()
    st.dataframe(summary_df, use_container_width=True, height=200)

    st.markdown("---")

    # Individual model forecasts
    tabs = st.tabs(list(forecaster.forecasts.keys()) + (["Ensemble"] if forecaster.ensemble_forecast is not None else []))

    all_forecasts = list(forecaster.forecasts.items())
    if forecaster.ensemble_forecast is not None:
        all_forecasts.append(("Ensemble", forecaster.ensemble_forecast))

    for tab, (name, df) in zip(tabs, all_forecasts):
        with tab:
            # Main forecast chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["predicted_revenue"],
                mode="lines+markers", name="Predicted",
                line=dict(color=MODEL_COLORS.get(name, COLORS["primary"]), width=3),
                marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["upper_bound"],
                mode="lines", name="Upper Bound",
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["lower_bound"],
                mode="lines", name="Lower Bound",
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(108,99,255,0.15)",
                showlegend=True,
            ))
            fig = _apply_plotly_style(fig, 400)
            fig.update_yaxes(title_text="Revenue (TND)")
            st.plotly_chart(fig, use_container_width=True)

            # Daily breakdown
            st.markdown("#### Daily Breakdown")
            display_df = df[["date", "predicted_revenue", "lower_bound", "upper_bound", "trend_pct"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d (%A)")
            display_df.columns = ["Date", "Predicted (TND)", "Lower (TND)", "Upper (TND)", "Trend (%)"]
            st.dataframe(display_df, use_container_width=True, height=250)

            # KPIs
            total = df["predicted_revenue"].sum()
            trend = df["trend_pct"].iloc[-1] if len(df) > 0 else 0
            trend_color = COLORS["success"] if trend >= 0 else COLORS["danger"]
            trend_icon = "📈" if trend >= 0 else "📉"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div style="color:#8B949E;font-size:0.85rem">7-Day Total</div>'
                    f'<div style="color:#3FB950;font-size:1.5rem;font-weight:700">{format_number(total)}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div style="color:#8B949E;font-size:0.85rem">Trend</div>'
                    f'<div style="color:{trend_color};font-size:1.5rem;font-weight:700">'
                    f'{trend_icon} {trend:+.1f}%</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col3:
                conf = df["confidence_95"].iloc[0] if len(df) > 0 else "N/A"
                conf_color = COLORS["success"] if conf == "High" else COLORS["warning"] if conf == "Medium" else COLORS["danger"]
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div style="color:#8B949E;font-size:0.85rem">Confidence</div>'
                    f'<div style="color:{conf_color};font-size:1.5rem;font-weight:700">{conf}</div>'
                    f'</div>', unsafe_allow_html=True
                )


def render_product_forecast(product_forecaster) -> None:
    """Render the Product Forecast page."""
    st.markdown('<div class="section-header"><h2>📦 Product Forecast</h2></div>', unsafe_allow_html=True)

    if not product_forecaster.product_forecasts:
        st.info("Generate predictions to see product forecasts.")
        return

    # Model selector
    available_models = list(product_forecaster.product_forecasts.keys())
    selected_model = st.selectbox("Select Model", available_models, index=0)

    # Top products summary
    st.markdown(f"#### 🏆 Top 20 Predicted Products ({selected_model})")
    summary = product_forecaster.get_top_products_summary(selected_model, top_n=20)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, height=500)

        # Treemap
        st.markdown("---")
        st.markdown("#### 🗺️ Revenue Distribution Treemap")
        fig = px.treemap(
            summary, path=["category", "product"],
            values="total_predicted_revenue",
            color="total_predicted_revenue",
            color_continuous_scale="Viridis",
        )
        fig = _apply_plotly_style(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        st.markdown("---")
        st.markdown("#### 📊 Predicted Revenue by Product")
        fig = px.bar(
            summary.head(15), x="total_predicted_revenue", y="product",
            orientation="h", color="category",
        )
        fig = _apply_plotly_style(fig, 500)
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        # Daily forecast for top products
        st.markdown("---")
        st.markdown("#### 📅 Daily Forecast (Top 10 Products)")
        daily_df = product_forecaster.get_product_daily_forecast(selected_model, top_n=10)
        if not daily_df.empty:
            fig = px.bar(
                daily_df, x="date", y="predicted_revenue",
                color="product", barmode="stack",
            )
            fig = _apply_plotly_style(fig, 400)
            fig.update_yaxes(title_text="Predicted Revenue (TND)")
            fig.update_xaxes(title_text="")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No product forecast data available.")


def render_analytics(data_loader, weather_df: Optional[pd.DataFrame]) -> None:
    """Render the Analytics page with deep-dive visualizations."""
    st.markdown('<div class="section-header"><h2>📊 Analytics</h2></div>', unsafe_allow_html=True)

    if data_loader.df_commands is None:
        st.info("Upload data to see analytics.")
        return

    # Sales by month
    if data_loader.df_daily is not None:
        st.markdown("#### 📅 Sales by Month")
        df = data_loader.df_daily.copy()
        df["month_label"] = df["date"].dt.strftime("%Y-%m")
        monthly = df.groupby("month_label")["total_revenue"].sum().reset_index()

        fig = px.bar(
            monthly, x="month_label", y="total_revenue",
            color="total_revenue", color_continuous_scale="Viridis",
        )
        fig = _apply_plotly_style(fig, 350)
        fig.update_yaxes(title_text="Revenue (TND)")
        fig.update_xaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Sales by day of week
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📆 Sales by Day of Week")
        if data_loader.df_daily is not None:
            df = data_loader.df_daily.copy()
            df["dow"] = df["date"].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_avg = df.groupby("dow")["total_revenue"].mean().reindex(dow_order).reset_index()

            fig = px.bar(
                dow_avg, x="dow", y="total_revenue",
                color="total_revenue", color_continuous_scale="Blues",
            )
            fig = _apply_plotly_style(fig, 350)
            fig.update_yaxes(title_text="Avg Revenue (TND)")
            fig.update_xaxes(title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🕐 Sales by Hour")
        if data_loader.df_commands is not None:
            cmds = data_loader.df_commands.copy()
            cmds["hour"] = cmds[data_loader._col_map["date"]].dt.hour
            hourly = cmds.groupby("hour")[data_loader._col_map["amount"]].sum().reset_index()

            fig = px.bar(
                hourly, x="hour", y=data_loader._col_map["amount"],
                color=data_loader._col_map["amount"], color_continuous_scale="Reds",
            )
            fig = _apply_plotly_style(fig, 350)
            fig.update_yaxes(title_text="Revenue (TND)")
            fig.update_xaxes(title_text="Hour of Day")
            st.plotly_chart(fig, use_container_width=True)

    # Sales by store
    if data_loader.df_daily is not None and data_loader._col_map.get("store"):
        st.markdown("---")
        st.markdown("#### 🏪 Sales by Store")
        store_col = data_loader._col_map["store"]
        store_sales = data_loader.df_daily.groupby(store_col)["daily_revenue"].sum().reset_index()

        if data_loader.df_stores is not None:
            name_map = data_loader.df_stores.set_index(data_loader._col_map.get("store_id", "_id"))[
                data_loader._col_map.get("store_name", "_id")
            ].to_dict()
            store_sales["store_name"] = store_sales[store_col].map(name_map).fillna("Unknown")
        else:
            store_sales["store_name"] = store_sales[store_col].astype(str)

        fig = px.pie(
            store_sales, values="daily_revenue", names="store_name",
            hole=0.4, color_discrete_sequence=COLORS["gradient"],
        )
        fig = _apply_plotly_style(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    # Interactive map
    if data_loader.df_stores is not None:
        st.markdown("---")
        st.markdown("#### 🗺️ Store Locations Map")
        store_coords = data_loader.get_store_coordinates()
        if not store_coords.empty and store_coords["lat"].notna().any():
            fig = px.scatter_mapbox(
                store_coords, lat="lat", lon="lon",
                hover_name="store_name",
                zoom=10, height=400,
                color_discrete_sequence=[COLORS["primary"]],
                mapbox_style="open-street-map",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Weather vs Sales scatter
    if weather_df is not None and not weather_df.empty and data_loader.df_daily is not None:
        merged = data_loader.df_merged
        if merged is not None and "temp_avg" in merged.columns:
            st.markdown("---")
            st.markdown("#### 🌡️ Temperature vs Revenue")
            fig = px.scatter(
                merged, x="temp_avg", y="total_revenue",
                trendline="ols", color_discrete_sequence=[COLORS["primary"]],
            )
            fig = _apply_plotly_style(fig, 350)
            fig.update_xaxes(title_text="Temperature (°C)")
            fig.update_yaxes(title_text="Revenue (TND)")
            st.plotly_chart(fig, use_container_width=True)


def render_about() -> None:
    """Render the About page."""
    st.markdown('<div class="section-header"><h2>ℹ️ About</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### AI Sales Forecasting Platform

    **Projet de Fin d'Études (PFE)**

    This application leverages Machine Learning to predict daily revenue and product sales
    for point-of-sale systems.

    ---

    #### 🤖 Models Used
    - **XGBoost** – Gradient Boosted Trees
    - **LightGBM** – Light Gradient Boosting Machine
    - **CatBoost** – Categorical Boosting
    - **Random Forest** – Ensemble of Decision Trees
    - **LSTM** – Long Short-Term Memory Neural Network

    #### 🌤️ Weather Integration
    Historical weather data is automatically downloaded from **NASA POWER API**
    using store GPS coordinates, providing temperature, precipitation, wind speed,
    and humidity features.

    #### 📊 Features Engineered
    - Temporal features (day, week, month, quarter, day of week, weekend, holidays)
    - Lag features (1, 2, 3, 7, 14, 21, 28 days)
    - Rolling statistics (mean & std for 7 and 14 days)
    - Expanding mean
    - Sales trends and momentum
    - Moving averages (SMA, EMA, WMA)
    - Fourier seasonality components
    - Weather features and interactions

    #### 📁 Architecture
    ```
    app.py              → Main Streamlit entry point
    data_loader.py      → CSV loading & preprocessing
    weather.py          → NASA POWER API client
    feature_engineering.py → Feature generation
    models.py           → Model training & evaluation
    forecast.py         → Revenue forecasting
    product_forecast.py → Product-level forecasting
    dashboard.py        → Visualization pages
    utils.py            → Utilities & metrics
    ```

    ---

    *Built with Streamlit, Plotly, XGBoost, LightGBM, CatBoost, scikit-learn, and PyTorch.*
    """)


def render_exports(
    forecaster, product_forecaster, model_trainer
) -> None:
    """Render export/download section."""
    st.markdown("#### 📥 Export Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if forecaster.ensemble_forecast is not None:
            csv = forecaster.ensemble_forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Revenue Forecast", csv, "revenue_forecast.csv", "text/csv",
                use_container_width=True,
            )

    with col2:
        if model_trainer.results:
            metrics_df = model_trainer.get_metrics_dataframe()
            csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Model Metrics", csv, "model_metrics.csv", "text/csv",
                use_container_width=True,
            )

    with col3:
        if forecaster.forecasts:
            all_fc = pd.concat(forecaster.forecasts.values(), ignore_index=True)
            csv = all_fc.to_csv(index=False).encode("utf-8")
            st.download_button(
                "All Forecasts", csv, "all_forecasts.csv", "text/csv",
                use_container_width=True,
            )

    with col4:
        if product_forecaster.product_forecasts:
            best_name = list(product_forecaster.product_forecasts.keys())[0]
            prod_df = product_forecaster.product_forecasts[best_name]
            csv = prod_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Product Forecast", csv, "product_forecast.csv", "text/csv",
                use_container_width=True,
            )