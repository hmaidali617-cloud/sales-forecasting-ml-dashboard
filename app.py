"""
AI Sales Forecasting - Main Application Entry Point
Streamlit dashboard for predicting daily revenue and product sales
using XGBoost, LightGBM, CatBoost, Random Forest, and LSTM.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import logging
import sys
import os

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import logger, get_dark_theme_css
from data_loader import DataLoader
from weather import WeatherClient
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from forecast import RevenueForecaster
from product_forecast import ProductForecaster
import dashboard as dash

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sales Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Apply Dark Theme CSS ────────────────────────────────────────────
st.markdown(get_dark_theme_css(), unsafe_allow_html=True)

# ─── Initialize Session State ────────────────────────────────────────
if "data_loader" not in st.session_state:
    st.session_state.data_loader = DataLoader()
if "weather_client" not in st.session_state:
    st.session_state.weather_client = WeatherClient()
if "feature_engineer" not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if "model_trainer" not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if "forecaster" not in st.session_state:
    st.session_state.forecaster = RevenueForecaster()
if "product_forecaster" not in st.session_state:
    st.session_state.product_forecaster = ProductForecaster()
if "feature_df" not in st.session_state:
    st.session_state.feature_df = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = []
if "weather_df" not in st.session_state:
    st.session_state.weather_df = None
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "page" not in st.session_state:
    st.session_state.page = "Home"


def reset_state() -> None:
    """Reset all session state except data loader."""
    st.session_state.feature_engineer = FeatureEngineer()
    st.session_state.model_trainer = ModelTrainer()
    st.session_state.forecaster = RevenueForecaster()
    st.session_state.product_forecaster = ProductForecaster()
    st.session_state.feature_df = None
    st.session_state.feature_columns = []
    st.session_state.weather_df = None
    st.session_state.models_trained = False
    st.session_state.predictions_made = False


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0">
            <div style="font-size:2rem">📊</div>
            <div style="font-size:1.3rem;font-weight:700;color:#E6EDF3">
                AI Sales Forecast
            </div>
            <div style="font-size:0.75rem;color:#8B949E;margin-top:4px">
                PFE – Machine Learning Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── File Upload Section ──
    st.markdown("### 📁 Upload Data")

    cmd_file = st.file_uploader(
        "Command file (commande.csv)",
        type=["csv"],
        key="cmd_upload",
        help="Upload the command/order CSV file",
    )

    store_file = st.file_uploader(
        "Store file (pointvente.csv)",
        type=["csv"],
        key="store_upload",
        help="Upload the point of sale CSV file",
    )

    prod_file = st.file_uploader(
        "Product file (produit.csv or produitpointvente.csv)",
        type=["csv"],
        key="prod_upload",
        help="Upload the product or product-point-of-sale CSV file",
    )

    # Load data button
    if st.button("📂 Load Data", use_container_width=True):
        if cmd_file is None:
            st.error("Please upload at least the command file.")
        else:
            with st.spinner("Loading and processing data..."):
                try:
                    dl = st.session_state.data_loader
                    dl.load_commands(cmd_file)

                    if store_file is not None:
                        dl.load_stores(store_file)

                    if prod_file is not None:
                        dl.load_products(prod_file)

                    dl.aggregate_daily_sales()
                    dl.estimate_product_sales()
                    dl.merge_datasets()

                    reset_state()
                    st.session_state.data_loader = dl
                    st.success(
                        f"✅ Data loaded successfully!\n"
                        f"• {len(dl.df_commands)} commands\n"
                        f"• {len(dl.df_stores)} stores\n"
                        f"• {len(dl.df_products)} product records\n"
                        f"• {len(dl.df_daily)} daily records"
                    )
                    logger.info("Data loading complete via UI")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    logger.exception("Data loading failed")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Action Buttons ──
    st.markdown("### ⚡ Actions")

    train_clicked = st.button(
        "🤖 Train Models",
        use_container_width=True,
        disabled=st.session_state.data_loader.df_commands is None,
        help="Train all 5 ML models (requires loaded data)",
    )

    predict_clicked = st.button(
        "🔮 Generate Predictions",
        use_container_width=True,
        disabled=not st.session_state.models_trained,
        help="Predict next 7 days using top 3 models",
    )

    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown("### 🧭 Navigation")

    pages = [
        "Home",
        "Data Explorer",
        "Weather",
        "Model Training",
        "Model Comparison",
        "Revenue Forecast",
        "Product Forecast",
        "Analytics",
        "About",
    ]

    for page in pages:
        icon = {
            "Home": "🏠",
            "Data Explorer": "🔍",
            "Weather": "🌤️",
            "Model Training": "🤖",
            "Model Comparison": "⚖️",
            "Revenue Forecast": "💰",
            "Product Forecast": "📦",
            "Analytics": "📊",
            "About": "ℹ️",
        }.get(page, "📄")

        is_active = st.session_state.page == page
        btn_style = (
            f"background:linear-gradient(135deg,#6C63FF,#8B85FF);color:white;"
            if is_active
            else "background:#1C2333;color:#8B949E;border:1px solid #30363D;"
        )
        btn_hover = (
            "background:linear-gradient(135deg,#8B85FF,#A5A0FF);color:white;"
            if is_active
            else "background:#262D3A;color:#E6EDF3;border:1px solid #6C63FF;"
        )

        st.markdown(
            f"""
            <style>
            div[data-testid="stSidebar"] button[title="{page}"] {{
                {btn_style}
                border-radius:8px;padding:0.5rem 1rem;margin:2px 0;
                font-weight:500;text-align:left;width:100%;
                transition:all 0.2s ease;cursor:pointer;
                display:flex;align-items:center;gap:0.5rem;
            }}
            div[data-testid="stSidebar"] button[title="{page}"]:hover {{
                {btn_hover}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()

    # ── Status Indicators ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📌 Status")

    data_loaded = st.session_state.data_loader.df_commands is not None
    models_ready = st.session_state.models_trained
    preds_ready = st.session_state.predictions_made

    st.markdown(
        f"""
        <div style="padding:0.3rem 0">
            <span style="color:{"#3FB950" if data_loaded else "#F85149"}">●</span>
            <span style="color:#8B949E;font-size:0.85rem"> Data Loaded</span>
        </div>
        <div style="padding:0.3rem 0">
            <span style="color:{"#3FB950" if models_ready else "#F85149"}">●</span>
            <span style="color:#8B949E;font-size:0.85rem"> Models Trained</span>
        </div>
        <div style="padding:0.3rem 0">
            <span style="color:{"#3FB950" if preds_ready else "#F85149"}">●</span>
            <span style="color:#8B949E;font-size:0.85rem"> Predictions Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Background Processing: Train Models ─────────────────────────────
if train_clicked:
    dl = st.session_state.data_loader
    if dl.df_daily is None or dl.df_daily.empty:
        st.error("No daily data available. Load data first.")
    else:
        with st.spinner("🔄 Downloading weather data & engineering features..."):
            try:
                wc = st.session_state.weather_client
                engineer = st.session_state.feature_engineer
                weather_df = None

                # Download weather if we have store coordinates
                if dl.df_stores is not None and dl._col_map.get("coord_x") and dl._col_map.get("coord_y"):
                    date_range = dl.get_date_range()
                    if date_range[0] and date_range[1]:
                        weather_df = wc.fetch_for_stores(
                            dl.df_stores,
                            dl._col_map["coord_x"],
                            dl._col_map["coord_y"],
                            dl._col_map.get("store_id", "_id"),
                            date_range[0],
                            date_range[1],
                        )
                        st.session_state.weather_df = weather_df

                # Build features
                feature_df = dl.df_merged.copy() if dl.df_merged is not None else dl.df_daily.copy()

                if weather_df is not None and not weather_df.empty:
                    feature_df = wc.merge_weather_with_sales(
                        feature_df, weather_df,
                        dl._col_map.get("coord_x", "coord_x"),
                        dl._col_map.get("coord_y", "coord_y"),
                    )

                feature_df = engineer.build_all_features(
                    feature_df,
                    target_col="total_revenue",
                    include_weather=True,
                )

                st.session_state.feature_df = feature_df
                st.session_state.feature_columns = engineer.feature_columns

            except Exception as e:
                st.error(f"Feature engineering failed: {e}")
                logger.exception("Feature engineering failed")
                st.stop()

        with st.spinner("🤖 Training 5 models (XGBoost, LightGBM, CatBoost, RF, LSTM)..."):
            try:
                trainer = st.session_state.model_trainer
                trainer.train_all_models(
                    st.session_state.feature_df,
                    st.session_state.feature_columns,
                    target_col="total_revenue",
                    test_ratio=0.2,
                )
                st.session_state.model_trainer = trainer
                st.session_state.models_trained = True

                best = trainer.results[0]
                st.success(
                    f"✅ All 5 models trained!\n\n"
                    f"🥇 **{best.name}** – RMSE: {best.metrics['RMSE']:.2f}, "
                    f"R²: {best.metrics['R² Score']:.4f}\n\n"
                    f"Top 3 selected for prediction:\n"
                    f"  1. {trainer.results[0].name}\n"
                    f"  2. {trainer.results[1].name}\n"
                    f"  3. {trainer.results[2].name}"
                )
                st.session_state.page = "Model Training"
                st.rerun()

            except Exception as e:
                st.error(f"Model training failed: {e}")
                logger.exception("Model training failed")


# ─── Background Processing: Generate Predictions ─────────────────────
if predict_clicked:
    with st.spinner("🔮 Generating 7-day revenue & product forecasts..."):
        try:
            trainer = st.session_state.model_trainer
            forecaster = st.session_state.forecaster
            prod_forecaster = st.session_state.product_forecaster

            # Revenue forecast
            forecasts = forecaster.forecast_next_7_days(
                trainer,
                st.session_state.feature_df,
                st.session_state.feature_columns,
                historical_weather=st.session_state.weather_df,
                target_col="total_revenue",
            )
            st.session_state.forecaster = forecaster

            # Product forecast
            dl = st.session_state.data_loader
            if dl.df_product_daily is not None and not dl.df_product_daily.empty:
                prod_forecasts = prod_forecaster.forecast_products(
                    dl.df_product_daily,
                    forecasts,
                    top_n=20,
                    days=7,
                )
                st.session_state.product_forecaster = prod_forecaster

            st.session_state.predictions_made = True

            ensemble = forecaster.ensemble_forecast
            total_rev = (
                ensemble["predicted_revenue"].sum()
                if ensemble is not None
                else 0
            )
            st.success(
                f"✅ Predictions generated!\n\n"
                f"💰 Predicted 7-day revenue: **{total_rev:,.2f} TND**\n"
                f"📊 {len(forecasts)} model forecasts + ensemble\n"
                f"📦 Product-level forecasts ready"
            )
            st.session_state.page = "Revenue Forecast"
            st.rerun()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.exception("Prediction failed")


# ─── Main Content Area ───────────────────────────────────────────────
page = st.session_state.page

try:
    if page == "Home":
        dash.render_home(st.session_state.data_loader)

    elif page == "Data Explorer":
        dash.render_data_explorer(st.session_state.data_loader)

    elif page == "Weather":
        dash.render_weather(
            st.session_state.data_loader,
            st.session_state.weather_df,
        )

    elif page == "Model Training":
        dash.render_model_training(
            st.session_state.model_trainer,
            st.session_state.feature_columns,
        )

    elif page == "Model Comparison":
        dash.render_model_comparison(
            st.session_state.model_trainer,
            st.session_state.feature_df,
            st.session_state.feature_columns,
        )

    elif page == "Revenue Forecast":
        dash.render_revenue_forecast(st.session_state.forecaster)
        if st.session_state.predictions_made:
            st.markdown("---")
            dash.render_exports(
                st.session_state.forecaster,
                st.session_state.product_forecaster,
                st.session_state.model_trainer,
            )

    elif page == "Product Forecast":
        dash.render_product_forecast(st.session_state.product_forecaster)
        if st.session_state.predictions_made:
            st.markdown("---")
            dash.render_exports(
                st.session_state.forecaster,
                st.session_state.product_forecaster,
                st.session_state.model_trainer,
            )

    elif page == "Analytics":
        dash.render_analytics(
            st.session_state.data_loader,
            st.session_state.weather_df,
        )

    elif page == "About":
        dash.render_about()

except Exception as e:
    logger.exception(f"Error rendering page '{page}'")
    st.error(f"An error occurred while rendering this page: {e}")
    st.info("Try reloading the data or navigating to a different page.")