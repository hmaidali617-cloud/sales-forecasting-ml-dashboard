"""
app.py
======
Streamlit entry point - AI Sales Forecasting Dashboard.
Run: streamlit run app.py

FULLY LOCAL - NO API KEYS REQUIRED FOR AI ASSISTANT
Uses Ollama, HuggingFace, or LM Studio for local LLM inference.

UPDATED: Now supports BPrice JSON format directly!
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    get_logger, rank_models, to_csv_bytes, format_currency, EXPORTS_DIR,
)
from data_loader import (
    load_csv_from_upload, load_json_from_upload, load_orders_from_upload,
    clean_orders, clean_stores, clean_products,
    aggregate_daily_revenue, build_global_daily_series, merge_with_stores,
)
from weather import fetch_weather_for_stores
from feature_engineering import build_features, feature_columns
from models import train_all_models, get_feature_importance
from forecast import predict_next_days
from product_forecast import aggregate_product_sales, forecast_products
import dashboard as dash
from downloads import show_chart, show_table, fig_to_png_bytes, df_to_png_bytes
from chat_agent import (
    get_bi_agent_response,
    LLMConfig,
    check_local_llm_status,
    get_recommended_models,
    reset_llm_cache,
)

logger = get_logger("app")

# ---------------------------------------------------------------- #
# Page config                                                      #
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0e1117; color: white; }
    .css-1d391kg, .css-1lcbmhc { background-color: #0e1117; }
    .st-bb { background-color: #1a1f29; }
    .st-b8 { color: white; }
    .metric-card {
        background-color: #1a1f29; border-radius: 12px; padding: 18px;
        border: 1px solid #2a2f3a; text-align: center;
    }
    .metric-card .label { color: #9ca3af; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #00CC96; font-size: 1.6rem; font-weight: 700; }
    .section-header {
        background: linear-gradient(90deg, #636EFA 0%, #00CC96 100%);
        padding: 12px 20px; border-radius: 8px; margin: 12px 0 18px 0;
        color: white; font-size: 1.2rem; font-weight: 700;
    }
    .stButton>button { background-color: #636EFA; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #4c5fd5; color: white; }
    .sidebar .sidebar-content { background-color: #0a0d13; }
    .stDataFrame { width: 100%; }
    .stDataFrame table { width: 100%; }
    .llm-status-connected { color: #00CC96; font-weight: bold; }
    .llm-status-disconnected { color: #FF6B6B; font-weight: bold; }
    .model-badge {
        display: inline-block; background-color: #2a2f3a; padding: 4px 10px;
        border-radius: 6px; margin: 2px; font-size: 0.85rem;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    sub_html = f"<div style='color:#9ca3af;font-size:0.8rem;'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Session state                                                    #
# ---------------------------------------------------------------- #
def init_state():
    defaults = {
        "orders": None, "stores": None, "products": None,
        "daily_store": None, "daily_global": None,
        "weather": None,
        "featured": None, "feature_cols": None,
        "metrics": None, "fitted": None, "preds": None,
        "train_df": None, "test_df": None, "histories": None,
        "top_models": None,
        "forecast_df": None,
        "product_daily": None, "product_forecasts": None,
        "trained": False, "forecasted": False,
        "orders_raw": None,
        "last_history_months": None,
        "bi_chat_messages": [],
        "llm_backend": "ollama",
        "llm_model": "llama3.2:3b",
        "llm_api_base": "http://localhost:11434",
        "llm_temperature": 0.3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------- #
# Sidebar                                                          #
# ---------------------------------------------------------------- #
st.sidebar.markdown("## 📊 AI Sales Forecasting")
st.sidebar.markdown("Upload your data files to begin.")

up_orders = st.sidebar.file_uploader("1️⃣ Commande (Orders CSV/JSON)", type=["csv", "json"], key="up_orders")
up_stores = st.sidebar.file_uploader("2️⃣ Point de Vente (Stores)", type=["csv"], key="up_stores")
up_products = st.sidebar.file_uploader("3️⃣ Produit / ProduitPointVente (Optional)", type=["csv"], key="up_products")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Training Window")
history_months = st.sidebar.slider(
    "Train on last X months:", 
    min_value=3, 
    max_value=48, 
    value=12, 
    step=3,
    help="Use 6-12 months for current revenue levels."
)

st.sidebar.markdown("---")
train_btn = st.sidebar.button("🚀 Train Models", use_container_width=True)
forecast_btn = st.sidebar.button("🔮 Predict Next 7 Days", use_container_width=True, disabled=not st.session_state["trained"])

# Status indicators
if st.session_state["trained"]:
    st.sidebar.success("✅ Models Trained", icon="✅")
else:
    st.sidebar.warning("⏳ Not trained yet", icon="⏳")

if not st.session_state["trained"]:
    st.sidebar.caption("⏳ *Train models first to unlock Predict button*")

st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
pages = [
    "🏠 Home", "📂 Data Explorer", "🌤️ Weather",
    "🧠 Model Training", "🏆 Model Comparison",
    "💰 Revenue Forecast", "📦 Product Forecast",
    "📈 Analytics", "🤖 BI Assistant", "ℹ️ About",
]
page = st.sidebar.radio("Go to", pages, index=0, label_visibility="collapsed")

# ---------------------------------------------------------------- #
# Local LLM Settings in Sidebar                                    #
# ---------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Local LLM Settings")
st.sidebar.caption("100% local AI - No API keys needed!")

# Check LLM status
@st.cache_data(ttl=30)
def get_llm_status():
    return check_local_llm_status()

llm_status = get_llm_status()

# Backend selection
backend_options = ["ollama", "openai_compatible"]
backend_labels = {
    "ollama": "🦙 Ollama (Recommended)",
    "openai_compatible": "🔌 LM Studio / LocalAI",
}
selected_backend = st.sidebar.selectbox(
    "Backend",
    backend_options,
    index=backend_options.index(st.session_state.get("llm_backend", "ollama")),
    format_func=lambda x: backend_labels.get(x, x),
    key="llm_backend_select",
)

# Update session state if changed
if selected_backend != st.session_state["llm_backend"]:
    st.session_state["llm_backend"] = selected_backend
    reset_llm_cache()

# Show status and model selection based on backend
if selected_backend == "ollama":
    if llm_status["ollama"]["available"]:
        st.markdown('<span class="llm-status-connected">● Connected</span>', unsafe_allow_html=True)
        available_models = llm_status["ollama"]["models"]
        if not available_models:
            available_models = ["llama3.2:3b"]
    else:
        st.markdown('<span class="llm-status-disconnected">● Not Running</span>', unsafe_allow_html=True)
        available_models = ["llama3.2:3b", "phi3:mini", "gemma2:2b", "llama3.2:1b", "mistral:7b"]
    
    selected_model = st.sidebar.selectbox(
        "Model",
        available_models,
        index=0 if st.session_state.get("llm_model") not in available_models else available_models.index(st.session_state["llm_model"]),
        key="llm_model_select_ollama",
    )
    api_base = "http://localhost:11434"
    
    # Show setup help if not connected
    if not llm_status["ollama"]["available"]:
        with st.sidebar.expander("💡 Quick Setup"):
            st.markdown("""
            **Install Ollama:**
            ```bash
            # macOS
            brew install ollama
            
            # Linux
            curl -fsSL https://ollama.ai/install.sh | sh
            
            # Windows
            # Download from https://ollama.ai
            ```
            
            **Pull a model:**
            ```bash
            ollama pull llama3.2:3b
            ```
            
            **Start Ollama:**
            ```bash
            ollama serve
            ```
            """)

else:  # openai_compatible
    if llm_status.get("openai_compatible", {}).get("available"):
        st.markdown('<span class="llm-status-connected">● Connected</span>', unsafe_allow_html=True)
        available_models = llm_status["openai_compatible"].get("models", ["local-model"])
        api_base = llm_status["openai_compatible"].get("url", "http://localhost:1234")
    else:
        st.markdown('<span class="llm-status-disconnected">● Not Running</span>', unsafe_allow_html=True)
        available_models = ["local-model"]
        api_base = "http://localhost:1234"
    
    selected_model = st.sidebar.text_input("Model Name", value=st.session_state.get("llm_model", "local-model"), key="llm_model_text")
    api_base = st.sidebar.text_input("API Base URL", value=api_base, key="llm_api_base_text")
    
    # Show setup help if not connected
    if not llm_status.get("openai_compatible", {}).get("available"):
        with st.sidebar.expander("💡 Quick Setup"):
            st.markdown("""
            **LM Studio (Easiest GUI):**
            1. Download from https://lmstudio.ai
            2. Search for "Llama 3.2 3B"
            3. Download and load the model
            4. Click "Start Server" (localhost:1234)
            
            **LocalAI:**
            ```bash
            docker run -p 8080:8080 localai/localai
            ```
            """)

# Update session state
st.session_state["llm_model"] = selected_model
st.session_state["llm_api_base"] = api_base

# Temperature slider
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.get("llm_temperature", 0.3), 
    step=0.1,
    help="Lower = more precise, Higher = more creative"
)
st.session_state["llm_temperature"] = temperature

# Recommended models expander
with st.sidebar.expander("💡 Recommended Models"):
    for rec in get_recommended_models():
        if rec["backend"] == selected_backend:
            st.markdown(f"**{rec['name']}** ({rec['size']}, {rec['ram']} RAM)")
            st.caption(rec['description'])
            if selected_backend == "ollama":
                st.code(rec['pull_cmd'], language="bash")


# ---------------------------------------------------------------- #
# Data loading                                                     #
# ---------------------------------------------------------------- #
def apply_time_filter_and_aggregate():
    """Apply time filter and aggregate daily data."""
    if st.session_state.get("orders_raw") is None:
        return
        
    try:
        orders_df = st.session_state["orders_raw"].copy()
        latest_date = orders_df["date"].max()
        earliest_date = orders_df["date"].min()
        
        # Calculate cutoff
        cutoff_date = latest_date - pd.DateOffset(months=history_months)
        
        # Don't filter if cutoff is before data start
        if cutoff_date <= earliest_date:
            filtered_orders = orders_df
            st.info(f"📅 Using all available data ({len(orders_df)} orders) - your file covers {(latest_date - earliest_date).days} days")
        else:
            original_count = len(orders_df)
            filtered_orders = orders_df[orders_df["date"] >= cutoff_date].copy()
            dropped = original_count - len(filtered_orders)
            
            if dropped > 0:
                st.info(f"📅 **Filtered Data**: Kept last {history_months} months. Dropped {dropped:,} old orders.")
                st.caption(f"📈 *Training Range: {cutoff_date.date()} → {latest_date.date()}*")
            else:
                st.info(f"📅 Using all {original_count:,} orders within {history_months}-month window.")
        
        # Check if filtered data is empty
        if filtered_orders.empty:
            st.error(f"❌ No data left after filtering! Your file only has {(latest_date - earliest_date).days} days of data, but slider is set to {history_months} months.")
            st.session_state["orders"] = None
            st.session_state["daily_store"] = None
            st.session_state["daily_global"] = None
            return
        
        st.session_state["orders"] = filtered_orders
        
        # Aggregate
        st.session_state["daily_store"] = aggregate_daily_revenue(st.session_state["orders"])
        
        if st.session_state["stores"] is not None:
            st.session_state["daily_store"] = merge_with_stores(st.session_state["daily_store"], st.session_state["stores"])
            
        st.session_state["daily_global"] = build_global_daily_series(st.session_state["daily_store"])
        
        if not st.session_state["daily_global"].empty:
            st.success("✅ Data loaded & aggregated successfully")
            
    except Exception as e:
        st.error(f"❌ Data processing error: {e}")
        logger.exception("Data processing failed")


def load_all_data():
    """Load and clean all uploaded files. Supports JSON for BPrice orders!"""
    try:
        # Track if orders came from JSON (for auto product extraction)
        orders_from_json = False
        
        if up_orders is not None:
            filename = up_orders.name.lower()
            
            if filename.endswith('.json'):
                # ============================================ #
                # JSON UPLOAD (BPrice CommandeCommandeDetails)  #
                # ============================================ #
                orders_from_json = True
                orders_df, products_from_json = load_orders_from_upload(up_orders)
                st.session_state["orders_raw"] = clean_orders(orders_df, from_json=True)
                
                # Auto-populate products from JSON's commandeDetails
                # ONLY if user didn't upload a separate products file
                if products_from_json is not None and not products_from_json.empty:
                    if up_products is None:
                        st.session_state["products"] = clean_products(products_from_json, from_json=True)
                        st.sidebar.success("📦 Products auto-extracted from JSON!")
            else:
                # ============================================ #
                # CSV UPLOAD (Standard format)               #
                # ============================================ #
                raw = load_csv_from_upload(up_orders)
                st.session_state["orders_raw"] = clean_orders(raw, from_json=False)
        
        # Stores CSV (always CSV - bprice_prod.pointVente.csv)
        if up_stores is not None:
            raw = load_csv_from_upload(up_stores)
            st.session_state["stores"] = clean_stores(raw)
            
        # Products CSV (optional - only if user uploaded it separately)
        # If JSON was used and auto-extracted products, this is skipped
        if up_products is not None:
            raw = load_csv_from_upload(up_products)
            st.session_state["products"] = clean_products(raw, from_json=False)
            
        # Apply filter and aggregate
        apply_time_filter_and_aggregate()
        
        # Remember the slider value so we know when it actually changes
        st.session_state["last_history_months"] = history_months
            
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        logger.exception("Data loading failed")


# ---------------------------------------------------------------- #
# Data loading trigger                                             #
# ---------------------------------------------------------------- #
# Case 1: First upload — orders_raw not set yet
if up_orders is not None and st.session_state["orders_raw"] is None:
    load_all_data()

# Case 2: Slider changed — re-filter but DON'T reset training state
elif (up_orders is not None 
      and st.session_state["orders_raw"] is not None 
      and st.session_state["last_history_months"] != history_months):
    apply_time_filter_and_aggregate()
    st.session_state["last_history_months"] = history_months

# Case 3: Nothing changed — do nothing, keep all session state intact


# ---------------------------------------------------------------- #
# Train models                                                     #
# ---------------------------------------------------------------- #
def do_train():
    if st.session_state["daily_global"] is None or st.session_state["daily_global"].empty:
        st.error("No aggregated daily data. Please upload files first.")
        return
    
    # ═══════════════════════════════════════════════════════════════════
    # FIX: Check if stores file is uploaded FIRST - show clear message
    # ═══════════════════════════════════════════════════════════════════
    if st.session_state["stores"] is None:
        st.warning("⚠️ No stores file uploaded. Weather data requires stores with coordinates (coord_x, coord_y).")
        st.session_state["weather"] = pd.DataFrame()
    else:
        # ═══════════════════════════════════════════════════════════════════
        # FIX: Check if weather is EMPTY (not just None) to allow retry
        # ═══════════════════════════════════════════════════════════════════
        weather_needs_fetch = (
            st.session_state["weather"] is None or 
            st.session_state["weather"].empty
        )
        
        if weather_needs_fetch:
            with st.spinner("Fetching weather from NASA POWER..."):
                try:
                    start_d = st.session_state["daily_global"]["date"].min()
                    end_d = st.session_state["daily_global"]["date"].max()
                    
                    # Debug: Show what we're fetching
                    logger.info(f"Fetching weather for {len(st.session_state['stores'])} stores")
                    logger.info(f"Date range: {start_d} to {end_d}")
                    logger.info(f"Stores columns: {st.session_state['stores'].columns.tolist()}")
                    
                    # Check if stores have valid coordinates
                    valid_coords = st.session_state["stores"].dropna(subset=["coord_x", "coord_y"])
                    if valid_coords.empty:
                        st.warning("⚠️ Stores file has no valid coordinates (coord_x, coord_y). Weather data cannot be fetched.")
                        logger.warning("No valid coordinates in stores file")
                    else:
                        logger.info(f"Found {len(valid_coords)} stores with valid coordinates")
                        st.session_state["weather"] = fetch_weather_for_stores(
                            st.session_state["stores"], start_d, end_d
                        )
                        
                        if st.session_state["weather"].empty:
                            st.warning("⚠️ NASA POWER returned no weather data. Check your date range (must be in the past).")
                        else:
                            st.success(f"✅ Fetched {len(st.session_state['weather'])} weather records")
                            
                except Exception as e:
                    st.warning(f"Weather fetch failed: {e}")
                    logger.exception("Weather fetch failed")
                    st.session_state["weather"] = pd.DataFrame()

     

    with st.spinner("Building features..."):
        try:
            daily = st.session_state["daily_global"].copy()
            daily["date"] = pd.to_datetime(daily["date"])
            weather = st.session_state["weather"]
            if weather is not None and not weather.empty:
                w_global = weather.groupby("date", as_index=False).agg(
                    temp_avg=("temp_avg", "mean"),
                    temp_max=("temp_max", "mean"),
                    temp_min=("temp_min", "mean"),
                    rain=("rain", "sum"),
                    wind_speed=("wind_speed", "mean"),
                    humidity=("humidity", "mean"),
                )
                w_global["date"] = pd.to_datetime(w_global["date"])
                # ================================================================
# FINAL DATE NORMALIZATION BEFORE WEATHER MERGE
# ================================================================

                daily["date"] = pd.to_datetime(
                    daily["date"],
                    errors="coerce",
                    utc=True
                ).dt.tz_localize(None).dt.normalize()

                w_global["date"] = pd.to_datetime(
                    w_global["date"],
                    errors="coerce",
                    utc=True
                ).dt.tz_localize(None).dt.normalize()

                # Debug
                print("DEBUG BEFORE WEATHER MERGE")
                print("daily date:", daily["date"].dtype)
                print("w_global date:", w_global["date"].dtype)

                # Remove invalid dates
                daily = daily.dropna(subset=["date"])
                w_global = w_global.dropna(subset=["date"])

                # Merge
                daily = daily.merge(
                    w_global,
                    on="date",
                    how="left"
                )
                daily = daily.merge(w_global, on="date", how="left")
            featured = build_features(daily, weather=None, target="revenue", group_col=None)
            st.session_state["featured"] = featured
            st.session_state["feature_cols"] = feature_columns(featured, "revenue")
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            logger.exception("Feature engineering failed")
            return

    with st.spinner("Training 5 models..."):
        progress = st.progress(0, text="Initializing training...")
        try:
            metrics, fitted, preds, train_df, test_df = train_all_models(
                st.session_state["featured"],
                st.session_state["feature_cols"],
                target="revenue",
                test_ratio=0.2,
            )
            st.session_state["metrics"] = metrics
            st.session_state["fitted"] = fitted
            st.session_state["preds"] = preds
            st.session_state["train_df"] = train_df
            st.session_state["test_df"] = test_df
            st.session_state["histories"] = {n: {} for n in fitted}
            ranked = rank_models(metrics, "rmse", "r2")
            st.session_state["top_models"] = [m.model_name for m in ranked[:3] if m.model_name in fitted]
            st.session_state["trained"] = True
            progress.progress(100, text="Done!")
            st.success(f"✅ Trained {len(fitted)} models. Top 3: {', '.join(st.session_state['top_models'])}")
        except Exception as e:
            st.error(f"Training failed: {e}")
            logger.exception("Training failed")
            return

    try:
        if st.session_state["orders"] is not None:
            st.session_state["product_daily"] = aggregate_product_sales(st.session_state["orders"], st.session_state["products"])
    except Exception as e:
        logger.warning(f"Product aggregation failed: {e}")


if train_btn:
    do_train()


# ---------------------------------------------------------------- #
# Forecast                                                         #
# ---------------------------------------------------------------- #
def do_forecast():
    if not st.session_state["trained"]:
        st.error("Please train models first.")
        return
    with st.spinner("Generating 7-day forecast..."):
        try:
            fc = predict_next_days(
                st.session_state["featured"],
                st.session_state["fitted"],
                st.session_state["feature_cols"],
                target="revenue",
                horizon=7,
                top_model_names=st.session_state["top_models"],
            )
            st.session_state["forecast_df"] = fc
            if st.session_state["product_daily"] is not None:
                st.session_state["product_forecasts"] = forecast_products(
                    st.session_state["product_daily"],
                    st.session_state["top_models"],
                    horizon=7,
                    top_n=20,
                )
            st.session_state["forecasted"] = True
            st.success("✅ Forecast generated!")
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            logger.exception("Forecast failed")


if forecast_btn:
    do_forecast()


# ---------------------------------------------------------------- #
# Pages                                                            #
# ---------------------------------------------------------------- #
def page_home():
    st.markdown("<div class='section-header'>🏠 Home</div>", unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 AI-Powered Sales Forecasting Platform

    Welcome! This dashboard forecasts **daily revenue** and **top-selling products** 
    for the next 7 days using 5 ML models (XGBoost, LightGBM, CatBoost, RandomForest, LSTM)
    enriched with **NASA POWER** historical weather data.

    #### 📋 How to use
    1. Upload your **JSON** orders file (`bprice_prod.CommandeCommandeDetails.json`) or CSV
    2. Upload your **CSV** stores file (`bprice_prod.pointVente.csv`) for coordinates
    3. Products are **auto-extracted** from JSON (no need to upload separately!)
    4. Click **🚀 Train Models**
    5. Click **🔮 Predict Next 7 Days**
    6. Navigate through the pages to explore insights
    7. Use the **🤖 BI Assistant** to ask questions about your data (100% local AI!)
    """)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        orders_n = len(st.session_state["orders"]) if st.session_state["orders"] is not None else 0
        metric_card("Orders", f"{orders_n:,}", "Total records")
    with c2:
        stores_n = st.session_state["stores"]["store_id"].nunique() if st.session_state["stores"] is not None else 0
        metric_card("Stores", f"{stores_n:,}", "Point de Vente")
    with c3:
        prods_n = st.session_state["products"]["product_id"].nunique() if st.session_state["products"] is not None else 0
        metric_card("Products", f"{prods_n:,}", "Catalog items")
    with c4:
        dg = st.session_state["daily_global"]
        if dg is not None and not dg.empty:
            d_range = f"{dg['date'].min().date()} → {dg['date'].max().date()}"
        else:
            d_range = "—"
        metric_card("Date Range", d_range, "Historical period")

    st.markdown("---")
    if st.session_state["trained"] and st.session_state["metrics"]:
        st.markdown("### 🏆 Best Model")
        ranked = rank_models(st.session_state["metrics"])
        bm = ranked[0]
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            metric_card("Best Model", bm.model_name, "Lowest RMSE")
        with bc2:
            metric_card("RMSE", f"{bm.rmse:,.2f}", "Lower is better")
        with bc3:
            metric_card("R²", f"{bm.r2:.4f}", "Higher is better")
        with bc4:
            metric_card("MAPE", f"{bm.mape:.2f}%", "Error percentage")

    if st.session_state["forecasted"] and st.session_state["forecast_df"] is not None:
        st.markdown("---")
        st.markdown("### 💰 7-Day Revenue Forecast")
        fc = st.session_state["forecast_df"]
        total = fc["predicted_revenue"].sum()
        avg = fc["predicted_revenue"].mean()
        show_chart(
            dash.plot_train_test_forecast(st.session_state["train_df"], st.session_state["test_df"], fc),
            filename="home_forecast.png",
            key="home_forecast"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Forecast", format_currency(total), "Next 7 days")
        with c2:
            metric_card("Daily Avg", format_currency(avg), "Predicted")
        with c3:
            metric_card("Confidence", f"{fc['confidence'].mean()*100:.1f}%", "Model agreement")
        show_table(
            fc[["date", "predicted_revenue", "prediction_lower", "prediction_upper", "trend", "confidence"]],
            filename="home_forecast_table.png",
            key="home_forecast_tbl",
            title="7-Day Forecast"
        )


def page_data_explorer():
    st.markdown("<div class='section-header'>📂 Data Explorer</div>", unsafe_allow_html=True)
    if st.session_state["orders"] is None:
        st.info("Upload files to explore.")
        return
    o = st.session_state["orders"]
    s = st.session_state["stores"]
    p = st.session_state["products"]
    dg = st.session_state["daily_global"]

    st.markdown("### Dataset Statistics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Records", f"{len(o):,}", "Orders")
    with c2: metric_card("Stores", f"{s['store_id'].nunique() if s is not None else 0:,}")
    with c3: metric_card("Products", f"{p['product_id'].nunique() if p is not None else 0:,}")
    with c4: metric_card("Date Range", f"{(dg['date'].max() - dg['date'].min()).days if dg is not None else 0} days")
    with c5: metric_card("Missing Values", f"{o.isnull().sum().sum():,}")

    st.markdown("---")
    st.markdown("### 📈 Daily Revenue Time Series")
    if dg is not None and not dg.empty:
        show_chart(dash.plot_revenue_time_series(dg), "daily_revenue.png", key="daily_revenue")

    st.markdown("### 🔍 Raw Orders Sample")
    show_table(o.head(50), filename="orders_sample.png", key="orders_sample", title="Orders (first 50)")

    if s is not None:
        st.markdown("### 🏪 Stores")
        show_table(s, filename="stores.png", key="stores_tbl", title="Stores")
    if p is not None:
        st.markdown("### 📦 Products")
        show_table(p.head(50), filename="products_sample.png", key="products_tbl", title="Products (first 50)")


def page_weather():
    st.markdown("<div class='section-header'>🌤️ Weather Data</div>", unsafe_allow_html=True)
    w = st.session_state["weather"]
    if w is None or w.empty:
        st.info("No weather data yet. Train models to fetch from NASA POWER.")
        return
    st.markdown(f"**Source:** NASA POWER API | **Records:** {len(w):,} | **Stores covered:** {w['store_id'].nunique()}")
    show_table(w.head(100), filename="weather_sample.png", key="weather_tbl", title="Weather Data (first 100)")
    st.markdown("---")
    st.markdown("### 🌡️ Weather Trends")
    dg = st.session_state["daily_global"]
    if dg is not None:
        w_global = w.groupby("date", as_index=False).agg(
            temp_avg=("temp_avg", "mean"),
            temp_max=("temp_max", "mean"),
            temp_min=("temp_min", "mean"),
            rain=("rain", "sum"),
            wind_speed=("wind_speed", "mean"),
            humidity=("humidity", "mean"),
        )
        w_global["date"] = pd.to_datetime(w_global["date"])
        # FIX: Strip UTC timezone to match weather data format
        dg["date"] = pd.to_datetime(dg["date"], utc=True).dt.tz_localize(None)
        merged = dg.merge(w_global, on="date", how="inner")
        if not merged.empty:
            show_chart(dash.plot_weather_vs_sales(merged), "weather_trends.png", key="weather_trends", height=800)


def page_model_training():
    st.markdown("<div class='section-header'>🧠 Model Training</div>", unsafe_allow_html=True)
    if not st.session_state["trained"]:
        st.info("Click '🚀 Train Models' in the sidebar to start.")
        return
    st.markdown("### Feature Importance")
    fitted = st.session_state["fitted"]
    fcols = st.session_state["feature_cols"]
    for name in st.session_state["top_models"]:
        m = fitted.get(name)
        if m is None:
            continue
        fi = get_feature_importance(m, fcols)
        if fi is not None and not fi.empty:
            st.markdown(f"#### {name}")
            show_chart(
                dash.plot_feature_importance(fi, f"Feature Importance - {name}"),
                filename=f"feature_importance_{name}.png",
                key=f"fi_{name}"
            )

    st.markdown("---")
    st.markdown("### Train / Test Split")
    tr, te = st.session_state["train_df"], st.session_state["test_df"]
    c1, c2 = st.columns(2)
    with c1: metric_card("Train Size", f"{len(tr):,}", "Records")
    with c2: metric_card("Test Size", f"{len(te):,}", "Records")
    show_chart(
        dash.plot_train_test_forecast(tr, te),
        filename="train_test_split.png",
        key="train_test_split"
    )


def page_model_comparison():
    st.markdown("<div class='section-header'>🏆 Model Comparison</div>", unsafe_allow_html=True)
    if not st.session_state["trained"]:
        st.info("Train models first.")
        return

    metrics = st.session_state["metrics"]
    st.markdown("### 📊 Metrics Leaderboard")
    metrics_table = dash.plot_metrics_table(metrics)
    show_table(metrics_table, filename="metrics_leaderboard.png", key="metrics_lb", title="Metrics Leaderboard")
    st.markdown("---")
    show_chart(dash.plot_model_comparison(metrics), "model_comparison.png", key="model_comparison")
    show_chart(dash.plot_leaderboard(metrics), "leaderboard.png", key="leaderboard")

    st.markdown("### 📉 Predictions vs Actual")
    show_chart(
        dash.plot_predictions_vs_actual(st.session_state["test_df"], st.session_state["preds"]),
        "pred_vs_actual.png", key="pred_vs_actual"
    )

    st.markdown("### 📋 Top 3 Models Predictions vs Actual (Last 7 Days of Test Set)")
    test_df = st.session_state["test_df"].tail(7).copy()
    top_models = st.session_state["top_models"]
    preds = st.session_state["preds"]
    df_comp = pd.DataFrame({
        "Date": test_df["date"].dt.date,
        "Actual Revenue": test_df["revenue"].round(2)
    })
    for model_name in top_models:
        if model_name in preds:
            pred_array = preds[model_name]
            df_comp[f"Pred ({model_name})"] = np.round(pred_array[-7:], 2)
    show_table(df_comp, filename="top3_pred_actual.png", key="top3_pred_actual", title="Top 3 Models Predictions vs Actual")

    st.markdown("### 🔍 Residuals")
    show_chart(
        dash.plot_residuals(st.session_state["test_df"], st.session_state["preds"]),
        "residuals.png", key="residuals"
    )

    st.markdown("### 📉 Loss Curves")
    show_chart(
        dash.plot_loss_curve(st.session_state["histories"]),
        "loss_curves.png", key="loss_curves"
    )

    st.markdown("---")
    st.markdown("### 💾 Export Metrics")
    met_df = pd.DataFrame([m.as_dict() for m in metrics])
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Metrics CSV", to_csv_bytes(met_df), file_name="model_metrics.csv", mime="text/csv")
    with c2:
        try:
            img = df_to_png_bytes(metrics_table, title="Model Metrics")
            st.download_button("📥 Download Metrics PNG", data=img, file_name="model_metrics.png", mime="image/png")
        except Exception as e:
            st.warning(f"PNG export failed: {e}")


def page_revenue_forecast():
    st.markdown("<div class='section-header'>💰 Revenue Forecast</div>", unsafe_allow_html=True)
    if not st.session_state["forecasted"]:
        st.info("Click '🔮 Predict Next 7 Days' to generate forecast.")
        return
    fc = st.session_state["forecast_df"]
    st.markdown("### 📈 7-Day Future Forecast")
    show_chart(
        dash.plot_train_test_forecast(st.session_state["train_df"], st.session_state["test_df"], fc),
        filename="revenue_forecast_chart.png", key="revenue_forecast_chart"
    )

    st.markdown("---")
    st.markdown("### 📋 Forecast Table (Ensemble)")
    disp = fc[["date", "predicted_revenue", "prediction_lower", "prediction_upper", "trend", "confidence"]].copy()
    disp["date"] = disp["date"].dt.date
    disp.columns = ["Date", "Predicted Revenue", "Lower Bound", "Upper Bound", "Trend", "Confidence"]
    show_table(disp, filename="forecast_ensemble.png", key="forecast_ensemble", title="Forecast Table (Ensemble)")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total 7-Day Revenue", format_currency(fc["predicted_revenue"].sum()))
    with c2: metric_card("Average Daily", format_currency(fc["predicted_revenue"].mean()))
    with c3: metric_card("Avg Confidence", f"{fc['confidence'].mean()*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🛠️ Next 7 Days Predictions by Top 3 Models")
    future_cols = ["date"] + [c for c in fc.columns if c.startswith("pred_")]
    future_disp = fc[future_cols].copy()
    future_disp["date"] = future_disp["date"].dt.date
    rename_map = {"date": "Date"}
    for c in future_cols:
        if c.startswith("pred_"):
            model_name = c.replace("pred_", "")
            rename_map[c] = f"Prediction ({model_name})"
    future_disp = future_disp.rename(columns=rename_map)
    future_disp["Ensemble (Final)"] = fc["predicted_revenue"].round(2)

    st.dataframe(
        future_disp.style.format({col: "{:.2f} DT" for col in future_disp.columns if col != "Date"}),
        hide_index=True
    )
    try:
        img = df_to_png_bytes(future_disp, title="Next 7 Days Predictions by Top 3 Models")
        st.download_button("📥 Download Table PNG", data=img, file_name="top3_forecast.png", mime="image/png", key="dl_top3_forecast")
    except Exception as e:
        st.warning(f"PNG export failed: {e}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Forecast CSV", to_csv_bytes(fc), file_name="revenue_forecast.csv", mime="text/csv")
    with c2:
        try:
            img = df_to_png_bytes(disp, title="7-Day Revenue Forecast")
            st.download_button("📥 Download Forecast PNG", data=img, file_name="revenue_forecast.png", mime="image/png", key="dl_revenue_fc_png")
        except Exception as e:
            st.warning(f"PNG export failed: {e}")


def page_product_forecast():
    st.markdown("<div class='section-header'>📦 Product Forecast</div>", unsafe_allow_html=True)
    if not st.session_state["forecasted"] or st.session_state["product_forecasts"] is None:
        st.info("Generate forecasts first.")
        return
    pfs = st.session_state["product_forecasts"]
    st.markdown("### 🛒 Top-20 Product Forecasts by Model")
    model_tab = st.tabs(list(pfs.keys()))
    for tab, (name, df) in zip(model_tab, pfs.items()):
        with tab:
            if df is None or df.empty:
                st.warning("No product data available.")
                continue
            show_chart(
                dash.plot_product_forecast(df, name),
                filename=f"product_forecast_{name}.png",
                key=f"pf_chart_{name}"
            )
            show_table(
                df,
                filename=f"product_table_{name}.png",
                key=f"pf_tbl_{name}",
                title=f"Top 20 Products - {name}"
            )
            show_chart(
                dash.plot_treemap_products(df),
                filename=f"product_treemap_{name}.png",
                key=f"pf_tm_{name}"
            )
            show_chart(
                dash.plot_pie_categories(df),
                filename=f"product_pie_{name}.png",
                key=f"pf_pie_{name}"
            )

    st.markdown("---")
    st.markdown("### 💾 Export Product Forecasts")
    all_pfs = pd.concat(list(pfs.values()), ignore_index=True) if pfs else pd.DataFrame()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download All Product Forecasts (CSV)", to_csv_bytes(all_pfs), file_name="product_forecasts.csv", mime="text/csv")
    with c2:
        try:
            img = df_to_png_bytes(all_pfs.head(30), title="All Product Forecasts (top 30)")
            st.download_button("📥 Download All Product Forecasts (PNG)", data=img, file_name="product_forecasts.png", mime="image/png", key="dl_all_pf_png")
        except Exception as e:
            st.warning(f"PNG export failed: {e}")


def page_analytics():
    st.markdown("<div class='section-header'>📈 Analytics</div>", unsafe_allow_html=True)
    dg = st.session_state["daily_global"]
    ds = st.session_state["daily_store"]
    if dg is None:
        st.info("Load data first.")
        return
    st.markdown("### 📊 Revenue Distribution")
    show_chart(dash.plot_revenue_distribution(dg), "revenue_distribution.png", key="rev_dist")
    show_chart(dash.plot_boxplot_by_dow(dg), "boxplot_dow.png", key="boxplot_dow")

    if ds is not None and "store_name" in ds.columns:
        st.markdown("### 🏪 Sales by Store")
        show_chart(dash.plot_sales_by_store(ds), "sales_by_store.png", key="sales_store")

    st.markdown("### 📅 Revenue by Month")
    show_chart(dash.plot_sales_by_month(dg), "sales_by_month.png", key="sales_month")

    if st.session_state["featured"] is not None:
        st.markdown("### 🔥 Correlation Heatmap")
        fcols = st.session_state["feature_cols"]
        show_chart(
            dash.plot_correlation_heatmap(st.session_state["featured"], ["revenue"] + fcols[:15]),
            "correlation_heatmap.png", key="corr_heat", height=700
        )

    st.markdown("### 🗺️ Store Map")
    m = dash.plot_store_map(st.session_state["stores"], ds, st.session_state["forecast_df"])
    if m is not None:
        try:
            from streamlit_folium import st_folium
            st_folium(m, width=900, height=500)
            st.info("ℹ️ Folium maps cannot be exported as PNG directly.")
        except Exception:
            st.warning("Folium map unavailable in this environment.")
    else:
        st.info("No valid coordinates available.")


# ---------------------------------------------------------------- #
# Chat Agent Page - LOCAL LLM                                      #
# ---------------------------------------------------------------- #
def page_bi_agent():
    st.markdown("<div class='section-header'>🤖 Forecast & BI Assistant</div>", unsafe_allow_html=True)
    
    if not st.session_state.get("forecasted"):
        st.info("Please generate a forecast first (Click '🔮 Predict Next 7 Days' in the sidebar).")
        return

    # Create LLM config from session state
    llm_config = LLMConfig(
        backend=st.session_state.get("llm_backend", "ollama"),
        model_name=st.session_state.get("llm_model", "llama3.2:3b"),
        api_base=st.session_state.get("llm_api_base", "http://localhost:11434"),
        temperature=st.session_state.get("llm_temperature", 0.3),
    )

    # Status banner
    c1, c2, c3 = st.columns(3)
    with c1:
        backend_display = {
            "ollama": "🦙 Ollama",
            "openai_compatible": "🔌 Local Server",
            "huggingface": "🤗 HuggingFace"
        }.get(llm_config.backend, llm_config.backend)
        st.info(f"**Backend:** {backend_display}")
    with c2:
        st.info(f"**Model:** `{llm_config.model_name}`")
    with c3:
        st.info(f"**Temperature:** {llm_config.temperature}")

    st.markdown("""
    Welcome to your **100% local AI Business Consultant**. Ask me anything about the 7-day forecast, 
    model confidence, expected product sales, or weather impacts.
    
    **Example Questions:**
    - *Why is the revenue predicted to drop on certain days?*
    - *How confident is the model in next week's forecast?*
    - *What are the top products we should stock up on?*
    - *Explain the R² and MAPE of our best model in simple terms.*
    - *Quel jour aura le plus de chiffre d'affaires?* (French works too!)
    
    > 💡 **No API keys needed** - Everything runs on your machine!
    """)
    
    st.markdown("---")

    # Chat history
    for message in st.session_state.get("bi_chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your forecast, metrics, or products..."):
        # Add user message
        st.session_state["bi_chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("🧠 Thinking locally..."):
                try:
                    response = get_bi_agent_response(
                        user_query=prompt,
                        state=st.session_state,
                        llm_config=llm_config,
                    )
                except Exception as e:
                    response = f"❌ Error: {str(e)}\n\nMake sure your local LLM backend is running and the model is available."
                response_placeholder.markdown(response)
        
        # Add assistant message
        st.session_state["bi_chat_messages"].append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.session_state.get("bi_chat_messages"):
        if st.button("🗑️ Clear Chat History"):
            st.session_state["bi_chat_messages"] = []
            st.rerun()


def page_about():
    st.markdown("<div class='section-header'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("""
    ### 🛒 AI Sales Forecasting Platform

    A **production-grade Machine Learning pipeline** for retail revenue and product forecasting.

    #### 🔧 Tech Stack
    - **Frontend:** Streamlit (dark mode)
    - **ML:** XGBoost, LightGBM, CatBoost, RandomForest, LSTM (TensorFlow)
    - **Weather:** NASA POWER API
    - **Viz:** Plotly, Folium
    - **AI Agent:** 100% Local LLM (Ollama / LM Studio / HuggingFace) - **NO API KEYS REQUIRED**
    - **RAG:** TF-IDF retrieval with sklearn
    - **Export:** PNG (kaleido + matplotlib), CSV

    #### 📐 Pipeline
    1. JSON/CSV ingestion with fuzzy column detection
    2. Data cleaning, daily aggregation, missing date imputation
    3. NASA POWER weather enrichment (cached)
    4. Feature engineering (calendar, lags, rolling, expanding, holidays, weather, popularity)
    5. Chronological train/test split (80/20)
    6. Training of 5 models with early stopping
    7. Auto-ranking → Top-3 models selected
    8. Recursive 7-day forecast with prediction intervals
    9. Top-20 product forecast per best model
    10. Export to CSV **and PNG** ✅

    #### 🤖 BI Assistant (100% Local)
    The AI assistant uses **Retrieval-Augmented Generation (RAG)**:
    ```
    User Question → TF-IDF Retriever → Dashboard Data → Local LLM → Answer
    ```
    
    **Supported Backends:**
    - 🦙 **Ollama** (Recommended) - Easiest setup, good performance
    - 🔌 **LM Studio / LocalAI** - OpenAI-compatible servers
    - 🤗 **HuggingFace Transformers** - Direct model loading

    **Recommended Models:**
    - `llama3.2:3b` - Best balance (2GB, 4GB RAM)
    - `phi3:mini` - Very fast (2.3GB, 4GB RAM)
    - `gemma2:2b` - Google's lightweight (1.6GB, 4GB RAM)

    #### 📊 Metrics
    - MAE, RMSE, MAPE, SMAPE, R², Training time, Inference time

    #### 🎓 Project
    Built as a **final-year engineering project (PFE)**.
    """)


# ---------------------------------------------------------------- #
# Router                                                           #
# ---------------------------------------------------------------- #
PAGES = {
    "🏠 Home": page_home,
    "📂 Data Explorer": page_data_explorer,
    "🌤️ Weather": page_weather,
    "🧠 Model Training": page_model_training,
    "🏆 Model Comparison": page_model_comparison,
    "💰 Revenue Forecast": page_revenue_forecast,
    "📦 Product Forecast": page_product_forecast,
    "📈 Analytics": page_analytics,
    "🤖 BI Assistant": page_bi_agent,
    "ℹ️ About": page_about,
}
PAGES[page]()