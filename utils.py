"""
Utility functions for AI Sales Forecasting application.
Provides logging setup, metrics computation, confidence intervals,
and styling helpers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def setup_logging() -> logging.Logger:
    """Configure and return application logger."""
    logger = logging.getLogger("SalesForecast")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logging()


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning 0 on division by zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (handles zeros)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics at once."""
    return {
        "MAE": round(compute_mae(y_true, y_pred), 4),
        "RMSE": round(compute_rmse(y_true, y_pred), 4),
        "MAPE (%)": round(compute_mape(y_true, y_pred), 2),
        "SMAPE (%)": round(compute_smape(y_true, y_pred), 2),
        "R² Score": round(compute_r2(y_true, y_pred), 4),
    }


def compute_prediction_intervals(
    predictions: np.ndarray,
    residuals_std: float,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals using residual standard deviation.
    Returns (lower_bound, upper_bound).
    """
    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * residuals_std
    lower = predictions - margin
    upper = predictions + margin
    return np.maximum(lower, 0), upper


def compute_confidence_score(r2: float, mape: float) -> str:
    """Convert R² and MAPE into a human-readable confidence label."""
    if r2 >= 0.85 and mape < 15:
        return "High"
    elif r2 >= 0.6 and mape < 30:
        return "Medium"
    elif r2 >= 0.3:
        return "Low"
    return "Very Low"


def get_dark_theme_css() -> str:
    """Return custom CSS for dark mode styling."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6C63FF;
        --primary-light: #8B85FF;
        --bg-primary: #0E1117;
        --bg-secondary: #161B22;
        --bg-card: #1C2333;
        --text-primary: #E6EDF3;
        --text-secondary: #8B949E;
        --border: #30363D;
        --success: #3FB950;
        --warning: #D29922;
        --danger: #F85149;
        --info: #58A6FF;
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }

    .main {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    p, span, label, div {
        color: var(--text-secondary) !important;
    }

    .stDataFrame {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .stDataFrame td, .stDataFrame th {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: var(--bg-secondary) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
    }

    div.stDownloadButton > button:first-child {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    .kpi-card {
        background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .section-header {
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.8rem;
    }

    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); color: #000; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #A0522D); color: #fff; }

    iframe {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    .stSelectbox label, .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    div[data-testid="stSidebar"] .stSelectbox,
    div[data-testid="stSidebar"] .stFileUploader {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 1rem 0 !important;
    }
    </style>
    """


def find_best_column(
    df: pd.DataFrame, candidates: List[str], exact_only: bool = False
) -> Optional[str]:
    """
    Find the best matching column name from a list of candidates.
    Tries exact match first, then case-insensitive, then substring.
    """
    columns = df.columns.tolist()

    for c in candidates:
        if c in columns:
            return c

    if not exact_only:
        lower_cols = {col.lower(): col for col in columns}
        for c in candidates:
            if c.lower() in lower_cols:
                return lower_cols[c.lower()]

        for c in candidates:
            for col in columns:
                if c.lower() in col.lower() or col.lower() in c.lower():
                    return col

    return None


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousand separators and specified decimals."""
    if pd.isna(value) or value == 0:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"