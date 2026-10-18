"""
utils.py
========
Centralized utilities: logging, metrics, caching, helpers.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------- #
# Logging                                                          #
# ---------------------------------------------------------------- #
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sales-forecast")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------- #
# Directories                                                      #
# ---------------------------------------------------------------- #
CACHE_DIR = "cache"
MODELS_DIR = "models"
EXPORTS_DIR = "exports"
for d in (CACHE_DIR, MODELS_DIR, EXPORTS_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- #
# Metrics                                                          #
# ---------------------------------------------------------------- #
@dataclass
class ModelMetrics:
    """Container for evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    train_time_sec: float
    inference_time_sec: float

    def as_dict(self) -> Dict:
        return self.__dict__


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, SMAPE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE with safety
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smask = denom > 1e-6
    smape = float(np.mean(2 * np.abs(y_pred[smask] - y_true[smask]) / denom[smask]) * 100) if smask.sum() else float("nan")
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2}


def rank_models(metrics_list, primary="rmse", secondary="r2"):
    """Rank models by RMSE ascending then R² descending."""
    return sorted(
        metrics_list,
        key=lambda m: (getattr(m, primary), -getattr(m, secondary)),
    )


# ---------------------------------------------------------------- #
# Misc helpers                                                     #
# ---------------------------------------------------------------- #
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-9 else default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def format_currency(v: float) -> str:
    try:
        return f"{v:,.2f} DT"
    except Exception:
        return str(v)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")