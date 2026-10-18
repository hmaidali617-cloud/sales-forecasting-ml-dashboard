"""
models.py
=========
Five forecasting models with chronological split + early stopping + metrics.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from utils import ModelMetrics, compute_metrics, get_logger

logger = get_logger("models")

# ---------------------------------------------------------------- #
# LSTM Configuration                                                #
# ---------------------------------------------------------------- #
SEQ_LENGTH = 30  # Lookback window for LSTM sequences

# Optional imports
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CB = True
except Exception:
    _HAS_CB = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    _HAS_TF = True
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(42)
except Exception:
    _HAS_TF = False
# models.py - ADD this function to clean infinite values before training

def clean_infinite_values(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove or replace infinite values in features."""
    # Replace inf with large but finite values
    X = np.where(np.isinf(X), np.sign(X) * 1e6, X)
    X = np.where(np.isnan(X), 0, X)
    
    # Clip extreme values
    X = np.clip(X, -1e6, 1e6)
    
    # Remove rows with infinite target
    valid_mask = np.isfinite(y) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y

def chronological_split(df: pd.DataFrame, date_col: str = "date", test_ratio: float = 0.2):
    """Chronological train/test split by date."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


# ---------------------------------------------------------------- #
# Individual model trainers                                        #
# ---------------------------------------------------------------- #
# models.py - UPDATE tree model trainers to use sample weights

def compute_sample_weights(y_train: np.ndarray, target_percentile: float = 75) -> np.ndarray:
    """
    Give higher weight to high-revenue days.
    This forces models to care more about predicting peaks correctly.
    """
    threshold = np.percentile(y_train, target_percentile)
    weights = np.ones_like(y_train)
    
    # Days above 75th percentile get 2x weight
    weights[y_train > threshold] = 2.0
    
    # Days above 90th percentile get 3x weight
    threshold_90 = np.percentile(y_train, 90)
    weights[y_train > threshold_90] = 3.0
    
    # Days above 95th percentile get 4x weight
    threshold_95 = np.percentile(y_train, 95)
    weights[y_train > threshold_95] = 4.0
    
    return weights


def train_xgb(X_train, y_train, X_val, y_val):
    if not _HAS_XGB:
        raise RuntimeError("xgboost not installed")
    
    # Compute sample weights
    sample_weights = compute_sample_weights(y_train)
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.3,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_alpha=0.01,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    # Pass sample_weight to fit
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        sample_weight=sample_weights,  # KEY: Emphasize high-revenue days!
        verbose=False
    )
    return model


# models.py - UPDATE LightGBM training to use DataFrame for feature names

def train_lgbm(X_train, y_train, X_val, y_val, feature_names=None):
    if not _HAS_LGB:
        raise RuntimeError("lightgbm not installed")
    
    sample_weights = compute_sample_weights(y_train)
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=3,
        reg_alpha=0.01,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    # Convert to DataFrame if feature names provided (fixes the warning)
    if feature_names is not None:
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_val_df = pd.DataFrame(X_val, columns=feature_names)
    else:
        X_train_df = X_train
        X_val_df = X_val
    
    model.fit(
        X_train_df, y_train,
        eval_set=[(X_val_df, y_val)],
        sample_weight=sample_weights,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    if not _HAS_CB:
        raise RuntimeError("catboost not installed")
    
    sample_weights = compute_sample_weights(y_train)
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train, 
        eval_set=(X_val, y_val), 
        sample_weight=sample_weights,  # KEY: Emphasize high-revenue days!
        use_best_model=True
    )
    return model


def train_random_forest(X_train, y_train):
    sample_weights = compute_sample_weights(y_train)
    
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)  # KEY!
    return model
# models.py - ADD the missing train_lstm function after train_random_forest

def train_lstm(y_train, y_val, lookback=30, epochs=80):
    """Train LSTM model for time series forecasting."""
    if not _HAS_TF:
        raise RuntimeError("TensorFlow not installed")
    
    # Scale data
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    # Create sequences
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    X_train_seq, y_train_seq = create_sequences(y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(y_val_scaled, lookback)
    
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        raise RuntimeError("Not enough data for LSTM sequences")
    
    # Build model
    model = models.Sequential([
        layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(lookback, 1)),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train with early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, scaler, lookback

def lstm_predict_recursive(model, scaler, history_scaled: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    preds = []
    cur = history_scaled[-lookback:].copy()
    for _ in range(horizon):
        x = cur[-lookback:].reshape(1, lookback, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        cur = np.append(cur, p)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


# ---------------------------------------------------------------- #
# Orchestrator                                                     #
# ---------------------------------------------------------------- #
# models.py - UPDATE train_all_models function - add data cleaning

def train_all_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str = "revenue",
    date_col: str = "date",
    test_ratio: float = 0.2,
) -> Tuple[List[ModelMetrics], Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Train all 5 models, return metrics list, fitted models dict, predictions dict, train, test.
    """
    train_df, test_df = chronological_split(df, date_col, test_ratio)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train or test set is empty (insufficient data)")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target].values.astype(np.float32)

    # ═══════════════════════════════════════════════════════════
    # NEW: Clean infinite/NaN values BEFORE training
    # ═══════════════════════════════════════════════════════════
    X_train, y_train = clean_infinite_values(X_train, y_train)
    X_test, y_test = clean_infinite_values(X_test, y_test)
    
    # Log if any values were cleaned
    if not np.all(np.isfinite(X_train)):
        logger.warning("Cleaned infinite/NaN values in training data")

    metrics_list: List[ModelMetrics] = []
    fitted: Dict = {}
    preds: Dict = {}
    histories: Dict = {}


    # ----- XGBoost -----
    try:
        t0 = time.time()
        m = train_xgb(X_train, y_train, X_test, y_test)
        train_time = time.time() - t0
        t1 = time.time()
        p = m.predict(X_test)
        inf_time = time.time() - t1
        p = np.clip(p, 0, None)
        met = compute_metrics(y_test, p)
        metrics_list.append(ModelMetrics("XGBoost", met["mae"], met["rmse"], met["mape"], met["smape"], met["r2"], train_time, inf_time))
        fitted["XGBoost"] = m
        preds["XGBoost"] = p
        histories["XGBoost"] = m.evals_result() if hasattr(m, "evals_result") else {}
        logger.info("XGBoost trained ✓")
    except Exception as e:
        logger.warning(f"XGBoost failed: {e}")

    # ----- LightGBM -----
    # models.py - UPDATE the LightGBM section in train_all_models

    # ----- LightGBM -----
    try:
        t0 = time.time()
        m = train_lgbm(X_train, y_train, X_test, y_test, feature_names=feature_cols)  # Pass feature names
        train_time = time.time() - t0
        t1 = time.time()
        p = m.predict(X_test)
        inf_time = time.time() - t1
        p = np.clip(p, 0, None)
        met = compute_metrics(y_test, p)
        metrics_list.append(ModelMetrics("LightGBM", met["mae"], met["rmse"], met["mape"], met["smape"], met["r2"], train_time, inf_time))
        fitted["LightGBM"] = m
        preds["LightGBM"] = p
        histories["LightGBM"] = {}
        logger.info("LightGBM trained ✓")
    except Exception as e:
        logger.warning(f"LightGBM failed: {e}")

    # ----- CatBoost -----
    try:
        t0 = time.time()
        m = train_catboost(X_train, y_train, X_test, y_test)
        train_time = time.time() - t0
        t1 = time.time()
        p = m.predict(X_test)
        inf_time = time.time() - t1
        p = np.clip(p, 0, None)
        met = compute_metrics(y_test, p)
        metrics_list.append(ModelMetrics("CatBoost", met["mae"], met["rmse"], met["mape"], met["smape"], met["r2"], train_time, inf_time))
        fitted["CatBoost"] = m
        preds["CatBoost"] = p
        histories["CatBoost"] = m.get_evals_result() if hasattr(m, "get_evals_result") else {}
        logger.info("CatBoost trained ✓")
    except Exception as e:
        logger.warning(f"CatBoost failed: {e}")

    # ----- Random Forest -----
    try:
        t0 = time.time()
        m = train_random_forest(X_train, y_train)
        train_time = time.time() - t0
        t1 = time.time()
        p = m.predict(X_test)
        inf_time = time.time() - t1
        p = np.clip(p, 0, None)
        met = compute_metrics(y_test, p)
        metrics_list.append(ModelMetrics("RandomForest", met["mae"], met["rmse"], met["mape"], met["smape"], met["r2"], train_time, inf_time))
        fitted["RandomForest"] = m
        preds["RandomForest"] = p
        histories["RandomForest"] = {}
        logger.info("RandomForest trained ✓")
    except Exception as e:
        logger.warning(f"RandomForest failed: {e}")

    # ----- LSTM -----
    try:
        t0 = time.time()
        lstm_model, scaler, lookback = train_lstm(y_train, y_test, lookback=SEQ_LENGTH, epochs=80)
        train_time = time.time() - t0
        t1 = time.time()
        hist_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
        p_full = lstm_predict_recursive(lstm_model, scaler, hist_scaled, len(y_test), lookback)
        p = p_full[:len(y_test)]
        inf_time = time.time() - t1
        p = np.clip(p, 0, None)
        met = compute_metrics(y_test, p)
        metrics_list.append(ModelMetrics("LSTM", met["mae"], met["rmse"], met["mape"], met["smape"], met["r2"], train_time, inf_time))
        fitted["LSTM"] = (lstm_model, scaler, lookback)
        preds["LSTM"] = p
        histories["LSTM"] = {}
        logger.info(f"LSTM trained ✓ (SEQ_LENGTH={lookback})")
    except Exception as e:
        logger.warning(f"LSTM failed: {e}")

    if not fitted:
        raise RuntimeError("No model could be trained. Check data quality.")

    return metrics_list, fitted, preds, train_df, test_df


def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance if available."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_booster"):
            scores = model.get_booster().get_score(importance_type="gain")
            imp = np.array([scores.get(f"f{i}", 0) for i in range(len(feature_names))])
        else:
            return None
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(25)
    except Exception:
        return None