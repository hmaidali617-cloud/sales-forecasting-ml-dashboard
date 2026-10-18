"""
Model training and evaluation module.
Implements XGBoost, LightGBM, CatBoost, Random Forest, and LSTM
with early stopping, hyperparameter optimization, and comprehensive evaluation.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import logger, compute_all_metrics, compute_prediction_intervals

# Suppress verbose output from libraries
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    """Stores training results for a single model."""
    name: str
    model: Any
    predictions: np.ndarray
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    residuals_std: float
    feature_importance: Optional[Dict[str, float]] = None
    rank: int = 0


class LSTMModel(nn.Module):
    """PyTorch LSTM model for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Take last timestep
        out = self.dropout(last_out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


class ModelTrainer:
    """Trains and evaluates all five forecasting models."""

    def __init__(self):
        self.results: List[ModelResult] = []
        self.best_models: List[ModelResult] = []
        self.sequence_length: int = 28

    def _chronological_split(
        self, df: pd.DataFrame, test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train and test sets."""
        split_idx = int(len(df) * (1 - test_ratio))
        # Ensure at least 7 days for testing
        split_idx = max(split_idx, len(df) - max(7, int(len(df) * 0.2)))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _prepare_lstm_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len : i])
            y_seq.append(y[i])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> ModelResult:
        """Train XGBoost regressor with early stopping."""
        import xgboost as xgb

        logger.info("Training XGBoost...")
        start_time = time.time()

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            # MOVED HERE: Required for XGBoost >= 2.0
            early_stopping_rounds=50,
        )

        # REMOVED from here: early_stopping_rounds=50
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        infer_start = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - infer_start
        training_time = time.time() - start_time

        metrics = compute_all_metrics(y_test, predictions)
        residuals_std = float(np.std(y_test - predictions))

        importance = dict(zip(
            [f"feat_{i}" for i in range(X_train.shape[1])],
            model.feature_importances_.tolist(),
        ))

        logger.info(f"XGBoost - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R² Score']:.4f}")
        return ModelResult(
            name="XGBoost", model=model, predictions=predictions,
            metrics=metrics, training_time=training_time,
            inference_time=inference_time, residuals_std=residuals_std,
            feature_importance=importance,
        )

    def train_lightgbm(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> ModelResult:
        """Train LightGBM regressor with early stopping."""
        import lightgbm as lgb

        logger.info("Training LightGBM...")
        start_time = time.time()

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        infer_start = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - infer_start
        training_time = time.time() - start_time

        metrics = compute_all_metrics(y_test, predictions)
        residuals_std = float(np.std(y_test - predictions))

        importance = dict(zip(
            [f"feat_{i}" for i in range(X_train.shape[1])],
            model.feature_importances_.tolist(),
        ))

        logger.info(f"LightGBM - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R² Score']:.4f}")
        return ModelResult(
            name="LightGBM", model=model, predictions=predictions,
            metrics=metrics, training_time=training_time,
            inference_time=inference_time, residuals_std=residuals_std,
            feature_importance=importance,
        )

    def train_catboost(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> ModelResult:
        """Train CatBoost regressor with early stopping."""
        from catboost import CatBoostRegressor

        logger.info("Training CatBoost...")
        start_time = time.time()

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            subsample=0.8,
            random_state=42,
            verbose=0,
            allow_writing_files=False,
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False,
        )

        infer_start = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - infer_start
        training_time = time.time() - start_time

        metrics = compute_all_metrics(y_test, predictions)
        residuals_std = float(np.std(y_test - predictions))

        importance = dict(zip(
            [f"feat_{i}" for i in range(X_train.shape[1])],
            model.get_feature_importance().tolist(),
        ))

        logger.info(f"CatBoost - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R² Score']:.4f}")
        return ModelResult(
            name="CatBoost", model=model, predictions=predictions,
            metrics=metrics, training_time=training_time,
            inference_time=inference_time, residuals_std=residuals_std,
            feature_importance=importance,
        )

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> ModelResult:
        """Train Random Forest regressor."""
        logger.info("Training Random Forest...")
        start_time = time.time()

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        infer_start = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - infer_start
        training_time = time.time() - start_time

        metrics = compute_all_metrics(y_test, predictions)
        residuals_std = float(np.std(y_test - predictions))

        importance = dict(zip(
            [f"feat_{i}" for i in range(X_train.shape[1])],
            model.feature_importances_.tolist(),
        ))

        logger.info(f"Random Forest - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R² Score']:.4f}")
        return ModelResult(
            name="Random Forest", model=model, predictions=predictions,
            metrics=metrics, training_time=training_time,
            inference_time=inference_time, residuals_std=residuals_std,
            feature_importance=importance,
        )

    def train_lstm(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> ModelResult:
        """Train PyTorch LSTM model with early stopping."""
        logger.info("Training LSTM...")
        start_time = time.time()

        seq_len = min(self.sequence_length, len(X_train) - 1)
        self.sequence_length = seq_len

        X_train_seq, y_train_seq = self._prepare_lstm_sequences(X_train, y_train, seq_len)
        X_test_seq, y_test_seq = self._prepare_lstm_sequences(
            np.vstack([X_train[-seq_len:], X_test]), np.concatenate([y_train[-seq_len:], y_test]),
            seq_len,
        )

        if len(X_train_seq) < 10:
            logger.warning("Not enough data for LSTM, returning dummy result")
            dummy_pred = np.full(len(y_test), np.mean(y_train))
            metrics = compute_all_metrics(y_test, dummy_pred)
            return ModelResult(
                name="LSTM", model=None, predictions=dummy_pred,
                metrics=metrics, training_time=0.0, inference_time=0.0,
                residuals_std=float(np.std(y_test - dummy_pred)),
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = TensorDataset(
            torch.tensor(X_train_seq), torch.tensor(y_train_seq)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = LSTMModel(
            input_size=X_train_seq.shape[2],
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # FIX: Removed 'verbose=False' for PyTorch 2.2+ compatibility
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 30

        model.train()
        for epoch in range(200):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"LSTM early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Inference
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
            infer_start = time.time()
            predictions = model(X_test_tensor).cpu().numpy()
            inference_time = time.time() - infer_start

        training_time = time.time() - start_time

        # Ensure predictions align with y_test
        predictions = predictions[-len(y_test):]
        if len(predictions) != len(y_test):
            predictions = np.full(len(y_test), np.mean(y_train))

        metrics = compute_all_metrics(y_test, predictions)
        residuals_std = float(np.std(y_test - predictions))

        logger.info(f"LSTM - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R² Score']:.4f}")
        return ModelResult(
            name="LSTM", model=model, predictions=predictions,
            metrics=metrics, training_time=training_time,
            inference_time=inference_time, residuals_std=residuals_std,
        )

    def train_all_models(
        self, df: pd.DataFrame, feature_columns: List[str],
        target_col: str = "total_revenue", test_ratio: float = 0.2,
    ) -> List[ModelResult]:
        """Train all five models and return results."""
        logger.info(f"Training all models on {len(df)} samples with {len(feature_columns)} features")

        train_df, test_df = self._chronological_split(df, test_ratio)
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        # ── Safely extract features and handle inf/nan ──
        X_train_df = train_df[feature_columns].copy()
        X_test_df = test_df[feature_columns].copy()
        
        y_train = train_df[target_col].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)

        # FIX: Replace infinity with NaN, then fill NaN with 0
        X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train_df.fillna(0, inplace=True)
        X_test_df.fillna(0, inplace=True)

        # Clip extreme finite values to prevent overflow in LSTM/trees
        X_train_df = X_train_df.clip(-1e6, 1e6)
        X_test_df = X_test_df.clip(-1e6, 1e6)

        X_train = X_train_df.values.astype(np.float32)
        X_test = X_test_df.values.astype(np.float32)

        # Ensure targets have no NaN/Inf
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

        self.results = []

        # Train tree-based models
        self.results.append(self.train_xgboost(X_train, y_train, X_test, y_test))
        self.results.append(self.train_lightgbm(X_train, y_train, X_test, y_test))
        self.results.append(self.train_catboost(X_train, y_train, X_test, y_test))
        self.results.append(self.train_random_forest(X_train, y_train, X_test, y_test))

        # Train LSTM
        self.results.append(self.train_lstm(X_train, y_train, X_test, y_test))

        # Rank models: best RMSE first, then R²
        self.results.sort(
            key=lambda r: (r.metrics["RMSE"], -r.metrics["R² Score"])
        )
        for i, result in enumerate(self.results):
            result.rank = i + 1

        # Select top 3
        self.best_models = self.results[:3]

        logger.info("Model ranking complete:")
        for r in self.results:
            logger.info(
                f"  #{r.rank} {r.name}: RMSE={r.metrics['RMSE']:.4f}, "
                f"R²={r.metrics['R² Score']:.4f}"
            )

        return self.results

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Return model metrics as a DataFrame."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for r in self.results:
            record = {
                "Rank": r.rank,
                "Model": r.name,
                **r.metrics,
                "Training Time (s)": round(r.training_time, 2),
                "Inference Time (s)": round(r.inference_time, 4),
            }
            records.append(record)
        return pd.DataFrame(records)

    def get_feature_importance_dataframe(
        self, feature_columns: List[str], model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Return feature importance as a DataFrame."""
        result = None
        if model_name:
            result = next((r for r in self.results if r.name == model_name), None)
        elif self.best_models:
            result = self.best_models[0]

        if result is None or result.feature_importance is None:
            return pd.DataFrame()

        # Map generic feature names to actual names
        importance = {}
        for key, val in result.feature_importance.items():
            idx = int(key.replace("feat_", ""))
            if idx < len(feature_columns):
                importance[feature_columns[idx]] = val
            else:
                importance[key] = val

        df = pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in importance.items()]
        )
        return df.sort_values("Importance", ascending=False)