"""
chat_agent.py
==============
RAG (Retrieval-Augmented Generation) BI Assistant - 100% LOCAL
Builds a knowledge base from session state, retrieves relevant chunks
via TF-IDF, and sends them to a LOCAL LLM (Ollama/HuggingFace) for
accurate, data-grounded responses.

NO API KEYS REQUIRED - Everything runs on your machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from utils import get_logger, format_currency

logger = get_logger("chat-agent")


# ================================================================== #
#                        LOCAL LLM CLIENT                            #
# ================================================================== #
@dataclass
class LLMConfig:
    """Configuration for local LLM backend."""
    backend: Literal["ollama", "huggingface", "openai_compatible"] = "ollama"
    model_name: str = "llama3.2:3b"
    api_base: str = "http://localhost:11434"
    hf_model_id: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    max_new_tokens: int = 768
    temperature: float = 0.3
    top_p: float = 0.9


class LocalLLM:
    """
    Unified interface for LOCAL LLM backends.
    
    Supported backends:
    - ollama: Ollama server (recommended, easiest setup)
    - huggingface: Direct model loading via transformers
    - openai_compatible: LM Studio, LocalAI, text-generation-webui
    
    Example usage:
        >>> llm = LocalLLM(LLMConfig(backend="ollama", model_name="llama3.2:3b"))
        >>> response = llm.chat([{"role": "user", "content": "Hello!"}])
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._pipeline = None
        self._tokenizer = None
        self._chat_template = None
        
        if config.backend == "huggingface":
            self._load_hf_model()
    
    def _load_hf_model(self):
        """Load HuggingFace model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires:\n"
                "  pip install transformers torch accelerate"
            )
        
        model_id = self.config.hf_model_id or "meta-llama/Llama-3.2-3B-Instruct"
        logger.info(f"Loading HuggingFace model: {model_id}")
        
        # Determine device
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipeline = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._pipeline = self._pipeline.to(device)
        
        self._device = device
        logger.info(f"HuggingFace model loaded on {device}")
    
    def _ollama_chat(self, messages: List[Dict]) -> str:
        """Generate using Ollama API."""
        url = f"{self.config.api_base}/api/chat"
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_new_tokens,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.api_base}.\n"
                "Make sure Ollama is running: https://ollama.ai"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.config.model_name}' not found in Ollama.\n"
                    f"Pull it first: ollama pull {self.config.model_name}"
                )
            raise
    
    def _openai_compatible_chat(self, messages: List[Dict]) -> str:
        """Generate using OpenAI-compatible API (LM Studio, LocalAI, etc.)."""
        url = f"{self.config.api_base}/v1/chat/completions"
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "top_p": self.config.top_p,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to server at {self.config.api_base}.\n"
                "Make sure your local LLM server is running."
            )
    
    def _huggingface_chat(self, messages: List[Dict]) -> str:
        """Generate using HuggingFace model directly."""
        import torch
        
        # Try to use apply_chat_template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = self._fallback_format(messages)
        else:
            prompt = self._fallback_format(messages)
        
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._pipeline.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                top_p=self.config.top_p if self.config.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    
    def _fallback_format(self, messages: List[Dict]) -> str:
        """Fallback message formatting if no chat template."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            if role == "System":
                parts.append(f"### System:\n{msg['content']}")
            elif role == "User":
                parts.append(f"### User:\n{msg['content']}")
            else:
                parts.append(f"### Assistant:\n{msg['content']}")
        parts.append("### Assistant:\n")
        return "\n\n".join(parts)
    
    def chat(self, messages: List[Dict]) -> str:
        """Generate a response from chat messages."""
        if self.config.backend == "ollama":
            return self._ollama_chat(messages)
        elif self.config.backend == "huggingface":
            return self._huggingface_chat(messages)
        elif self.config.backend == "openai_compatible":
            return self._openai_compatible_chat(messages)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        if self.config.backend == "ollama":
            try:
                resp = requests.get(f"{self.config.api_base}/api/tags", timeout=3)
                return resp.status_code == 200
            except:
                return False
        elif self.config.backend == "huggingface":
            return self._pipeline is not None
        elif self.config.backend == "openai_compatible":
            try:
                resp = requests.get(f"{self.config.api_base}/v1/models", timeout=3)
                return resp.status_code == 200
            except:
                return False
        return False
    
    def list_models(self) -> List[str]:
        """List available models for the backend."""
        if self.config.backend == "ollama":
            try:
                resp = requests.get(f"{self.config.api_base}/api/tags", timeout=5)
                if resp.status_code == 200:
                    return [m["name"] for m in resp.json().get("models", [])]
            except:
                pass
        elif self.config.backend == "openai_compatible":
            try:
                resp = requests.get(f"{self.config.api_base}/v1/models", timeout=5)
                if resp.status_code == 200:
                    return [m["id"] for m in resp.json().get("data", [])]
            except:
                pass
        return []


# ================================================================== #
#                     CACHED LLM INSTANCE                            #
# ================================================================== #
_cached_llm: Optional[LocalLLM] = None
_llm_config_hash: Optional[int] = None


def get_llm(config: LLMConfig) -> LocalLLM:
    """Get or create cached LLM instance."""
    global _cached_llm, _llm_config_hash
    
    config_hash = hash((
        config.backend,
        config.model_name,
        config.api_base,
        config.hf_model_id,
        config.device,
        config.temperature,
    ))
    
    if _cached_llm is None or _llm_config_hash != config_hash:
        _cached_llm = LocalLLM(config)
        _llm_config_hash = config_hash
        logger.info(f"Created new LLM instance: {config.backend}/{config.model_name}")
    
    return _cached_llm


def reset_llm_cache():
    """Reset the cached LLM instance (e.g., when changing models)."""
    global _cached_llm, _llm_config_hash
    _cached_llm = None
    _llm_config_hash = None


# ================================================================== #
#                     KNOWLEDGE BASE BUILDER                         #
# ================================================================== #
def build_knowledge_base(state: Dict) -> List[Dict]:
    """
    Convert session state into a list of text documents (chunks)
    that can be searched via TF-IDF.
    Each chunk has 'text' and 'source' fields.
    """
    docs = []
    
    # --- 1. Data Overview ---
    orders = state.get("orders")
    stores = state.get("stores")
    products = state.get("products")
    dg = state.get("daily_global")
    
    if orders is not None and not orders.empty:
        docs.append({
            "text": (
                f"DATA OVERVIEW: The dataset contains {len(orders):,} orders. "
                f"Date range: {orders['date'].min().date()} to {orders['date'].max().date()} "
                f"({(orders['date'].max() - orders['date'].min()).days} days). "
                f"Total revenue: {format_currency(orders['amount'].sum())}. "
                f"Average order value: {format_currency(orders['amount'].mean())}. "
                f"Stores: {stores['store_id'].nunique() if stores is not None else 0}. "
                f"Products: {products['product_id'].nunique() if products is not None else 0}."
            ),
            "source": "data_overview"
        })
    
    # --- 2. Daily Revenue Statistics ---
    if dg is not None and not dg.empty:
        dg_copy = dg.copy()
        dg_copy["dayofweek"] = pd.to_datetime(dg_copy["date"]).dt.day_name()
        dow_stats = dg_copy.groupby("dayofweek")["revenue"].agg(["mean", "max", "min", "std"]).round(2)
        
        dow_text_parts = []
        for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            if dow in dow_stats.index:
                row = dow_stats.loc[dow]
                dow_text_parts.append(
                    f"{dow}: avg={format_currency(row['mean'])}, max={format_currency(row['max'])}, min={format_currency(row['min'])}"
                )
        
        docs.append({
            "text": (
                f"DAILY REVENUE BY DAY OF WEEK: "
                + " | ".join(dow_text_parts)
                + f". Overall daily average: {format_currency(dg['revenue'].mean())}. "
                f"Overall daily max: {format_currency(dg['revenue'].max())}. "
                f"Overall daily min: {format_currency(dg['revenue'].min())}. "
                f"Daily revenue std dev: {dg['revenue'].std():.2f}."
            ),
            "source": "daily_revenue_stats"
        })
        
        # Monthly trend
        dg_copy["month"] = pd.to_datetime(dg_copy["date"]).dt.to_period("M").astype(str)
        monthly = dg_copy.groupby("month")["revenue"].sum().round(2)
        last_3_months = monthly.tail(3)
        docs.append({
            "text": (
                f"MONTHLY REVENUE TREND: Last 3 months - "
                + ", ".join([f"{m}: {format_currency(v)}" for m, v in last_3_months.items()])
                + f". Best month ever: {monthly.idxmax()} ({format_currency(monthly.max())}). "
                f"Worst month ever: {monthly.idxmin()} ({format_currency(monthly.min())})."
            ),
            "source": "monthly_trend"
        })
    
    # --- 3. Model Metrics ---
    metrics = state.get("metrics")
    if metrics:
        metrics_text_parts = []
        for m in metrics:
            metrics_text_parts.append(
                f"{m.model_name}: RMSE={m.rmse:.2f}, MAE={m.mae:.2f}, MAPE={m.mape:.2f}%, "
                f"SMAPE={m.smape:.2f}%, R²={m.r2:.4f}, "
                f"Train time={m.train_time_sec:.2f}s, Inference time={m.inference_time_sec:.4f}s"
            )
        
        best = min(metrics, key=lambda x: x.rmse)
        worst = max(metrics, key=lambda x: x.rmse)
        best_r2 = max(metrics, key=lambda x: x.r2)
        
        docs.append({
            "text": (
                f"MODEL PERFORMANCE COMPARISON: " + " | ".join(metrics_text_parts)
                + f". Best model by RMSE: {best.model_name} (RMSE={best.rmse:.2f}). "
                f"Best model by R²: {best_r2.model_name} (R²={best_r2.r2:.4f}). "
                f"Worst model by RMSE: {worst.model_name} (RMSE={worst.rmse:.2f}). "
                f"Total models trained: {len(metrics)}."
            ),
            "source": "model_metrics"
        })
        
        # Individual model cards for precise retrieval
        for m in metrics:
            docs.append({
                "text": (
                    f"MODEL DETAILS - {m.model_name}: "
                    f"RMSE={m.rmse:.2f} (root mean squared error, lower is better). "
                    f"MAE={m.mae:.2f} (mean absolute error, lower is better). "
                    f"MAPE={m.mape:.2f}% (mean absolute percentage error, lower is better). "
                    f"R²={m.r2:.4f} (coefficient of determination, 1.0 is perfect). "
                    f"Training took {m.train_time_sec:.2f} seconds. "
                    f"Inference took {m.inference_time_sec:.4f} seconds."
                ),
                "source": f"model_{m.model_name}"
            })
    
    # --- 4. Forecast Data ---
    fc = state.get("forecast_df")
    if fc is not None and not fc.empty:
        fc_dates = fc["date"].dt.strftime("%Y-%m-%d (%A)").tolist()
        fc_values = fc["predicted_revenue"].round(2).tolist()
        fc_trends = fc["trend"].tolist()
        fc_confidences = (fc["confidence"] * 100).round(1).tolist()
        fc_lower = fc["prediction_lower"].round(2).tolist()
        fc_upper = fc["prediction_upper"].round(2).tolist()
        
        forecast_lines = []
        for i in range(len(fc)):
            forecast_lines.append(
                f"{fc_dates[i]}: {format_currency(fc_values[i])} [{format_currency(fc_lower[i])} - {format_currency(fc_upper[i])}] "
                f"trend={fc_trends[i]}, confidence={fc_confidences[i]}%"
            )
        
        total_fc = fc["predicted_revenue"].sum()
        avg_fc = fc["predicted_revenue"].mean()
        avg_conf = fc["confidence"].mean() * 100
        
        docs.append({
            "text": (
                f"7-DAY REVENUE FORECAST: " + " | ".join(forecast_lines)
                + f". Total forecasted revenue: {format_currency(total_fc)}. "
                f"Daily average forecast: {format_currency(avg_fc)}. "
                f"Average confidence: {avg_conf:.1f}%. "
                f"Highest forecast day: {fc_dates[fc_values.index(max(fc_values))]} ({format_currency(max(fc_values))}). "
                f"Lowest forecast day: {fc_dates[fc_values.index(min(fc_values))]} ({format_currency(min(fc_values))})."
            ),
            "source": "forecast_summary"
        })
        
        # Per-model predictions if available
        pred_cols = [c for c in fc.columns if c.startswith("pred_")]
        if pred_cols:
            per_model_lines = []
            for col in pred_cols:
                model_name = col.replace("pred_", "")
                model_total = fc[col].sum()
                per_model_lines.append(f"{model_name}: {format_currency(model_total)}")
            docs.append({
                "text": (
                    f"FORECAST BY MODEL: " + " | ".join(per_model_lines)
                    + f". The final ensemble prediction is a weighted average of top models, "
                    f"with boldness adjustments for recent trends."
                ),
                "source": "forecast_by_model"
            })
    
    # --- 5. Product Forecasts ---
    pfs = state.get("product_forecasts")
    if pfs:
        for model_name, pf_df in pfs.items():
            if pf_df is None or pf_df.empty:
                continue
            top_5 = pf_df.head(5)
            product_lines = []
            for _, row in top_5.iterrows():
                product_lines.append(
                    f"{row['product_name']} ({row['category']}): "
                    f"qty={row['predicted_quantity_7d']:.0f}, "
                    f"revenue={format_currency(row['expected_revenue_7d'])}, "
                    f"confidence={row['confidence']*100:.0f}%"
                )
            docs.append({
                "text": (
                    f"TOP 5 PRODUCT FORECAST ({model_name}): " + " | ".join(product_lines)
                    + f". Total products forecasted: {len(pf_df)}."
                ),
                "source": f"product_forecast_{model_name}"
            })
    
    # --- 6. Feature Importance (if available) ---
    fitted = state.get("fitted")
    fcols = state.get("feature_cols")
    if fitted and fcols:
        from models import get_feature_importance
        for name in (state.get("top_models") or []):
            m = fitted.get(name)
            if m is None:
                continue
            fi = get_feature_importance(m, fcols)
            if fi is not None and not fi.empty:
                top_10 = fi.head(10)
                fi_lines = [f"{row['feature']}={row['importance']:.4f}" for _, row in top_10.iterrows()]
                docs.append({
                    "text": (
                        f"FEATURE IMPORTANCE ({name}): Top 10 features - " + " | ".join(fi_lines)
                        + f". These features drive the {name} model's predictions the most."
                    ),
                    "source": f"feature_importance_{name}"
                })
    
    # --- 7. Generic Knowledge ---
    docs.append({
        "text": (
            "METHODOLOGY: The pipeline uses 5 ML models (XGBoost, LightGBM, CatBoost, RandomForest, LSTM). "
            "Data is split chronologically (80% train, 20% test). "
            "Features include calendar (day of week, month, holidays), lags (1-28 days), "
            "rolling statistics (7/14 day windows), expanding mean, weather data from NASA POWER, "
            "peak detection features, volatility features, and momentum features. "
            "The forecast uses an ensemble of top-3 models with boldness adjustments "
            "based on recent trend direction and volatility. "
            "RMSE measures average prediction error in original units. "
            "R² measures how much variance is explained (1.0 = perfect). "
            "MAPE measures percentage error."
        ),
        "source": "methodology"
    })
    
    docs.append({
        "text": (
            "BUSINESS INTERPRETATION GUIDE: "
            "RMSE < 10% of average daily revenue = excellent accuracy. "
            "RMSE 10-20% = good accuracy. "
            "RMSE > 20% = needs improvement. "
            "R² > 0.8 = model explains most patterns. "
            "R² 0.5-0.8 = moderate explanatory power. "
            "R² < 0.5 = model struggles with patterns. "
            "MAPE < 10% = highly accurate. "
            "MAPE 10-20% = reasonable. "
            "MAPE > 20% = high error rate. "
            "Confidence > 80% = models agree well. "
            "Confidence < 70% = models disagree, be cautious."
        ),
        "source": "interpretation_guide"
    })
    
    logger.info(f"Built knowledge base with {len(docs)} documents")
    return docs


# ================================================================== #
#                         TF-IDF RETRIEVER                           #
# ================================================================== #
class Retriever:
    """TF-IDF based retriever for RAG - runs 100% locally."""
    
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.vectorizer = None
        self.tfidf_matrix = None
        
        if documents:
            self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index from documents."""
        texts = [doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF index built: {self.tfidf_matrix.shape[0]} docs x {self.tfidf_matrix.shape[1]} terms")
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k most relevant documents for a query."""
        if not self.documents or self.vectorizer is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum similarity threshold
                results.append({
                    "text": self.documents[idx]["text"],
                    "source": self.documents[idx]["source"],
                    "score": float(similarities[idx])
                })
        
        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results


# ================================================================== #
#                    CACHED RETRIEVER INSTANCE                       #
# ================================================================== #
_cached_retriever: Optional[Retriever] = None
_cached_state_hash: Optional[int] = None


def _get_state_hash(state: Dict) -> int:
    """Create a simple hash to detect if session state changed."""
    try:
        fc = state.get("forecast_df")
        fc_hash = hash(tuple(fc["predicted_revenue"].tolist())) if fc is not None and not fc.empty else 0
        metrics = state.get("metrics")
        m_hash = hash(tuple((m.rmse, m.r2) for m in metrics)) if metrics else 0
        return hash((fc_hash, m_hash, state.get("trained", False), state.get("forecasted", False)))
    except Exception:
        return 0


# ================================================================== #
#                        RAG GENERATOR                               #
# ================================================================== #
# System prompt for the BI assistant
SYSTEM_PROMPT = """You are an expert Business Intelligence Assistant for a retail sales forecasting dashboard. 

RULES:
1. ONLY use the provided context chunks to answer questions about numbers, metrics, and data.
2. If the context doesn't contain the answer, say "I don't have that specific data available" - NEVER make up numbers.
3. Always explain metrics in simple business terms when asked.
4. Format numbers as currency when discussing revenue (e.g., "1,234.56 DT").
5. Be concise but thorough. Use bullet points for comparisons.
6. When explaining model performance, reference the specific metrics from context.
7. Speak in the same language as the user's question (English, French, Arabic, etc.).
8. If asked about methodology or "how it works", explain the pipeline steps from context.
9. Keep responses focused and avoid unnecessary preamble."""


def get_bi_agent_response(
    user_query: str,
    state: Dict,
    llm_config: Optional[LLMConfig] = None,
) -> str:
    """
    RAG-based BI Agent: Retrieve relevant context, then generate answer using LOCAL LLM.
    
    Pipeline (all local):
    1. User Question
    2. TF-IDF Retriever → finds relevant dashboard data chunks
    3. Local LLM (Ollama/HuggingFace) → generates answer from chunks
    
    Args:
        user_query: The user's question
        state: Session state containing forecasts, metrics, etc.
        llm_config: Configuration for the local LLM backend
    
    Returns:
        Generated response string
    """
    global _cached_retriever, _cached_state_hash
    
    # Default config
    if llm_config is None:
        llm_config = LLMConfig()
    
    # Check if forecast exists
    if not state.get("forecasted"):
        return "⚠️ Please generate a forecast first by clicking **🔮 Predict Next 7 Days** in the sidebar."
    
    # Build or reuse retriever
    current_hash = _get_state_hash(state)
    if _cached_retriever is None or _cached_state_hash != current_hash:
        docs = build_knowledge_base(state)
        _cached_retriever = Retriever(docs)
        _cached_state_hash = current_hash
    
    # Retrieve relevant context
    retrieved = _cached_retriever.retrieve(user_query, top_k=4)
    
    if not retrieved:
        context = "No specific data chunks matched this query. Answer based on general knowledge about retail BI dashboards if relevant, otherwise say you don't have that data."
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            context_parts.append(f"[Chunk {i} - Source: {chunk['source']}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)
    
    # Build user prompt
    user_prompt = f"""CONTEXT (retrieved from dashboard data):
{context}

USER QUESTION: {user_query}

ANSWER:"""
    
    # Get LLM instance
    try:
        llm = get_llm(llm_config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return _fallback_response(retrieved, str(e))
    
    # Check availability
    if not llm.is_available():
        return _unavailable_response(llm_config, retrieved)
    
    # Generate response
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        answer = llm.chat(messages)
        
        # Append sources
        if retrieved:
            sources = ", ".join(set(chunk["source"] for chunk in retrieved))
            answer += f"\n\n*📄 Sources: {sources}*"
        
        return answer
        
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return _unavailable_response(llm_config, retrieved)
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return _fallback_response(retrieved, str(e))


def _fallback_response(retrieved: List[Dict], error: str) -> str:
    """Return raw context when LLM fails."""
    if retrieved:
        return (
            "⚠️ **LLM Error:** " + error[:200] + "\n\n"
            "Here are the relevant data chunks I found:\n\n"
            + "\n---\n".join(
                f"**📊 {chunk['source']}** (relevance: {chunk['score']:.1%})\n{chunk['text']}"
                for chunk in retrieved
            )
        )
    return f"❌ Error: {error}"


def _unavailable_response(config: LLMConfig, retrieved: List[Dict]) -> str:
    """Return helpful message when LLM backend is not available."""
    if config.backend == "ollama":
        help_msg = (
            "🦙 **Ollama not available.**\n\n"
            "**To fix:**\n"
            "1. Install Ollama: https://ollama.ai\n"
            "2. Run: `ollama pull llama3.2:3b`\n"
            "3. Make sure Ollama is running\n\n"
            f"Looking for model: `{config.model_name}`"
        )
    elif config.backend == "huggingface":
        help_msg = (
            "🤗 **HuggingFace model not loaded.**\n\n"
            "**To fix:**\n"
            "1. Install: `pip install transformers torch accelerate`\n"
            "2. Ensure you have enough RAM/VRAM for the model\n"
            f"Model: `{config.hf_model_id or 'default'}`"
        )
    else:
        help_msg = (
            f"🔌 **Server not available** at `{config.api_base}`\n\n"
            "Make sure your local LLM server is running (LM Studio, LocalAI, etc.)"
        )
    
    if retrieved:
        help_msg += "\n\n---\n**Relevant data found (no LLM to process it):**\n\n"
        help_msg += "\n---\n".join(
            f"**{chunk['source']}** ({chunk['score']:.0%})\n{chunk['text'][:300]}..."
            for chunk in retrieved[:2]
        )
    
    return help_msg


# ================================================================== #
#                     UTILITY FUNCTIONS                              #
# ================================================================== #
def check_local_llm_status(
    backend: str = "ollama",
    api_base: str = "http://localhost:11434",
) -> Dict:
    """
    Check the status of local LLM backends.
    Returns a dict with status info for UI display.
    """
    result = {
        "ollama": {"available": False, "models": []},
        "openai_compatible": {"available": False, "models": []},
    }
    
    # Check Ollama
    try:
        resp = requests.get(f"{api_base}/api/tags", timeout=3)
        if resp.status_code == 200:
            result["ollama"]["available"] = True
            result["ollama"]["models"] = [m["name"] for m in resp.json().get("models", [])]
    except:
        pass
    
    # Check OpenAI-compatible (common ports)
    for port in [1234, 8080, 5000]:  # LM Studio, LocalAI, text-gen-webui
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if resp.status_code == 200:
                result["openai_compatible"]["available"] = True
                result["openai_compatible"]["url"] = f"http://localhost:{port}"
                result["openai_compatible"]["models"] = [m["id"] for m in resp.json().get("data", [])]
                break
        except:
            pass
    
    return result


def get_recommended_models() -> List[Dict]:
    """Get list of recommended models for different hardware."""
    return [
        {
            "name": "llama3.2:3b",
            "size": "~2GB",
            "ram": "4GB",
            "description": "Best balance of speed and quality",
            "backend": "ollama",
            "pull_cmd": "ollama pull llama3.2:3b",
        },
        {
            "name": "phi3:mini",
            "size": "~2.3GB",
            "ram": "4GB",
            "description": "Microsoft's compact model, very fast",
            "backend": "ollama",
            "pull_cmd": "ollama pull phi3:mini",
        },
        {
            "name": "gemma2:2b",
            "size": "~1.6GB",
            "ram": "4GB",
            "description": "Google's lightweight model",
            "backend": "ollama",
            "pull_cmd": "ollama pull gemma2:2b",
        },
        {
            "name": "llama3.2:1b",
            "size": "~1.3GB",
            "ram": "2GB",
            "description": "Ultra-lightweight, fastest option",
            "backend": "ollama",
            "pull_cmd": "ollama pull llama3.2:1b",
        },
        {
            "name": "mistral:7b",
            "size": "~4.7GB",
            "ram": "8GB",
            "description": "Better quality, needs more RAM",
            "backend": "ollama",
            "pull_cmd": "ollama pull mistral:7b",
        },
    ]