"""
Quant Analyst — Phase 3

Loads the symbol-specific trained RandomForest model (.pkl) and generates
a BUY / SELL signal with a confidence score for each symbol.

Supports hot-reload via reload_model(symbol) so the self-retraining module
(Phase 9) can swap in a fresh model without restarting the agent loop.
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from threading import Lock

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

from utils.ml_utils import get_model_paths, get_model_interval, MODELS_DIR

FALLBACK_MODEL  = "ml_strategy_model.pkl"
FALLBACK_SCALER = "ml_strategy_scaler.pkl"


class QuantAnalyst(BaseAgent):
    """
    ML-based quant analyst. Maps each symbol to its trained RandomForest
    model and returns BUY/SELL signals with confidence scores.
    """

    name = "QuantAnalyst"
    role = "analyst"
    priority = 2  # Runs after MarketDataAnalyst

    # Exclude these column names when building feature vectors
    _NON_FEATURE_COLS = {"target", "future_close", "symbol"}

    def __init__(self):
        super().__init__()
        # {symbol: (model, scaler)} — lazily loaded on first use
        self._model_cache: dict[str, tuple] = {}
        self._lock = Lock()   # protects cache during hot-reload

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """
        Reads context["market_data"] and writes context["quant"][symbol].

        Output per symbol:
            {
                "quant_signal":     "BUY" | "SELL",
                "quant_confidence": float (0-1),
                "reasoning":        str,
            }
        """
        market_data: dict = context.get("market_data", {})
        quant_results: dict = {}

        for symbol, data in market_data.items():
            try:
                result = self._predict(symbol, data)
                quant_results[symbol] = result
                logger.info(
                    f"[QuantAnalyst] {symbol}: {result['quant_signal']} "
                    f"(conf={result['quant_confidence']:.3f})"
                )
            except Exception as exc:
                logger.error(f"[QuantAnalyst] Error for {symbol}: {exc}", exc_info=True)
                quant_results[symbol] = {
                    "quant_signal": "HOLD", 
                    "quant_confidence": 0.0,
                    "reasoning": f"Error: {exc}"
                }

        context["quant"] = quant_results
        return context

    def reload_model(self, symbol: str) -> bool:
        """
        Hot-swap the .pkl files for a symbol without restarting the loop.
        Called by the Retrainer agent (Phase 9) after retraining completes.

        Returns True if reload succeeded, False otherwise.
        """
        with self._lock:
            if symbol in self._model_cache:
                del self._model_cache[symbol]
                logger.info(f"[QuantAnalyst] Evicted cached model for {symbol}")

        # Force a reload on next prediction by calling _load_model
        try:
            self._load_model(symbol)
            logger.info(f"[QuantAnalyst] Hot-reload successful for {symbol}")
            return True
        except Exception as exc:
            logger.error(f"[QuantAnalyst] Hot-reload failed for {symbol}: {exc}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self, symbol: str, interval: str = None) -> tuple:
        """Load (and cache) the model+scaler for a symbol.

        Standard resolution order:
          1. Shared get_model_paths() logic (e.g. models/btc-usd_1h_model.pkl)
          2. Legacy names (e.g. models/btc-usd_model.pkl)
          3. Generic fallback (ml_strategy_model.pkl)
        """
        cache_key = f"{symbol}_{interval}" if interval else symbol
        with self._lock:
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

        # 1. Standard dynamic path (Interval-aware)
        if interval is None:
            from utils.config import AGENT_INTERVAL_MINUTES
            interval = get_model_interval(AGENT_INTERVAL_MINUTES)
        
        std_model, std_scaler = get_model_paths(symbol, interval)
        
        # Build candidate list
        candidates = [(std_model, std_scaler)]
        
        # 2. Add legacy/short names if they exist and are different
        safe_sym = symbol.lower().replace("/", "_").replace("=", "_").replace("-", "-")
        legacy_model = os.path.join(MODELS_DIR, f"{safe_sym}_model.pkl")
        legacy_scaler = os.path.join(MODELS_DIR, f"{safe_sym}_scaler.pkl")
        
        if legacy_model != std_model:
            candidates.append((legacy_model, legacy_scaler))

        # 3. Generic fallback
        candidates.append((os.path.join(MODELS_DIR, FALLBACK_MODEL), os.path.join(MODELS_DIR, FALLBACK_SCALER)))

        model_path = scaler_path = None
        for mp, sp in candidates:
            if os.path.exists(mp) and os.path.exists(sp):
                model_path, scaler_path = mp, sp
                break

        if model_path is None:
            logger.warning(
                f"[QuantAnalyst] No model file found for {symbol} (interval={interval}). "
                "Returning None to signal that bootstrapping may be needed."
            )
            return None, None

        logger.info(f"[QuantAnalyst] Loading model for {symbol}: {model_path}")
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Single-core inference is faster for 1-row prediction
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        with self._lock:
            self._model_cache[cache_key] = (model, scaler)

        return model, scaler

    def _predict(self, symbol: str, data: dict) -> dict:
        """Run inference for a single symbol."""
        df_feat: pd.DataFrame = data.get("features")
        if df_feat is None or df_feat.empty:
            return {"quant_signal": "HOLD", "quant_confidence": 0.0}

        model, scaler = self._load_model(symbol)
        if model is None or scaler is None:
            return {"quant_signal": "HOLD", "quant_confidence": 0.0}

        # Build feature row (last row only)
        feature_cols = [
            c for c in df_feat.columns
            if c not in self._NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df_feat[c])
        ]
        row = df_feat[feature_cols].iloc[-1:]

        # Guard: feature count mismatch
        if len(feature_cols) != scaler.n_features_in_:
            # Try to align columns the scaler was fitted on
            if hasattr(scaler, "feature_names_in_"):
                expected = list(scaler.feature_names_in_)
                row = row.reindex(columns=expected, fill_value=0.0)
            else:
                logger.warning(
                    f"[QuantAnalyst] Feature mismatch for {symbol}: "
                    f"expected {scaler.n_features_in_}, got {len(feature_cols)}"
                )
                return {"quant_signal": "HOLD", "quant_confidence": 0.0}

        row_scaled = scaler.transform(row)
        prediction = int(model.predict(row_scaled)[0])
        proba      = model.predict_proba(row_scaled)[0]
        confidence = float(np.max(proba))

        signal = "BUY" if prediction == 1 else "SELL"
        
        # Generate qualitative reasoning for the CIO
        rsi = row.get("rsi", [0]).iloc[0]
        sma_ratio = row.get("sma_ratio", [1]).iloc[0]
        macd_hist = row.get("macd_hist", [0]).iloc[0]
        
        reasoning = (
            f"RandomForest predicted {signal} with {confidence:.1%} confidence. "
            f"Key metrics: RSI is {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}), "
            f"Price/SMA20 ratio is {sma_ratio:.3f}, MACD Histogram is {macd_hist:.4f}."
        )

        return {
            "quant_signal": signal, 
            "quant_confidence": confidence,
            "reasoning": reasoning
        }
