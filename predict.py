import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from utils.data_loader import get_yfinance_data
from utils.features import add_technical_features
from train_model import load_model
import utils.config as config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_model_path(symbol: str, model_path: str, scaler_path: str, interval: str):
    """
    Try to find the actual .pkl files when the explicit path doesn't exist.

    Resolution order:
      1. Explicit path (as given)
      2. Interval-suffixed  e.g. btc-usd_1h_model.pkl  (exact interval match)
      3. For sub-hourly (1m/5m/15m/30m) — the 1h model (same feature distributions)
      4. Auto-derived daily  e.g. btc-usd_model.pkl
      5. Generic fallback    ml_strategy_model.pkl
    """
    import os
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return model_path, scaler_path

    safe_sym = symbol.lower().replace("/", "_").replace("=", "_")
    models_dir = os.path.dirname(model_path) or "models"

    candidates = []
    # Intraday-suffixed variant (exact match first)
    if interval and interval != "1d":
        candidates.append((
            os.path.join(models_dir, f"{safe_sym}_{interval}_model.pkl"),
            os.path.join(models_dir, f"{safe_sym}_{interval}_scaler.pkl"),
        ))
    # For sub-hourly intervals, prefer the 1h model over the daily model.
    # The daily model produces all-NaN features when applied to intraday OHLCV data.
    if interval in ("1m", "5m", "15m", "30m"):
        candidates.append((
            os.path.join(models_dir, f"{safe_sym}_1h_model.pkl"),
            os.path.join(models_dir, f"{safe_sym}_1h_scaler.pkl"),
        ))
    # Plain daily variant
    candidates.append((
        os.path.join(models_dir, f"{safe_sym}_model.pkl"),
        os.path.join(models_dir, f"{safe_sym}_scaler.pkl"),
    ))
    # Generic fallback
    candidates.append((
        os.path.join(models_dir, "ml_strategy_model.pkl"),
        os.path.join(models_dir, "ml_strategy_scaler.pkl"),
    ))

    for mp, sp in candidates:
        if os.path.exists(mp) and os.path.exists(sp):
            logger.warning(
                f"[predict] '{model_path}' not found — using '{mp}' instead."
            )
            return mp, sp

    # Nothing found — return originals so the caller can trigger auto-train
    return model_path, scaler_path


def _auto_train(symbol: str, interval: str, model_path: str, scaler_path: str) -> bool:
    """
    Automatically train a model for the given symbol/interval when no model exists.
    Returns True if training succeeded, False otherwise.
    """
    logger.info(f"[predict] No model found for {symbol} ({interval}). Auto-training now...")
    try:
        from train_model import train_model
        import pandas as pd

        now = pd.Timestamp.now()

        # Determine training window — Yahoo Finance caps intraday at ~59 days
        _INTRADAY = {"1m", "2m", "5m", "15m", "30m", "1h", "60m", "90m"}
        if interval in _INTRADAY:
            start_date = (now - pd.Timedelta(days=58)).strftime("%Y-%m-%d")
        else:
            start_date = (now - pd.Timedelta(days=1825)).strftime("%Y-%m-%d")  # 5 years for daily

        end_date = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"[predict] Auto-training {symbol} ({interval}) from {start_date} to {end_date}...")
        result = train_model(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            model_path=model_path,
            scaler_path=scaler_path,
            interval=interval,
        )
        if result is not None:
            logger.info(f"[predict] Auto-training complete for {symbol} ({interval}).")
            return True
        else:
            logger.error(f"[predict] Auto-training failed for {symbol} ({interval}).")
            return False
    except Exception as e:
        logger.error(f"[predict] Auto-training exception for {symbol} ({interval}): {e}")
        return False


def _get_agent_insights(symbol: str, interval: str, current_price: float, atr: float) -> dict:
    """
    Run multi-agent analysis pipeline to get insights from all analysts.
    Returns a dict with quant, sentiment, fundamentals, CIO, and risk manager insights.
    """
    try:
        from agents.market_data_analyst import MarketDataAnalyst
        from agents.quant_analyst import QuantAnalyst
        from agents.sentiment_analyst import SentimentAnalyst
        from agents.fundamentals_analyst import FundamentalsAnalyst
        from agents.chief_investment_officer import ChiefInvestmentOfficer
        from agents.risk_manager import RiskManager

        # Build context similar to run_agent.py
        context = {
            "symbols": [symbol],
        }

        # Run agents in sequence
        market_data = MarketDataAnalyst()
        context = market_data.run(context)

        quant = QuantAnalyst()
        context = quant.run(context)

        sentiment = SentimentAnalyst()
        context = sentiment.run(context)

        fundamentals = FundamentalsAnalyst()
        context = fundamentals.run(context)

        cio = ChiefInvestmentOfficer()
        context = cio.run(context)

        risk = RiskManager()
        context = risk.run(context)

        # Extract insights
        quant_data = context.get("quant", {}).get(symbol, {})
        sentiment_data = context.get("sentiment", {}).get(symbol, {})
        fundamentals_data = context.get("fundamentals", {}).get(symbol, {})
        cio_data = context.get("cio", {}).get(symbol, {})
        risk_data = context.get("risk", {}).get(symbol, {})

        return {
            "quant": {
                "signal": quant_data.get("quant_signal", "N/A"),
                "confidence": quant_data.get("quant_confidence", 0),
            },
            "sentiment": {
                "signal": sentiment_data.get("sentiment_signal", "N/A"),
                "score": sentiment_data.get("sentiment_score", 0),
            },
            "fundamentals": {
                "signal": fundamentals_data.get("fundamentals_signal", "N/A"),
            },
            "cio": {
                "signal": cio_data.get("cio_signal", "N/A"),
                "memo": cio_data.get("memo", ""),
            },
            "risk_manager": {
                "action": risk_data.get("action", "HOLD"),
                "vote_score": risk_data.get("vote_score", 0),
                "weights": risk_data.get("weights_used", {}),
            },
        }
    except Exception as e:
        logger.error(f"[predict] Agent insights failed for {symbol}: {e}")
        return None


def predict_next_movement(symbol, model_path, scaler_path, **kwargs):
    """
    Predict the next movement for a symbol using the trained model.
    If no model exists, automatically triggers training first.
    Optionally includes multi-agent insights.
    """
    # Extract kwargs
    lookback = kwargs.get('lookback', 100)
    interval = kwargs.get('interval', '1d')
    include_insights = kwargs.get('include_insights', True)

    # Resolve paths — handle interval-suffix mismatch automatically
    model_path, scaler_path = _resolve_model_path(symbol, model_path, scaler_path, interval)

    # Load model and scaler — auto-train if missing
    import os
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning(f"[predict] No model found at '{model_path}'. Triggering auto-train...")
        trained_ok = _auto_train(symbol, interval, model_path, scaler_path)
        if not trained_ok:
            logger.error(f"[predict] Auto-train failed for {symbol} ({interval}). Cannot predict.")
            return None

    model, scaler = load_model(model_path, scaler_path)
    if model is None or scaler is None:
        logger.error("Failed to load model or scaler")
        return None

    # Fetch recent data (enough to calculate features and warm up indicators)
    now = pd.Timestamp.now()
    end_date = (now + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Calculate days_back based on interval
    if interval == '1d':
        days_back = lookback + 100
    elif interval == '1h':
        days_back = (lookback // 24) + 14  # ~2 weeks
    elif interval == '15m':
        days_back = (lookback // 96) + 7   # ~1 week
    elif interval == '5m':
        days_back = 7
    elif interval == '1m':
        days_back = 7  # yfinance max for 1m is 7 days; need all for indicator warmup
    else:
        days_back = 30  # Default safety

    start_date = (now - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')

    logger.info(f"Fetching recent data for {symbol} (Interval: {interval}, Start: {start_date})...")
    data = pd.DataFrame()
    for attempt in range(1, 4):  # 3 attempts to handle transient yfinance rate-limits
        data = get_yfinance_data(symbol, start_date, end_date, interval=interval)
        if not data.empty:
            break
        if attempt < 3:
            import time
            logger.warning(f"[predict] Empty data for {symbol} ({interval}), retry {attempt}/3 in 5s...")
            time.sleep(5)

    if data.empty:
        logger.error(f"No data found for {symbol} ({interval}) after 3 attempts — yfinance may be rate-limiting.")
        return None

    if len(data) < 30:
        logger.warning(f"Very few bars ({len(data)}) returned for {symbol}. Technical indicators may fail.")

    # Add features
    logger.info(f"Calculating technical features for {len(data)} bars...")
    data_with_features = add_technical_features(data)

    if data_with_features.empty:
        logger.error(
            f"Not enough data to calculate features for {symbol} (Input rows: {len(data)}). "
            f"Try retraining a model specifically for interval '{interval}' or use a longer timeframe."
        )
        return None

    # Get the latest VALID data point (last non-NaN row after feature engineering)
    latest_data = data_with_features.iloc[[-1]]
    latest_date = latest_data.index[0]

    logger.info(f"Making prediction based on data from {latest_date}")

    # Prepare features for prediction
    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = scaler.feature_names_in_
        missing_cols = [col for col in feature_cols if col not in latest_data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data for prediction: {missing_cols}")
            return None
        features = latest_data[feature_cols]
    else:
        # Fallback for older scalers without feature_names_in_
        feature_cols = [col for col in latest_data.columns
                        if col not in ['target', 'future_close'] and pd.api.types.is_numeric_dtype(latest_data[col])]
        features = latest_data[feature_cols]

    # Check feature count mismatch
    if features.shape[1] != scaler.n_features_in_:
        logger.warning(f"Feature mismatch: Model expects {scaler.n_features_in_}, got {features.shape[1]}")
        return None

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    direction = "UP" if prediction == 1 else "DOWN"
    confidence = probabilities[prediction]

    # Calculate Trade Specs
    entry_price = latest_data['close'].values[0]
    atr = latest_data['atr'].values[0] if 'atr' in latest_data.columns else 0
    rsi = latest_data['rsi'].values[0] if 'rsi' in latest_data.columns else 0
    bb_pos = latest_data['bb_position'].values[0] if 'bb_position' in latest_data.columns else 0.5

    tp_mult = kwargs.get('tp_mult', config.ATR_TP_MULT)
    sl_mult = kwargs.get('sl_mult', config.ATR_SL_MULT)

    if direction == "UP":
        tp = entry_price + (tp_mult * atr)
        sl = entry_price - (sl_mult * atr)
    else:
        tp = entry_price - (tp_mult * atr)
        sl = entry_price + (sl_mult * atr)

    result = {
        'symbol': symbol,
        'date': latest_date,
        'current_price': entry_price,
        'prediction': direction,
        'confidence': confidence,
        'entry': entry_price,
        'tp': tp,
        'sl': sl,
        'holding_time': "1 Candle",
        'history': data_with_features,
        'tp_mult': tp_mult,
        'sl_mult': sl_mult,
        'rsi': rsi,
        'atr': atr,
        'bb_pos': bb_pos,
        'reasoning': f"{direction} signal with {confidence:.1%} confidence. RSI={rsi:.1f}, ATR={atr:.4f}.",
    }

    # Get multi-agent insights (if enabled)
    if include_insights:
        try:
            insights = _get_agent_insights(symbol, interval, entry_price, atr)
            result['agent_insights'] = insights
        except Exception as e:
            logger.warning(f"[predict] Could not fetch agent insights: {e}")
            result['agent_insights'] = None

    # Log to prediction tracker
    try:
        from services.prediction_tracker_service import PredictionTrackerService
        tracker_service = PredictionTrackerService()
        tracker_id = tracker_service.create_tracker_from_prediction(
            prediction_result=result,
            source="standalone",
            prediction_id=None  # Will be linked if caller provides it
        )
        result['tracker_id'] = tracker_id
        logger.info(f"[predict] Tracker created for {symbol}: ID={tracker_id}")
    except Exception as e:
        logger.warning(f"[predict] Could not create tracker entry: {e}")
        result['tracker_id'] = None

    return result


# ── CLI helpers ─────────────────────────────────────────────────────────────

_RST  = "\033[0m"
_BOLD = "\033[1m"
_GRN  = "\033[92m"
_RED  = "\033[91m"
_YEL  = "\033[93m"
_CYN  = "\033[96m"
_W    = 54


def _conf_bar(conf: float, width: int = 12) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


def _pct(base: float, target: float) -> str:
    if not base:
        return "N/A"
    p = (target - base) / base * 100
    return f"{'+' if p >= 0 else ''}{p:.2f}%"


def _rr(entry: float, sl: float, tp: float) -> str:
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    return f"1 : {reward/risk:.2f}" if risk else "N/A"


def _fmt(price: float) -> str:
    return config.format_price(price)


def print_prediction(result):
    if not result:
        return

    direction = result['prediction']
    conf      = result['confidence']
    entry     = result['entry']
    tp        = result['tp']
    sl        = result['sl']

    dir_colour = _GRN if direction == "UP" else _RED
    signal_lbl = "📈 LONG / BUY" if direction == "UP" else "📉 SHORT / SELL"
    bar        = _conf_bar(conf)
    tp_pct     = _pct(entry, tp)
    sl_pct     = _pct(entry, sl)
    rr_str     = _rr(entry, sl, tp)

    W = _W
    top    = f"╔{'═'*(W+2)}╗"
    bot    = f"╚{'═'*(W+2)}╝"
    div_eq = f"╠{'═'*(W+2)}╣"
    div_da = f"╠{'─'*(W+2)}╣"

    def row(text):
        import re
        visible_len = len(re.sub(r'\033\[[0-9;]*m', '', text))
        pad = W - visible_len
        return f"║  {text}{' ' * max(pad, 0)}║"

    lines = [
        "",
        top,
        row(f"{_BOLD}{_CYN}  PREDICTION: {result['symbol']}  ·  Next Candle{_RST}"),
        div_eq,
        row(f"  Date          │ {result['date']}"),
        row(f"  Current Price │ {_fmt(result['current_price'])}"),
        div_da,
        row(f"  Signal        │ {dir_colour}{_BOLD}{signal_lbl}{_RST}"),
        row(f"  Confidence    │ {bar}  {conf:.1%}"),
        div_da,
        row("  TRADE SPECIFICS (estimations based on ATR)"),
        row(f"  Entry          │ {_fmt(entry)}"),
        row(f"  Take Profit    │ {_fmt(tp)}  ({_GRN}{tp_pct}{_RST})  [{result['tp_mult']}× ATR]"),
        row(f"  Stop Loss      │ {_fmt(sl)}  ({_RED}{sl_pct}{_RST})  [{result['sl_mult']}× ATR]"),
        row(f"  Risk : Reward  │ {rr_str}"),
        row(f"  Holding Time   │ {result['holding_time']}"),
    ]

    if conf < 0.60:
        lines.append(div_da)
        lines.append(row(f"  ⚠️  {_YEL}LOW CONFIDENCE — treat as indicative only{_RST}"))

    # ── Agent Insights Section ────────────────────────────────────────────
    insights = result.get('agent_insights')
    if insights:
        lines.append(div_eq)
        lines.append(row(f"{_BOLD}{_CYN}  AI AGENT INSIGHTS{_RST}"))
        lines.append(div_da)

        # Quant Analyst
        quant = insights.get("quant", {})
        q_signal = quant.get("signal", "N/A")
        q_conf = quant.get("confidence", 0)
        q_color = _GRN if q_signal == "BUY" else (_RED if q_signal == "SELL" else _YEL)
        lines.append(row(f"  🤖 Quant        │ {q_color}{_BOLD}{q_signal:<6}{_RST}  conf {_conf_bar(q_conf, 8)}"))

        # Sentiment Analyst
        sentiment = insights.get("sentiment", {})
        s_signal = sentiment.get("signal", "N/A")
        s_score = sentiment.get("score", 0)
        s_color = _GRN if "BULL" in s_signal.upper() else (_RED if "BEAR" in s_signal.upper() else _YEL)
        lines.append(row(f"  📰 Sentiment    │ {s_color}{_BOLD}{s_signal:<6}{_RST}  score {s_score:.3f}"))

        # Fundamentals Analyst
        fundamentals = insights.get("fundamentals", {})
        f_signal = fundamentals.get("signal", "N/A")
        f_color = _GRN if "BULL" in f_signal.upper() else (_RED if "BEAR" in f_signal.upper() else _YEL)
        lines.append(row(f"  📊 Fundamentals │ {f_color}{_BOLD}{f_signal:<6}{_RST}"))

        # CIO
        cio = insights.get("cio", {})
        cio_signal = cio.get("signal", "N/A")
        cio_memo = cio.get("memo", "")
        c_color = _GRN if "BULL" in cio_signal.upper() else (_RED if "BEAR" in cio_signal.upper() else _YEL)
        lines.append(row(f"  🎯 CIO          │ {c_color}{_BOLD}{cio_signal:<6}{_RST}"))

        if cio_memo:
            lines.append(div_da)
            # Wrap memo to fit box
            import textwrap
            memo_lines = textwrap.wrap(cio_memo, width=W - 16)
            lines.append(row(f"  {_BOLD}CIO Memo:{_RST}"))
            for memo_line in memo_lines[:3]:  # Show max 3 lines
                lines.append(row(f"    {_DIM}{memo_line}{_RST}"))
            if len(memo_lines) > 3:
                lines.append(row(f"    {_DIM}...{_RST}"))

        # Risk Manager
        risk = insights.get("risk_manager", {})
        r_action = risk.get("action", "HOLD")
        r_score = risk.get("vote_score", 0)
        r_color = _GRN if r_action == "BUY" else (_RED if r_action == "SELL" else _YEL)
        lines.append(div_da)
        lines.append(row(f"  ⚖️  Risk Mgr     │ {r_color}{_BOLD}{r_action:<6}{_RST}  score {r_score:.3f}"))

        # Analyst Weights
        weights = risk.get("weights", {})
        if weights:
            lines.append(row(f"    {_DIM}Weights: Q:{weights.get('quant', 0):.0%} S:{weights.get('sentiment', 0):.0%} F:{weights.get('fundamentals', 0):.0%}{_RST}"))

    lines += [bot, ""]
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict next market movement')
    parser.add_argument('--symbol',      type=str,   default='GC=F',                          help='Symbol to predict')
    parser.add_argument('--model_path',  type=str,   default='models/ml_strategy_model.pkl',  help='Path to trained model')
    parser.add_argument('--scaler_path', type=str,   default='models/ml_strategy_scaler.pkl', help='Path to scaler')
    parser.add_argument('--lookback',    type=int,   default=100,                             help='Lookback period')
    parser.add_argument('--interval',    type=str,   default='1d',                            help='Data interval')
    parser.add_argument('--sl_mult',     type=float, default=2.0,                             help='Stop Loss ATR Multiplier')
    parser.add_argument('--tp_mult',     type=float, default=3.0,                             help='Take Profit ATR Multiplier')

    args = parser.parse_args()
    result = predict_next_movement(
        args.symbol, args.model_path, args.scaler_path,
        lookback=args.lookback, interval=args.interval,
        sl_mult=args.sl_mult, tp_mult=args.tp_mult,
    )
    print_prediction(result)