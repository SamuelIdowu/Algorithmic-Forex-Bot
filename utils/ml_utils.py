import os

MODELS_DIR = "models"

def get_model_interval(minutes: int) -> str:
    """Map AGENT_INTERVAL_MINUTES to the closest yfinance compatible interval."""
    if minutes >= 1440:
        return "1d"
    if minutes >= 60:
        return f"{minutes // 60}h" if minutes % 60 == 0 else "1h"
    if minutes >= 30:
        return "30m"
    if minutes >= 15:
        return "15m"
    if minutes >= 5:
        return "5m"
    return "1m"

def get_model_paths(symbol: str, interval: str) -> tuple[str, str]:
    """
    Resolve standard model and scaler paths for a symbol and interval.
    Uses the convention: models/{symbol}_{interval}_model.pkl
    (Daily interval '1d' is omitted from the filename for backward compatibility).
    """
    safe_sym = symbol.lower().replace("/", "_").replace("=", "_").replace("-", "-")
    tf_suffix = f"_{interval}" if interval and interval != "1d" else ""

    model_path  = os.path.join(MODELS_DIR, f"{safe_sym}{tf_suffix}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{safe_sym}{tf_suffix}_scaler.pkl")

    return model_path, scaler_path
