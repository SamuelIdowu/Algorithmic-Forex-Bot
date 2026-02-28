import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from utils.data_loader import get_stock_data
from utils.features import add_technical_features
from train_model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_next_movement(symbol, model_path, scaler_path, **kwargs):
    """
    Predict the next movement for a symbol using the trained model.
    """
    # Extract kwargs
    lookback = kwargs.get('lookback', 100)
    interval = kwargs.get('interval', '1d')
    
    # Load model and scaler
    model, scaler = load_model(model_path, scaler_path)
    if model is None or scaler is None:
        logger.error("Failed to load model or scaler")
        return

    # Fetch recent data (enough to calculate features)
    # We need at least 50 days for features like SMA, RSI, etc.
    # Fetching last 100 days to be safe
    # Set end_date to tomorrow to ensure we get the latest data (including today if available)
    end_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Adjust start date based on interval and lookback
    if interval == '1d':
        days_back = lookback * 2 # Buffer
    else:
        # Approximate for intraday
        days_back = 59 # Default to 59 days buffer (limit is strictly < 60d for intraday data)
        
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching recent data for {symbol} (Interval: {interval})...")
    data = get_stock_data(symbol, start_date, end_date, provider='yfinance', interval=interval)
    
    if data.empty:
        logger.error("No data found")
        return

    # Add features
    logger.info("Calculating technical features...")
    data_with_features = add_technical_features(data)
    
    if data_with_features.empty:
        logger.error("Not enough data to calculate features")
        return

    # Get the latest data point (the most recent closed candle)
    latest_data = data_with_features.iloc[[-1]]
    latest_date = latest_data.index[0]
    
    logger.info(f"Making prediction based on data from {latest_date}")
    
    # Prepare features for prediction
    # Use feature_names_in_ from scaler if available to ensure correct order and selection
    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = scaler.feature_names_in_
        # Check if all required columns are present
        missing_cols = [col for col in feature_cols if col not in latest_data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            return
        features = latest_data[feature_cols]
    else:
        # Fallback to previous logic if feature_names_in_ is not available (older sklearn versions)
        feature_cols = [col for col in latest_data.columns 
                       if col not in ['target', 'future_close'] and pd.api.types.is_numeric_dtype(latest_data[col])]
        features = latest_data[feature_cols]
    
    # Check feature mismatch
    if features.shape[1] != scaler.n_features_in_:
        logger.warning(f"Feature mismatch: Model expects {scaler.n_features_in_}, got {features.shape[1]}")
        return

    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    direction = "UP" if prediction == 1 else "DOWN"
    confidence = probabilities[prediction]
    
    # Calculate Trade Specs
    entry_price = latest_data['close'].values[0]
    atr = latest_data['atr'].values[0] if 'atr' in latest_data.columns else 0
    
    tp = 0.0
    sl = 0.0
    
    # Prioritize kwargs, then args, then default
    tp_mult = kwargs.get('tp_mult', getattr(args, 'tp_mult', 3.0) if 'args' in globals() else 3.0)
    sl_mult = kwargs.get('sl_mult', getattr(args, 'sl_mult', 2.0) if 'args' in globals() else 2.0)
    
    if direction == "UP":
        tp = entry_price + (tp_mult * atr)
        sl = entry_price - (sl_mult * atr)
    else:
        tp = entry_price - (tp_mult * atr)
        sl = entry_price + (sl_mult * atr)
        
    holding_time = "1 Candle" # Generic based on interval

    result = {
        'symbol': symbol,
        'date': latest_date,
        'current_price': entry_price,
        'prediction': direction,
        'confidence': confidence,
        'entry': entry_price,
        'tp': tp,
        'sl': sl,
        'holding_time': holding_time,
        'history': data_with_features,
        'tp_mult': tp_mult,
        'sl_mult': sl_mult
    }
    
    return result

def print_prediction(result):
    if not result:
        return

    print("\n" + "="*50)
    print(f"PREDICTION FOR {result['symbol']} (Next Candle)")
    print("="*50)
    print(f"Date: {result['date']}")
    print(f"Current Price: {result['current_price']:.5f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("-" * 50)
    print("TRADE SPECIFICATIONS (ESTIMATED)")
    print(f"Entry: {result['entry']:.5f}")
    print(f"Take Profit (TP): {result['tp']:.5f} ({result['tp_mult']}x ATR)")
    print(f"Stop Loss (SL): {result['sl']:.5f} ({result['sl_mult']}x ATR)")
    print(f"Holding Time: {result['holding_time']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict next market movement')
    parser.add_argument('--symbol', type=str, default='GC=F', help='Symbol to predict (e.g., GC=F for Gold)')
    parser.add_argument('--model_path', type=str, default='models/ml_strategy_model.pkl', help='Path to trained model')
    parser.add_argument('--scaler_path', type=str, default='models/ml_strategy_scaler.pkl', help='Path to scaler')
    parser.add_argument('--lookback', type=int, default=100, help='Lookback period in days/candles')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1d, 1h, etc.)')
    parser.add_argument('--sl_mult', type=float, default=2.0, help='Stop Loss ATR Multiplier')
    parser.add_argument('--tp_mult', type=float, default=3.0, help='Take Profit ATR Multiplier')
    
    args = parser.parse_args()
    
    # We need to pass args to predict_next_movement so it can use them
    # Refactoring signature or attaching to function
    # Let's change signature to accept **kwargs
    result = predict_next_movement(args.symbol, args.model_path, args.scaler_path, 
                                 lookback=args.lookback, interval=args.interval, 
                                 sl_mult=args.sl_mult, tp_mult=args.tp_mult)
    print_prediction(result)
