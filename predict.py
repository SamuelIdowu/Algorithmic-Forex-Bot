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

def predict_next_movement(symbol, model_path, scaler_path):
    """
    Predict the next movement for a symbol using the trained model.
    """
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
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching recent data for {symbol}...")
    data = get_stock_data(symbol, start_date, end_date, provider='yfinance')
    
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
    
    result = {
        'symbol': symbol,
        'date': latest_date,
        'current_price': latest_data['close'].values[0],
        'prediction': direction,
        'confidence': confidence,
        'history': data_with_features  # Return historical data with features for visualization
    }
    
    return result

def print_prediction(result):
    if not result:
        return

    print("\n" + "="*50)
    print(f"PREDICTION FOR {result['symbol']} (Next Candle)")
    print("="*50)
    print(f"Date: {result['date'].strftime('%Y-%m-%d')}")
    print(f"Close Price: {result['current_price']:.2f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict next market movement')
    parser.add_argument('--symbol', type=str, default='GC=F', help='Symbol to predict (e.g., GC=F for Gold)')
    parser.add_argument('--model_path', type=str, default='models/ml_strategy_model.pkl', help='Path to trained model')
    parser.add_argument('--scaler_path', type=str, default='models/ml_strategy_scaler.pkl', help='Path to scaler')
    
    args = parser.parse_args()
    
    result = predict_next_movement(args.symbol, args.model_path, args.scaler_path)
    print_prediction(result)
