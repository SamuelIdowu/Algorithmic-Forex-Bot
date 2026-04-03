import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from utils.data_loader import get_yfinance_data
from utils.features import add_technical_features, prepare_training_data
from data.db_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(
        symbol: str = "AAPL",
        start_date: str = "2020-01-01",
        end_date: str = "2025-01-01",
        model_path: str = None,
        scaler_path: str = None,
        tune: bool = False,
        interval: str = "1d",
):
    """
    Train a RandomForest model for a given symbol and candle interval.

    Args:
        symbol:     Yahoo Finance ticker (e.g. "BTC-USD").
        start_date: Training window start "YYYY-MM-DD".
        end_date:   Training window end   "YYYY-MM-DD".
        model_path: Where to save the .pkl (auto-named if None).
        scaler_path:Where to save the scaler .pkl (auto-named if None).
        tune:       Run RandomizedSearchCV hyperparameter tuning.
        interval:   Candle size: "1d" (default), "1h", "30m", "15m", "5m".
    """
    # ── Auto-name model files to include the interval ─────────────────────
    safe_sym = symbol.lower().replace("/", "_").replace("=", "_").replace("-", "-")
    tf_suffix = f"_{interval}" if interval != "1d" else ""
    if model_path is None:
        model_path  = f"models/{safe_sym}{tf_suffix}_model.pkl"
    if scaler_path is None:
        scaler_path = f"models/{safe_sym}{tf_suffix}_scaler.pkl"

    logger.info(f"Training {symbol} on {interval} candles → {model_path}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ── Warn about Yahoo Finance history limits for intraday ──────────────
    _INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    if interval in _INTRADAY:
        from datetime import datetime, timedelta
        _max_days = 59  # Yahoo Finance allows up to ~60 days; stay 1 day inside the limit
        _cutoff = (datetime.utcnow() - timedelta(days=_max_days)).strftime("%Y-%m-%d")
        if start_date < _cutoff:
            logger.warning(
                f"Yahoo Finance only provides ~60 days of {interval} history. "
                f"Clamping start_date from {start_date} to {_cutoff}."
            )
            start_date = _cutoff
        else:
            logger.warning(
                f"Yahoo Finance only provides ~60 days of {interval} history. "
                "Model will train on that window only."
            )

    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    data = get_yfinance_data(symbol, start_date, end_date, interval=interval)
    
    if data.empty:
        logger.error(f"No data found for {symbol} in the specified date range")
        return None
    
    logger.info(f"Loaded {len(data)} rows of data for {symbol}")
    
    # Add technical features
    logger.info("Adding technical features to the data")
    data_with_features = add_technical_features(data)
    
    if data_with_features.empty:
        logger.error("No data after adding technical features")
        return None
        
    logger.info(f"Data shape after feature engineering: {data_with_features.shape}")
    
    # Prepare training data
    logger.info("Preparing training data with target variable")
    X, y = prepare_training_data(data_with_features)
    
    if X.empty or y.empty:
        logger.error("No valid training data after target preparation")
        return None
    
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Split data into train and test sets
    # Use shuffle=False for time series data to prevent look-ahead bias in the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    if tune:
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        
        # Define parameter grid
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        model = random_search.best_estimator_
    else:
        # Default parameters
        logger.info("Using default hyperparameters")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    logger.info("Making predictions on test set")
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 15 Most Important Features:")
    logger.info(feature_importance.head(15))
    
    logger.info(f"Model training completed successfully. Model saved to {model_path}")
    
    return model, scaler, accuracy


def load_model(model_path: str = "models/ml_strategy_model.pkl", 
               scaler_path: str = "models/ml_strategy_scaler.pkl"):
    """
    Load a trained model and scaler.
    
    Args:
        model_path (str): Path to the trained model
        scaler_path (str): Path to the fitted scaler
    
    Returns:
        tuple: (model, scaler) or (None, None) if loading fails
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model and scaler: {e}")
        return None, None


def predict_with_model(model, scaler, features: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features (pd.DataFrame): DataFrame with the same features used during training
    
    Returns:
        np.ndarray: Predictions (0 or 1)
    """
    try:
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return np.array([])


if __name__ == "__main__":
    import argparse

    _VALID_INTERVALS = ["1d", "1h", "30m", "15m", "5m", "2m", "1m"]

    parser = argparse.ArgumentParser(
        description="Train a RandomForest ML model for a trading symbol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --symbol BTC-USD --interval 1h
  python train_model.py --symbol EURUSD=X --interval 15m --start 2025-01-01
  python train_model.py --symbol GC=F   --interval 1d  --tune
""",
    )
    parser.add_argument("--symbol",      type=str, default="BTC-USD",
                        help="Yahoo Finance ticker (default: BTC-USD)")
    parser.add_argument("--start",       type=str, default="2020-01-01",
                        help="Training start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end",         type=str, default="2025-12-31",
                        help="Training end date   YYYY-MM-DD (default: 2025-12-31)")
    parser.add_argument("--interval",    type=str, default="1d",
                        choices=_VALID_INTERVALS,
                        help="Candle size (default: 1d). Intraday limited to 60 days by Yahoo Finance.")
    parser.add_argument("--model_path",  type=str, default=None,
                        help="Override model save path (auto-named by default)")
    parser.add_argument("--scaler_path", type=str, default=None,
                        help="Override scaler save path (auto-named by default)")
    parser.add_argument("--tune",        action="store_true",
                        help="Run RandomizedSearchCV hyperparameter tuning (slower)")

    args = parser.parse_args()

    result = train_model(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        tune=args.tune,
        interval=args.interval,
    )

    if result is not None:
        _, _, accuracy = result
        logger.info(f"Training complete — accuracy: {accuracy:.4f}")
    else:
        logger.error("Training failed")