import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from utils.data_loader import get_stock_data
from utils.features import add_technical_features, prepare_training_data
from data.db_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(symbol: str = "AAPL", start_date: str = "2020-01-01", end_date: str = "2025-01-01", 
                model_path: str = "models/ml_strategy_model.pkl", 
                scaler_path: str = "models/ml_strategy_scaler.pkl",
                tune: bool = False):
    """
    Train a machine learning model for predictive trading.
    
    Args:
        symbol (str): Stock symbol to train on
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        model_path (str): Path to save the trained model
        scaler_path (str): Path to save the fitted scaler
        tune (bool): Whether to perform hyperparameter tuning
    """
    logger.info(f"Starting to train model for {symbol}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load data using the database manager and data loader
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    data = get_stock_data(symbol, start_date, end_date, provider='yfinance')
    
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
    
    parser = argparse.ArgumentParser(description='Train ML Strategy Model')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to train on')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model_path', type=str, default='models/ml_strategy_model.pkl', help='Path to save model')
    parser.add_argument('--scaler_path', type=str, default='models/ml_strategy_scaler.pkl', help='Path to save scaler')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    logger.info(f"Starting model training for {args.symbol}")
    
    # Train the model
    result = train_model(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        tune=args.tune
    )
    
    if result is not None:
        trained_model, trained_scaler, accuracy = result
        logger.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
    else:
        logger.error("Model training failed")