import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os
from utils.features import add_technical_features
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLPredictiveStrategy(bt.Strategy):
    """
    A machine learning-based predictive trading strategy.
    Uses a trained model to predict future price movements (Buy/Sell/Hold).
    """
    params = dict(
        model_path="models/ml_strategy_model.pkl",
        scaler_path="models/ml_strategy_scaler.pkl",
        printlog=False
    )

    def __init__(self):
        # Keep track of closing price
        self.data_close = self.datas[0].close
        
        # Load the trained model and scaler
        try:
            self.model = joblib.load(self.params.model_path)
            self.model.n_jobs = 1  # Set to 1 for faster single-row inference
            self.scaler = joblib.load(self.params.scaler_path)
            logger.info("ML model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
            raise

        # Initialize order tracking
        self.order = None
        self.buy_price = None
        self.buy_date = None

        # Track if we have enough data for feature engineering
        self.has_enough_data = False

    def log(self, txt, dt=None, doprint=False):
        """Logging function"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Build a DataFrame with the current data to calculate features
        # For this, we need at least enough historical data for feature calculation
        # We'll check if we have enough data points
        if len(self) < 50:  # Need at least 50 data points to have meaningful features
            return
        
        # Create a DataFrame with the most recent data to calculate features
        # We'll build it with enough history to calculate all features properly
        df = pd.DataFrame({
            'open': [self.datas[0].open[i] for i in range(-49, 1)],  # 50 data points
            'high': [self.datas[0].high[i] for i in range(-49, 1)],
            'low': [self.datas[0].low[i] for i in range(-49, 1)],
            'close': [self.datas[0].close[i] for i in range(-49, 1)],
            'volume': [self.datas[0].volume[i] for i in range(-49, 1)]
        })
        
        # Convert index to a temporary datetime-like format
        df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
        
        # Calculate features using the features module
        df_with_features = add_technical_features(df)
        
        # Check if there are valid features (not all NaN)
        if df_with_features.empty or df_with_features.iloc[-1:].empty:
            return
            
        # Take the last row (current bar) with features
        current_features = df_with_features.iloc[-1:].dropna(axis=1, how='all')
        
        # Ensure we have all the features the model expects
        # For simplicity, we'll take all numeric columns that aren't the target
        feature_cols = [col for col in current_features.columns 
                       if col not in ['target', 'future_close'] and pd.api.types.is_numeric_dtype(current_features[col])]
        
        if len(feature_cols) == 0:
            return
            
        current_features_filtered = current_features[feature_cols]
        
        # Check if we have the right number of features
        if len(current_features_filtered.columns) != self.scaler.n_features_in_:
            logger.warning(f"Feature mismatch: expected {self.scaler.n_features_in_}, got {len(current_features_filtered.columns)}")
            return

        # Make prediction
        try:
            # Pass DataFrame to scaler to avoid "X does not have valid feature names" warning
            # Reshape is not needed if we pass a DataFrame with correct columns
            features_scaled = self.scaler.transform(current_features_filtered)
            
            # Make prediction (0: Sell/Short, 1: Buy/Long)
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Get the confidence of the prediction (max probability)
            confidence = max(prediction_proba)
            
            # Check if we are in the market
            if not self.position:
                # Not yet ... we MIGHT BUY if ...
                if prediction == 1 and confidence > 0.6:  # Buy signal with confidence threshold
                    self.log(f'BUY CREATE, Close: {self.data_close[0]:.2f}, Confidence: {confidence:.2f}')
                    self.buy_price = self.data_close[0]
                    self.buy_date = self.datas[0].datetime.date(0)
                    
                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

            else:
                # Already in the market ... we might sell
                if prediction == 0 and confidence > 0.6:  # Sell signal with confidence threshold
                    self.log(f'SELL CREATE, Close: {self.data_close[0]:.2f}, Confidence: {confidence:.2f}')
                    
                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.sell()

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                if self.buy_price:
                    pnl = order.executed.price - self.buy_price
                    self.log(f'P&L: {pnl:.2f} ({pnl/self.buy_price*100:.2f}%)')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset the order tracking
        self.order = None

    def stop(self):
        self.log(f'Ending Value: {self.broker.getvalue():.2f}', doprint=True)
        if self.buy_price:
            self.log(f'Last Buy Price: {self.buy_price:.2f}', doprint=True)


# Risk-managed version of the strategy
class MLPredictiveStrategyWithRiskManagement(MLPredictiveStrategy):
    """
    ML strategy with risk management (stop loss and take profit)
    """
    params = dict(
        model_path="models/ml_strategy_model.pkl",
        scaler_path="models/ml_strategy_scaler.pkl",
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.04,  # 4% take profit
        atr_period=14,
        printlog=False
    )

    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Keep track of ATR for risk management
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        
        # Track position for risk management
        self.position_risk_managed = False
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def next(self):
        # Check if an order is pending
        if self.order:
            return

        # Check if we have an open position and need to manage risk
        if self.position and self.position_risk_managed:
            current_price = self.data_close[0]
            
            # Check stop loss
            if self.stop_price and ((self.position.size > 0 and current_price <= self.stop_price) or 
                                   (self.position.size < 0 and current_price >= self.stop_price)):
                self.log(f'STOP LOSS triggered at {current_price:.2f}')
                self.order = self.close()
                self.position_risk_managed = False
                return
            
            # Check take profit
            if self.target_price and ((self.position.size > 0 and current_price >= self.target_price) or 
                                     (self.position.size < 0 and current_price <= self.target_price)):
                self.log(f'TAKE PROFIT triggered at {current_price:.2f}')
                self.order = self.close()
                self.position_risk_managed = False
                return

        # Call parent's next method for signal generation
        # Build a DataFrame with the current data to calculate features
        if len(self) < 50:  # Need at least 50 data points to have meaningful features
            return
        
        # Create a DataFrame with the most recent data to calculate features
        df = pd.DataFrame({
            'open': [self.datas[0].open[i] for i in range(-49, 1)],  # 50 data points
            'high': [self.datas[0].high[i] for i in range(-49, 1)],
            'low': [self.datas[0].low[i] for i in range(-49, 1)],
            'close': [self.datas[0].close[i] for i in range(-49, 1)],
            'volume': [self.datas[0].volume[i] for i in range(-49, 1)]
        })
        
        # Convert index to a temporary datetime-like format
        df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
        
        # Calculate features using the features module
        df_with_features = add_technical_features(df)
        
        # Check if there are valid features
        if df_with_features.empty or df_with_features.iloc[-1:].empty:
            return
            
        # Take the last row (current bar) with features
        current_features = df_with_features.iloc[-1:].dropna(axis=1, how='all')
        
        # Ensure we have all the features the model expects
        feature_cols = [col for col in current_features.columns 
                       if col not in ['target', 'future_close'] and pd.api.types.is_numeric_dtype(current_features[col])]
        
        if len(feature_cols) == 0:
            return
            
        current_features_filtered = current_features[feature_cols]
        
        # Check if we have the right number of features
        if len(current_features_filtered.columns) != self.scaler.n_features_in_:
            logger.warning(f"Feature mismatch: expected {self.scaler.n_features_in_}, got {len(current_features_filtered.columns)}")
            return

        # Make prediction
        try:
            # Reshape the features to match what the model expects
            features_array = current_features_filtered.values.reshape(1, -1)
            
            # Scale the features
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction (0: Sell/Short, 1: Buy/Long)
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Get the confidence of the prediction (max probability)
            confidence = max(prediction_proba)
            
            # Check if we are in the market
            if not self.position:
                # Not yet ... we MIGHT BUY if ...
                if prediction == 1 and confidence > 0.6:  # Buy signal with confidence threshold
                    self.log(f'BUY CREATE, Close: {self.data_close[0]:.2f}, Confidence: {confidence:.2f}')
                    
                    # Record entry price for risk management
                    self.entry_price = self.data_close[0]
                    
                    # Calculate stop loss and take profit prices based on ATR
                    current_atr = self.atr[0]
                    self.stop_price = self.entry_price - (2 * current_atr)
                    self.target_price = self.entry_price + (3 * current_atr)
                    
                    self.buy_date = self.datas[0].datetime.date(0)
                    
                    # Place order
                    self.order = self.buy()

            else:
                # Already in the market ... we might sell
                if prediction == 0 and confidence > 0.6:  # Sell signal with confidence threshold
                    self.log(f'SELL CREATE, Close: {self.data_close[0]:.2f}, Confidence: {confidence:.2f}')
                    
                    # Close position and reset risk management flags
                    self.order = self.close()
                    self.position_risk_managed = False
                    self.entry_price = None
                    self.stop_price = None
                    self.target_price = None

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                # Mark that this position is under risk management
                self.position_risk_managed = True
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                if self.buy_price:
                    pnl = order.executed.price - self.buy_price
                    self.log(f'P&L: {pnl:.2f} ({pnl/self.buy_price*100:.2f}%)')
                
                # Reset risk management flags
                self.position_risk_managed = False
                self.entry_price = None
                self.stop_price = None
                self.target_price = None

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset the order tracking
        self.order = None

    def stop(self):
        self.log(f'Ending Value: {self.broker.getvalue():.2f}', doprint=True)
        if self.buy_price:
            self.log(f'Last Buy Price: {self.buy_price:.2f}', doprint=True)


if __name__ == "__main__":
    # Example usage would go here
    pass