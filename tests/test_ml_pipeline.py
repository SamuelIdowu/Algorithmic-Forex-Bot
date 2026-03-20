import unittest
import os
import shutil
import pandas as pd
import numpy as np
import backtrader as bt
from unittest.mock import patch, MagicMock
from train_model import train_model
from strategies.ml_predictive import MLPredictiveStrategy

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.model_path = "models/test_model.pkl"
        self.scaler_path = "models/test_scaler.pkl"
        
        # Create dummy data
        dates = pd.date_range(start="2023-01-01", periods=200)
        self.df = pd.DataFrame({
            'open': np.linspace(100, 200, 200) + np.random.normal(0, 2, 200),
            'high': np.linspace(105, 205, 200) + np.random.normal(0, 2, 200),
            'low': np.linspace(95, 195, 200) + np.random.normal(0, 2, 200),
            'close': np.linspace(100, 200, 200) + np.random.normal(0, 2, 200),
            'volume': np.random.randint(1000, 5000, 200)
        }, index=dates)
        self.df.index.name = 'Date'

    def tearDown(self):
        # Clean up models
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.scaler_path):
            os.remove(self.scaler_path)

    @patch('train_model.get_yfinance_data')
    def test_train_model(self, mock_get_data):
        mock_get_data.return_value = self.df
        
        # Train model
        model, scaler, accuracy = train_model(
            symbol="TEST", 
            start_date="2023-01-01", 
            end_date="2023-07-01",
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.scaler_path))

    @patch('train_model.get_yfinance_data')
    def test_strategy_execution(self, mock_get_data):
        # 1. Train the model first
        mock_get_data.return_value = self.df
        train_model(
            symbol="TEST", 
            start_date="2023-01-01", 
            end_date="2023-07-01",
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        # 2. Run Backtrader
        cerebro = bt.Cerebro()
        
        # Add data
        data = bt.feeds.PandasData(dataname=self.df)
        cerebro.adddata(data)
        
        # Add strategy
        cerebro.addstrategy(
            MLPredictiveStrategy, 
            model_path=self.model_path,
            scaler_path=self.scaler_path
        )
        
        # Run
        try:
            cerebro.run()
        except Exception as e:
            self.fail(f"Strategy execution failed: {e}")

if __name__ == '__main__':
    unittest.main()
