import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sqlite3
import os

from data.db_manager import DatabaseManager
from agents.portfolio_manager import PortfolioManager

class TestClearPositions(unittest.TestCase):
    def setUp(self):
        # Use a temporary test database
        self.db_path = "data/test_market_data.db"
        self.db = DatabaseManager(self.db_path)
        
    def tearDown(self):
        self.db.close_connection()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_database_clear_all_positions(self):
        # Set some initial positions
        self.db.set_position("AAPL", 10.0, 150.0, 140.0, 160.0, 1)
        self.db.set_position("TSLA", 5.0, 700.0, 650.0, 750.0, 2)
        
        # Verify they exist
        df = self.db.get_all_positions()
        self.assertEqual(len(df), 2)
        
        # Clear all
        self.db.clear_all_positions()
        
        # Verify they are gone (size=0)
        df = self.db.get_all_positions()
        self.assertTrue(df.empty)

    @patch('agents.portfolio_manager.send_telegram_message_sync')
    @patch('agents.portfolio_manager.PortfolioManager._dispatch_order')
    def test_portfolio_manager_liquidate_all(self, mock_dispatch, mock_telegram):
        # Mock DB
        mock_db = MagicMock()
        pm = PortfolioManager()
        pm._db = mock_db
        
        # Mock open positions
        df_positions = pd.DataFrame([
            {'symbol': 'AAPL', 'position_size': 10.0, 'entry_price': 150.0, 'trade_id': 1},
            {'symbol': 'TSLA', 'position_size': 5.0, 'entry_price': 700.0, 'trade_id': 2}
        ])
        mock_db.get_all_positions.return_value = df_positions
        
        # Context with market data
        context = {
            "market_data": {
                "AAPL": {"latest_close": 155.0},
                "TSLA": {"latest_close": 690.0}
            },
            "quant": {"AAPL": {}, "TSLA": {}},
            "sentiment": {"AAPL": {}, "TSLA": {}},
            "fundamentals": {"AAPL": {}, "TSLA": {}},
            "_agents": []
        }
        
        results = pm.liquidate_all_positions("paper", context)
        
        self.assertEqual(len(results), 2)
        # Verify close_trade was called
        self.assertEqual(mock_db.close_trade.call_count, 2)
        # Verify clear_position was called
        self.assertEqual(mock_db.clear_position.call_count, 2)
        # Verify dispatch_order was called (paper mode)
        self.assertEqual(mock_dispatch.call_count, 2)
        
        # Verify PnL calculation in close_trade calls
        # AAPL: (155-150)*10 = 50
        # TSLA: (690-700)*5 = -50
        pnl_calls = [call[0][1] for call in mock_db.close_trade.call_args_list]
        self.assertIn(50.0, pnl_calls)
        self.assertIn(-50.0, pnl_calls)

if __name__ == '__main__':
    unittest.main()
