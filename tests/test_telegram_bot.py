import unittest
from unittest.mock import MagicMock, patch
from telegram_bot import AlgoTelegramBot

class TestTelegramBotLogic(unittest.TestCase):
    def setUp(self):
        self.bot = AlgoTelegramBot()
        self.bot.db = MagicMock()
        self.bot.cio = MagicMock()

    def test_status_text(self):
        # Mock DB responses
        self.bot.db.get_portfolio_value.return_value = 11000.0
        self.bot.db.get_total_deployed.return_value = 2000.0
        self.bot.db.get_all_positions.return_value = MagicMock(len=lambda: 2)
        
        # We'll just check if it constructs the status without error
        update = MagicMock()
        update.callback_query = None
        update.message.reply_text = unittest.mock.AsyncMock()
        context = MagicMock()
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.bot.status_cmd(update, context))
        
        self.bot.db.get_portfolio_value.assert_called_once()
        self.bot.db.get_total_deployed.assert_called_once()
        update.message.reply_text.assert_called_once()
        args, kwargs = update.message.reply_text.call_args
        self.assertIn("Portfolio Status", args[0])
        self.assertIn("11,000.00", args[0])

    @patch('telegram_bot.ChiefInvestmentOfficer')
    def test_analyze_cmd(self, mock_cio):
        # Mock DB response for analyze
        mock_cursor = MagicMock()
        self.bot.db.conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {
            'cio_memo': 'Bullish trend detected.',
            'quant_signal': 'BUY',
            'sentiment_signal': 'BULLISH'
        }
        
        update = MagicMock()
        update.callback_query = None
        update.message.reply_text = unittest.mock.AsyncMock()
        context = MagicMock()
        context.args = ['BTC-USD']
        
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.bot.analyze_cmd(update, context))
        
        # The first call is "Analyzing...", the second is the result
        self.assertIn("Bullish trend detected.", update.message.reply_text.return_value.edit_text.call_args[0][0])

    def test_predict_with_args(self):
        """Verify that /predict BTC-USD skips category picking and shows interval menu."""
        update = MagicMock()
        update.callback_query = None
        update.message.reply_text = unittest.mock.AsyncMock()
        context = MagicMock()
        context.args = ['BTC-USD']
        
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.bot.predict_menu(update, context))
        
        # Should call interval_menu which calls reply_text with intervals
        update.message.reply_text.assert_called_once()
        args, kwargs = update.message.reply_text.call_args
        self.assertIn("Select Interval for BTC-USD", args[0])

    @patch('utils.config.write_config')
    def test_config_update_via_chat(self, mock_write_config):
        """Verify that sending a number when a setting is selected updates config."""
        update = MagicMock()
        update.message.text = "2.5"
        update.message.reply_text = unittest.mock.AsyncMock()
        
        # Mock callback query for config_menu navigation
        update.callback_query = MagicMock()
        update.callback_query.answer = unittest.mock.AsyncMock()
        update.callback_query.edit_message_text = unittest.mock.AsyncMock()
        
        context = MagicMock()
        context.user_data = {'setting_to_change': 'ATR_SL_MULT'}
        
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.bot.chat_handler(update, context))
        
        # Should call write_config with {'ATR_SL_MULT': '2.5'}
        mock_write_config.assert_called_once_with({'ATR_SL_MULT': '2.5'})
        # Should clear setting_to_change
        self.assertIsNone(context.user_data.get('setting_to_change'))
        # Should reply with success
        update.message.reply_text.assert_called()
        self.assertIn("Updated ATR_SL_MULT to 2.5", update.message.reply_text.call_args_list[0][0][0])
        # Should also show config menu (callback_query was mocked)
        update.callback_query.edit_message_text.assert_called()

if __name__ == '__main__':
    unittest.main()
