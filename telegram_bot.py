import logging
import asyncio
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.request import HTTPXRequest

import utils.config as config
from data.db_manager import DatabaseManager
from agents.chief_investment_officer import ChiefInvestmentOfficer
from agents.registry import discover_agents
from cli import COMMON_SYMBOLS, _AGENT_PAIR_GROUPS

SYMBOL_CATEGORIES = {
    "forex": "Forex",
    "crypto": "Crypto",
    "commodities": "Commodities",
    "indices": "Indices / ETFs",
    "stocks": "US Stocks"
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlgoTelegramBot:
    def __init__(self):
        self.db = DatabaseManager()
        self.cio = ChiefInvestmentOfficer()
        # Lazily setup Groq for chat if needed
        self._groq_client = None

    @property
    def groq_client(self):
        if self._groq_client is None and config.GROQ_API_KEY:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=config.GROQ_API_KEY)
            except ImportError:
                logger.error("Groq library not installed.")
        return self._groq_client

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data='status'),
                InlineKeyboardButton("💰 PnL", callback_data='pnl'),
            ],
            [
                InlineKeyboardButton("📂 Positions", callback_data='positions'),
                InlineKeyboardButton("🔍 Analyze BTC", callback_data='analyze_BTC-USD'),
            ],
            [
                InlineKeyboardButton("📈 Train", callback_data='menu_train'),
                InlineKeyboardButton("🔮 Predict", callback_data='menu_predict'),
            ],
            [
                InlineKeyboardButton("📉 Backtest", callback_data='menu_backtest'),
                InlineKeyboardButton("💵 Paper", callback_data='menu_paper'),
            ],
            [
                InlineKeyboardButton("⚙️ Config", callback_data='config_menu'),
                InlineKeyboardButton("❓ Help", callback_data='help'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        target = update.effective_message or update.effective_chat
        send_func = target.reply_text if hasattr(target, 'reply_text') else target.send_message
        await send_func(
            "👋 <b>Welcome to the AI Hedge Fund Bot!</b>\n\n"
            "I'm monitoring the markets and managing your portfolio. "
            "Use the buttons below or chat with me directly about your investments.",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

    async def category_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str):
        """Show categories for symbol selection."""
        query = update.callback_query
        if query: await query.answer()

        keyboard = []
        # Group categories: Forex, Crypto, Stocks, Commodities, etc.
        cat_items = list(SYMBOL_CATEGORIES.items())
        for i in range(0, len(cat_items), 2):
            row = [InlineKeyboardButton(cat_items[i][1], callback_data=f"cat_{action}_{cat_items[i][0]}")]
            if i + 1 < len(cat_items):
                row.append(InlineKeyboardButton(cat_items[i+1][1], callback_data=f"cat_{action}_{cat_items[i+1][0]}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("➕ Custom Ticker", callback_data=f"cat_{action}_custom")])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data='start')])

        text = f"🔍 <b>Select Category for {action.capitalize()}:</b>"
        await (query.edit_message_text if query else update.message.reply_text)(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML'
        )

    async def symbol_picker(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, category: str, page: int = 0):
        """Show symbols in a category."""
        query = update.callback_query
        await query.answer()

        # Filter symbols based on category search in _AGENT_PAIR_GROUPS labels
        target_label = SYMBOL_CATEGORIES.get(category, "")
        found_pairs = []
        for label, pairs in _AGENT_PAIR_GROUPS:
            if target_label.lower() in label.lower():
                found_pairs = pairs
                break
        
        if not found_pairs:
            # Fallback for stocks/crypto if labels don't match exactly
            if category == "stocks":
                label_search = "US Tech"
            elif category == "crypto":
                label_search = "Crypto"
            else:
                label_search = target_label
            
            for label, pairs in _AGENT_PAIR_GROUPS:
                if label_search.lower() in label.lower():
                    found_pairs.extend(pairs)

        keyboard = []
        per_page = 10
        start_idx = page * per_page
        end_idx = start_idx + per_page
        page_pairs = found_pairs[start_idx:end_idx]

        for i in range(0, len(page_pairs), 2):
            row = [InlineKeyboardButton(page_pairs[i][0], callback_data=f"sel_{action}_{page_pairs[i][1]}")]
            if i + 1 < len(page_pairs):
                row.append(InlineKeyboardButton(page_pairs[i+1][0], callback_data=f"sel_{action}_{page_pairs[i+1][1]}"))
            keyboard.append(row)

        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"cat_{action}_{category}_{page-1}"))
        if end_idx < len(found_pairs):
            nav_row.append(InlineKeyboardButton("Next ➡️", callback_data=f"cat_{action}_{category}_{page+1}"))
        if nav_row: keyboard.append(nav_row)
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Categories", callback_data=f"menu_{action}")])

        await query.edit_message_text(f"� <b>{target_label} Symbols:</b>", 
                                      reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def custom_ticker_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str):
        """Prompt for manual ticker entry."""
        query = update.callback_query
        await query.answer()
        context.user_data['custom_ticker_action'] = action
        await query.edit_message_text("⌨️ <b>Enter Custom Ticker:</b>\n<i>(e.g., AAPL, NVDA, ETH-USD)</i>", parse_mode='HTML')

    async def train_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.category_menu(update, context, "train")

    async def predict_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Allow command argument: /predict BTC-USD
        if context.args:
            symbol = context.args[0].upper()
            await self.interval_menu(update, context, "predict", symbol)
            return
        await self.category_menu(update, context, "predict")

    async def backtest_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.category_menu(update, context, "backtest")

    async def paper_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.category_menu(update, context, "paper")

    async def interval_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, symbol: str):
        """Show interval options for an action."""
        query = update.callback_query
        if query: await query.answer()

        intervals = [("1d (Daily)", "1d"), ("1h (Hourly)", "1h"), ("15m (15-min)", "15m"), ("5m (5-min)", "5m")]
        keyboard = []
        for display, code in intervals:
            keyboard.append([InlineKeyboardButton(display, callback_data=f"run_{action}_{symbol}_{code}")])
        
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data=f"menu_{action}")])
        text = f"⏱ <b>Select Interval for {symbol}:</b>"
        await (query.edit_message_text if query else update.message.reply_text)(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML'
        )

    async def run_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run training in a thread."""
        query = update.callback_query
        target = query.message if query else update.message
        msg = await target.reply_text(f"🚜 Training model for {symbol} ({interval})...")
        
        try:
            from train_model import train_model
            # Run blocking task in thread
            _, _, acc = await asyncio.to_thread(train_model, symbol=symbol, interval=interval)
            await msg.edit_text(f"✅ <b>Training Complete!</b>\nSymbol: {symbol}\nInterval: {interval}\nAccuracy: {acc:.2f}", parse_mode='HTML')
        except Exception as e:
            await msg.edit_text(f"❌ Training failed: {e}")

    async def run_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run prediction in a thread."""
        query = update.callback_query
        target = query.message if query else update.message
        msg = await target.reply_text(f"🔮 Fetching prediction for {symbol} ({interval})...")
        
        try:
            from predict import predict_next_movement
            # Fix canonical path generation
            safe_name = symbol.lower().replace('/', '_').replace('=', '_')
            tf_suffix = f"_{interval}" if interval != "1d" else ""
            m_path = f"models/{safe_name}{tf_suffix}_model.pkl"
            s_path = f"models/{safe_name}{tf_suffix}_scaler.pkl"
            
            res = await asyncio.to_thread(predict_next_movement, symbol=symbol, interval=interval, model_path=m_path, scaler_path=s_path)
            if res:
                emoji = "🚀" if res['prediction'] == "UP" else "📉"
                text = (
                    f"🔮 <b>{symbol} Prediction</b>\n"
                    f"Direction: {emoji} {res['prediction']}\n"
                    f"Confidence: {res['confidence']:.1%}\n"
                    f"Price: ${res['current_price']:.2f}\n"
                    f"SL: {res['sl']:.2f} | TP: {res['tp']:.2f}"
                )
                await msg.edit_text(text, parse_mode='HTML')
            else:
                await msg.edit_text(f"❌ Could not generate prediction for {symbol}. Is model trained? Checked: {m_path}")
        except Exception as e:
            await msg.edit_text(f"❌ Prediction failed: {e}")

    async def run_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE, strategy: str, symbol: str):
        """Run backtest in a thread."""
        query = update.callback_query
        msg = await query.message.reply_text(f"📉 Running backtest: <b>{strategy}</b> on {symbol}...", parse_mode='HTML')
        
        try:
            # We use a subprocess for main.py to keep legacy logic isolated
            import os, sys, subprocess
            env = os.environ.copy()
            env.update({"STRATEGY": strategy, "MODE": "backtest", "TRADING_SYMBOL": symbol})
            
            # This is long-running, so we definitely do it in a thread
            def _runner():
                return subprocess.run([sys.executable, "main.py"], env=env, capture_output=True, text=True)
            
            cp = await asyncio.to_thread(_runner)
            # Find the final line which usually contains the result
            output = cp.stdout
            summary = "Backtest finished. Check CLI for detailed results."
            if "Final Portfolio Value" in output:
                for line in output.split("\n"):
                    if "Final Portfolio Value" in line:
                        summary = line.strip()
            
            await msg.edit_text(f"✅ <b>Backtest Complete: {strategy}</b>\n{summary}", parse_mode='HTML')
        except Exception as e:
            await msg.edit_text(f"❌ Backtest failed: {e}")

    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current portfolio status."""
        query = update.callback_query
        if query:
            # Check if this came from a callback that needs starting
            if query.data == 'status':
                await query.answer()

        val = self.db.get_portfolio_value() or config.AGENT_INITIAL_CAPITAL
        total_pnl = val - config.AGENT_INITIAL_CAPITAL
        pnl_pct = (total_pnl / config.AGENT_INITIAL_CAPITAL) * 100
        deployed = self.db.get_total_deployed()
        
        # Get active positions count
        positions = self.db.get_all_positions()
        pos_count = len(positions)

        text = (
            "🏦 <b>Portfolio Status</b>\n"
            f"Value: ${val:,.2f}\n"
            f"Total PnL: {total_pnl:+.2f} ({pnl_pct:+.2f}%)\n"
            f"Deployed: ${deployed:,.2f}\n"
            f"Active Positions: {pos_count}\n"
            f"Mode: {config.AGENT_MODE.upper()}\n"
            f"Cycle: {config.AGENT_INTERVAL_MINUTES}min"
        )
        
        if query:
            # Try to edit, if fails (e.g. content same), just reply
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
            except:
                target = update.effective_message or update.effective_chat
                await target.send_message(text, parse_mode='HTML') if hasattr(target, 'send_message') else await target.reply_text(text, parse_mode='HTML')
        else:
            await update.message.reply_text(text, parse_mode='HTML')

    async def pnl_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get PnL summary."""
        query = update.callback_query
        if query:
            await query.answer()

        trades = self.db.get_all_trades(limit=50)
        if trades.empty:
            text = "No closed trades yet."
        else:
            total_pnl = trades['pnl'].sum()
            wins = len(trades[trades['pnl'] > 0])
            losses = len(trades[trades['pnl'] < 0])
            win_rate = (wins / (wins + losses)) * 100 if (wins+losses) > 0 else 0

            text = (
                "💰 <b>PnL Report (Last 50 Trades)</b>\n"
                f"Net PnL: ${total_pnl:+.4f}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Wins: {wins} | Losses: {losses}"
            )

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
            except:
                target = update.effective_message or update.effective_chat
                if hasattr(target, 'reply_text'):
                    await target.reply_text(text, parse_mode='HTML')
                else:
                    await target.send_message(text, parse_mode='HTML')
        else:
            await update.message.reply_text(text, parse_mode='HTML')

    async def positions_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List open positions."""
        query = update.callback_query
        if query:
            await query.answer()

        df = self.db.get_all_positions()
        if df.empty:
            text = "No open positions."
        else:
            rows = []
            for _, r in df.iterrows():
                rows.append(f"• <b>{r['symbol']}</b>: Size {r['position_size']:.4f} @ ${r['entry_price']:.2f}")
            text = "📂 <b>Active Positions</b>\n\n" + "\n".join(rows)

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
            except:
                target = update.effective_message or update.effective_chat
                if hasattr(target, 'reply_text'):
                    await target.reply_text(text, parse_mode='HTML')
                else:
                    await target.send_message(text, parse_mode='HTML')
        else:
            await update.message.reply_text(text, parse_mode='HTML')

    async def analyze_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger manual analysis for a symbol."""
        query = update.callback_query
        symbol = None
        
        if query:
            await query.answer()
            if query.data.startswith('analyze_'):
                symbol = query.data.split('_')[1]
        elif context.args:
            symbol = context.args[0].upper()

        if not symbol:
            await update.message.reply_text("Please specify a symbol, e.g., /analyze BTC-USD")
            return

        msg_ref = await (query.message.reply_text if query else update.message.reply_text)(f"🔍 Analyzing {symbol}...")

        try:
            # We need a context for analysts to run. 
            # For a quick analyze, we'll run a mini cycle.
            # However, CIO expects market_data, quant, sentiment in context.
            # A full run_cycle is overkill, but we can call the analysts directly if needed.
            # For now, let's use the DB's latest trade log as a "status" check of the last cycle.
            
            trades = self.db.get_all_trades(limit=1) # Get mostly recent record
            # actually better to just run the CIO deliberation if we have some context
            # but we don't have live data here without running other agents.
            
            # Simplified: Use the latest CIO memo from the trades log
            query_db = "SELECT cio_memo, quant_signal, sentiment_signal FROM agent_trades WHERE symbol=? ORDER BY timestamp DESC LIMIT 1"
            cursor = self.db.conn.cursor()
            cursor.execute(query_db, (symbol,))
            row = cursor.fetchone()
            
            if row:
                text = (
                    f"🔍 <b>Analysis: {symbol}</b>\n"
                    f"Quant: {row['quant_signal']}\n"
                    f"Sentiment: {row['sentiment_signal']}\n\n"
                    f"📝 <b>CIO Memo:</b>\n{row['cio_memo']}"
                )
            else:
                text = f"No recent analysis found for {symbol}."
            
            await msg_ref.edit_text(text, parse_mode='HTML')

        except Exception as e:
            await msg_ref.edit_text(f"❌ Error during analysis: {e}")

    async def retrain_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger manual retraining."""
        if not context.args:
            await update.message.reply_text("Usage: /retrain <symbol>")
            return
        
        symbol = context.args[0].upper()
        msg = await update.message.reply_text(f"🚜 Triggering retrain for {symbol}...")
        
        try:
            from run_agent import trigger_retrain
            # This is sync, we might want to run in thread but for simple bot it's okay
            trigger_retrain(symbol)
            await msg.edit_text(f"✅ Retrain cycle initiated for {symbol}. Check logs for progress.")
        except Exception as e:
            await msg.edit_text(f"❌ Failed to trigger retrain: {e}")

    async def settings_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot settings."""
        query = update.callback_query
        if query:
            await query.answer()

        text = (
            "⚙️ <b>Current Settings</b>\n"
            f"Symbols: {', '.join(config.AGENT_SYMBOLS)}\n"
            f"Interval: {config.AGENT_INTERVAL_MINUTES}m\n"
            f"Initial Capital: ${config.AGENT_INITIAL_CAPITAL}\n"
            f"Mode: {config.AGENT_MODE}"
        )
        keyboard = [[InlineKeyboardButton("✏️ Edit Config", callback_data='config_menu')],
                    [InlineKeyboardButton("⬅️ Back", callback_data='start')]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                await query.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)


    async def chat_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-command text."""
        user_text = update.message.text
        
        # ── Check IF user is currently updating a config setting ─────────────────────
        setting_key = context.user_data.get('setting_to_change')
        if setting_key:
            try:
                # Basic validation: must be a number
                val = float(user_text)
                from utils.config import write_config
                write_config({setting_key: str(val)})
                
                context.user_data['setting_to_change'] = None
                await update.message.reply_text(f"✅ <b>Updated {setting_key} to {val}</b>\n(Note: Restart the bot or agent loop to apply some deep changes).", parse_mode='HTML')
                await self.config_menu(update, context)
                return
            except ValueError:
                await update.message.reply_text("❌ Invalid value. Please enter a number (e.g. 1.5). Type /cancel to stop.")
                return

        # ── Check IF user is providing a custom ticker ─────────────────────
        custom_action = context.user_data.get('custom_ticker_action')
        if custom_action:
            symbol = user_text.strip().upper()
            context.user_data['custom_ticker_action'] = None
            if custom_action in ('train', 'predict', 'tp'):
                await self.interval_menu(update, context, custom_action, symbol)
            elif custom_action == 'backtest':
                await self.run_backtest(update, context, "ml_predictive", symbol)
            elif custom_action == 'paper':
                await self.run_paper(update, context, symbol)
            return

        # ── Normal AI Chat ──────────────────────────────────────────────────
        client = self.groq_client
        if not client:
            await update.message.reply_text("Groq API key missing or library not installed. AI chat disabled.")
            return

        # Prepare context for AI
        val = self.db.get_portfolio_value() or config.AGENT_INITIAL_CAPITAL
        positions = self.db.get_all_positions()
        pos_summary = positions[['symbol', 'position_size', 'entry_price']].to_string() if not positions.empty else "No open positions."

        system_prompt = (
            "You are a helpful AI Hedge Fund Assistant. You monitor an algorithmic trading bot. "
            f"Current Portfolio Value: ${val:,.2f}. "
            f"Current Positions:\n{pos_summary}\n"
            "Answer the user's questions about the bot, trading strategies, or portfolio performance. "
            "Be professional, concise, and logical."
        )

        try:
            from groq import Groq
            chat = await asyncio.to_thread(
                client.chat.completions.create,
                model=config.GROQ_MODEL_8B,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=300
            )
            response = chat.choices[0].message.content.strip()
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            await update.message.reply_text("Sorry, I'm having trouble thinking right now.")
    async def train_predict_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show options for training then predicting."""
        query = update.callback_query
        await query.answer()
        keyboard = [
            [InlineKeyboardButton("BTC-USD (1h)", callback_data='tp_run_BTC-USD_1h')],
            [InlineKeyboardButton("Gold (1d)", callback_data='tp_run_GC=F_1d')],
            [InlineKeyboardButton("⬅️ Back", callback_data='start')]
        ]
        await query.edit_message_text("📈🔮 <b>Train → Predict Chain:</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def run_train_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run train then predict."""
        await self.run_train(update, context, symbol, interval)
        await self.run_predict(update, context, symbol, interval)

    async def config_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show configuration menu."""
        query = update.callback_query
        if query: await query.answer()
        
        keyboard = [
            [InlineKeyboardButton(f"Stop Loss: {config.ATR_SL_MULT}x", callback_data='conf_set_ATR_SL_MULT')],
            [InlineKeyboardButton(f"Take Profit: {config.ATR_TP_MULT}x", callback_data='conf_set_ATR_TP_MULT')],
            [InlineKeyboardButton(f"Risk/Trade: {config.RISK_PER_TRADE_PCT*100}%", callback_data='conf_set_RISK_PER_TRADE_PCT')],
            [InlineKeyboardButton(f"Vote Score: {config.REQUIRED_VOTE_SCORE}", callback_data='conf_set_REQUIRED_VOTE_SCORE')],
            [InlineKeyboardButton("⬅️ Back", callback_data='start')]
        ]
        text = "⚙️ <b>Interactive Configuration</b>\nSelect a setting to modify it:"
        if query:
            try:
                await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
            except Exception as e:
                logger.error(f"Error editing config menu: {e}")
                target = update.effective_message or update.effective_chat
                if hasattr(target, 'reply_text'):
                    await target.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
                else:
                    await target.send_message(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
        else:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def config_set_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE, key: str):
        """Prompt user for a new value."""
        query = update.callback_query
        await query.answer()
        context.user_data['setting_to_change'] = key
        await query.edit_message_text(f"📝 <b>Updating {key}</b>\nPlease type the new value (e.g., 2.5):", parse_mode='HTML')

    async def dashboard_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send dashboard link."""
        await update.message.reply_text("📊 <b>AI Hedge Fund Dashboard</b>\nLink: <a href='http://localhost:8501'>Open Dashboard</a>\n<i>Note: Dashboard must be running on your host machine.</i>", parse_mode='HTML')

    async def list_symbols_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all supported symbols grouped by category."""
        lines = ["📋 <b>Supported Symbols:</b>\n"]
        for label, pairs in _AGENT_PAIR_GROUPS:
            lines.append(f"<b>{label}</b>")
            for display, ticker in pairs:
                lines.append(f"• {display}")
            lines.append("")
        
        await update.message.reply_text("\n".join(lines), parse_mode='HTML')

    async def run_paper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, sym: str):
        """Launch legacy paper trading."""
        query = update.callback_query
        target = query.message if query else update.message
        await target.reply_text(f"💵 Launching paper trading for {sym} in background... check logs.")
        import subprocess, sys, os
        env = os.environ.copy()
        env.update({"MODE": "paper", "TRADING_SYMBOL": sym})
        subprocess.Popen([sys.executable, "main.py"], env=env)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses."""
        query = update.callback_query
        data = query.data

        if data == 'start': await self.start(update, context)
        elif data == 'status': await self.status_cmd(update, context)
        elif data == 'pnl': await self.pnl_cmd(update, context)
        elif data == 'positions': await self.positions_cmd(update, context)
        elif data == 'settings': await self.settings_cmd(update, context)
        elif data == 'config_menu': await self.config_menu(update, context)
        elif data == 'help': await self.help_cmd(update, context)
        elif data == 'menu_train': await self.train_menu(update, context)
        elif data == 'menu_predict': await self.predict_menu(update, context)
        elif data == 'menu_backtest': await self.backtest_menu(update, context)
        elif data == 'menu_paper': await self.paper_menu(update, context)
        elif data == 'menu_tp': await self.train_predict_menu(update, context)
        elif data.startswith('cat_'):
            # Handling: cat_action_category[_page]
            parts = data.split('_')
            action = parts[1]
            category = parts[2]
            page = int(parts[3]) if len(parts) > 3 else 0
            if category == 'custom':
                await self.custom_ticker_prompt(update, context, action)
            else:
                await self.symbol_picker(update, context, action, category, page)
        elif data.startswith('sel_'):
            # Handling: sel_action_symbol
            _, action, symbol = data.split('_', 2)
            if action in ('train', 'predict', 'tp'):
                await self.interval_menu(update, context, action, symbol)
            elif action == 'backtest':
                # For backtest, we might need strategy first. Let's simplify and use ML Basic
                await self.run_backtest(update, context, "ml_predictive", symbol)
            elif action == 'paper':
                await self.run_paper(update, context, symbol)
        elif data.startswith('run_'):
            # Handling: run_action_symbol_interval
            _, action, symbol, interval = data.split('_', 3)
            if action == 'train': await self.run_train(update, context, symbol, interval)
            elif action == 'predict': await self.run_predict(update, context, symbol, interval)
            elif action == 'tp': await self.run_train_predict(update, context, symbol, interval)
        elif data.startswith('analyze_'): await self.analyze_cmd(update, context)
        elif data.startswith('conf_set_'):
            key = data.replace('conf_set_', '')
            await self.config_set_prompt(update, context, key)

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display help message."""
        query = update.callback_query
        if query:
            await query.answer()

        text = (
            "🤖 <b>AI Hedge Fund Bot - Command List</b>\n\n"
            "<b>General:</b>\n"
            "/start - Main menu with action buttons\n"
            "/settings - View bot mode, symbols, and interval\n"
            "/help - Show this list of commands\n"
            "/dashboard - Linked to the Streamlit UI\n\n"
            "<b>Account & Portfolio:</b>\n"
            "/status - Total value, PnL, and summary\n"
            "/pnl - Detailed report of last 50 trades\n"
            "/positions - List all currently active trades\n\n"
            "<b>Analysis & Training:</b>\n"
            "/analyze &lt;symbol&gt; - Get latest CIO memo\n"
            "/train - Interactive model training menu\n"
            "/predict - Next-candle price prediction\n"
            "/tp - Train &amp; Predict chain\n"
            "/retrain &lt;symbol&gt; - Force immediate model retraining\n"
            "/list_symbols - View symbols by category\n\n"
            "<b>Configuration:</b>\n"
            "/config - Edit trade risk and reward settings\n\n"
            "<i>Type any command or simply chat with the AI Fund Manager.</i>"
        )
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='start')]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Error editing message for help: {e}")
                target = update.effective_message or update.effective_chat
                if hasattr(target, 'reply_text'):
                    await target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)
                else:
                    await target.send_message(text, parse_mode='HTML', reply_markup=reply_markup)
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

def main():
    if not config.TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not found in .env. Exiting.")
        return

    bot = AlgoTelegramBot()
    
    # Increase timeouts for flaky networks
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).request(request).build()

    # Commands
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("status", bot.status_cmd))
    app.add_handler(CommandHandler("pnl", bot.pnl_cmd))
    app.add_handler(CommandHandler("positions", bot.positions_cmd))
    app.add_handler(CommandHandler("analyze", bot.analyze_cmd))
    app.add_handler(CommandHandler("retrain", bot.retrain_cmd))
    app.add_handler(CommandHandler("settings", bot.settings_cmd))
    app.add_handler(CommandHandler("train", bot.train_menu))
    app.add_handler(CommandHandler("predict", bot.predict_menu))
    app.add_handler(CommandHandler("backtest", bot.backtest_menu))
    app.add_handler(CommandHandler("paper", bot.paper_menu))
    app.add_handler(CommandHandler("config", bot.config_menu))
    app.add_handler(CommandHandler("tp", bot.train_predict_menu))
    app.add_handler(CommandHandler("dashboard", bot.dashboard_cmd))
    app.add_handler(CommandHandler("list_symbols", bot.list_symbols_cmd))
    app.add_handler(CommandHandler("help", bot.help_cmd))

    # Button Callbacks
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    # Non-command messages (Chat / Custom Input)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.chat_handler))

    print("🤖 Telegram Bot is running...")
    # Add bootstrap retries to handle initial connection issues
    app.run_polling(bootstrap_retries=5)

if __name__ == "__main__":
    main()
