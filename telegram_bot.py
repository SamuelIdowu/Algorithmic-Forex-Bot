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
from agents.sentiment_analyst import SentimentAnalyst
from agents.portfolio_manager import PortfolioManager
from agents.registry import discover_agents
from cli import COMMON_SYMBOLS, AGENT_PAIR_GROUPS

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
        self.sentiment_bot = SentimentAnalyst()
        # Lazily setup Groq for chat if needed
        self._groq_client = None
        
        # Monitoring state: { (chat_id, symbol): { 'interval': str, 'last_run': datetime } }
        self.active_monitors = {}
        self._load_monitors()
        self._monitor_task = None

    def _load_monitors(self):
        """Load monitors from database."""
        db_monitors = self.db.get_active_monitors()
        for m in db_monitors:
            self.active_monitors[(m['chat_id'], m['symbol'])] = {
                'interval': m['interval'],
                'last_run': m['last_run']
            }
        if self.active_monitors:
            logger.info(f"Loaded {len(self.active_monitors)} monitors from database.")

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
                InlineKeyboardButton("📂 Positions", callback_data='positions'),
            ],
            [
                InlineKeyboardButton("� Analyze", callback_data='menu_analyze'),
                InlineKeyboardButton("🔮 Predict", callback_data='menu_predict'),
                InlineKeyboardButton("📋 Trackers", callback_data='menu_monitors'),
            ],
            [
                InlineKeyboardButton("📈 Train", callback_data='menu_train'),
                InlineKeyboardButton("📉 Backtest", callback_data='menu_backtest'),
                InlineKeyboardButton("� Monitor", callback_data='menu_monitor'),
            ],
            [
                InlineKeyboardButton("� Paper", callback_data='menu_paper'),
                InlineKeyboardButton("⚙️ Config", callback_data='config_menu'),
                InlineKeyboardButton("❓ Help", callback_data='help'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        target = update.effective_message or update.effective_chat
        send_func = target.reply_text if hasattr(target, 'reply_text') else target.send_message
        await send_func(
            "🚀 <b>AI Hedge Fund Terminal</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "Welcome, Commander. I am your automated trading floor manager. "
            "I'm currently monitoring global markets and optimizing your capital allocation.\n\n"
            "<b>Institutional-grade AI analysis</b> is at your fingertips. Use the dashboard below to control your fund.",
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

        # Filter symbols based on category search in AGENT_PAIR_GROUPS labels
        target_label = SYMBOL_CATEGORIES.get(category, "")
        found_pairs = []
        for label, pairs in AGENT_PAIR_GROUPS:
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
            
            for label, pairs in AGENT_PAIR_GROUPS:
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

    async def monitor_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show categories for symbol selection for monitoring."""
        await self.category_menu(update, context, "monitor")

    async def interval_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, symbol: str):
        """Show interval options for an action."""
        query = update.callback_query
        if query: await query.answer()

        intervals = [("1d (Daily)", "1d"), ("1h (Hourly)", "1h"), ("15m (15-min)", "15m"), ("5m (5-min)", "5m"), ("1m (Test Mode)", "1m")]
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
            await msg.edit_text(
                f"✅ <b>Model Optimized!</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"<b>Asset:</b> <code>{symbol}</code>\n"
                f"<b>Timeframe:</b> <code>{interval}</code>\n"
                f"<b>Backtest Accuracy:</b> <code>{acc:.2f}</code>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"<i>The AI has updated its internal weights for this asset.</i>", 
                parse_mode='HTML'
            )
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
            
            res = await asyncio.to_thread(predict_next_movement, symbol=symbol, interval=interval, model_path=m_path, scaler_path=s_path, tp_mult=config.ATR_TP_MULT, sl_mult=config.ATR_SL_MULT)
            if res:
                emoji = "🚀" if res['prediction'] == "UP" else "📉"
                rr_ratio = abs(res['tp'] - res['current_price']) / abs(res['current_price'] - res['sl']) if abs(res['current_price'] - res['sl']) != 0 else 0
                
                text = (
                    f"🔮 <b>{symbol} Forecast ({interval})</b>\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Direction:</b> {emoji} <code>{res['prediction']}</code>\n"
                    f"<b>Confidence:</b> <code>{res['confidence']:.1%}</code>\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Entry Price:</b> <code>${res['current_price']:.2f}</code>\n"
                    f"<b>Target (TP):</b> <code>${res['tp']:.2f}</code>\n"
                    f"<b>Stop (SL):</b>   <code>${res['sl']:.2f}</code>\n"
                    f"<b>R/R Ratio:</b>  <code>{rr_ratio:.2f}</code>\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    f"<i>Forecast generate based on latest ML model.</i>"
                )
                await msg.edit_text(text, parse_mode='HTML')
            else:
                await msg.edit_text(f"❌ <b>Could not generate prediction for {symbol}.</b>\nIs the model trained? Checked: <code>{m_path}</code>", parse_mode='HTML')
        except Exception as e:
            await msg.edit_text(f"❌ <b>Prediction failed:</b>\n<code>{e}</code>", parse_mode='HTML')

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
            final_summary = f"✅ <b>Simulation Complete: {strategy}</b>\n━━━━━━━━━━━━━━━━━━\n"
            if "Final Portfolio Value" in output:
                vals = []
                for line in output.split("\n"):
                    if any(x in line for x in ["Final Portfolio Value", "Profit/Loss"]):
                        vals.append(f"<code>{line.strip()}</code>")
                final_summary += "\n".join(vals)
            else:
                final_summary += "Backtest finished. Check CLI for detailed logs."
            
            await msg.edit_text(final_summary, parse_mode='HTML')
        except Exception as e:
            await msg.edit_text(f"❌ Backtest failed: {e}")

    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get current portfolio status with richer metrics."""
        query = update.callback_query
        if query:
            await query.answer()

        val = self.db.get_portfolio_value() or config.AGENT_INITIAL_CAPITAL
        total_pnl = val - config.AGENT_INITIAL_CAPITAL
        pnl_pct = (total_pnl / config.AGENT_INITIAL_CAPITAL) * 100
        deployed = self.db.get_total_deployed()
        cash = val - deployed
        
        # Get active positions
        positions = self.db.get_all_positions()
        pos_count = len(positions)

        # Calculate Daily PnL (trades closed in last 24h)
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT SUM(pnl) FROM agent_trades WHERE closed_at >= datetime('now', '-1 day') AND pnl IS NOT NULL")
            daily_pnl = float(cursor.fetchone()[0] or 0.0)
        except:
            daily_pnl = 0.0

        # Create a simple progress bar for Deployed vs Cash
        bar_length = 10
        deployed_ratio = min(1.0, deployed / val) if val > 0 else 0
        filled_blocks = int(deployed_ratio * bar_length)
        bar = "▓" * filled_blocks + "░" * (bar_length - filled_blocks)

        text = (
            "🏦 <b>Portfolio Executive Summary</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Total Value:</b>  <code>${val:,.2f}</code>\n"
            f"📈 <b>Total PnL:</b>    <code>{total_pnl:+.2f} ({pnl_pct:+.2f}%)</code>\n"
            f"📅 <b>24h Change:</b>   <code>{daily_pnl:+.2f}</code>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>Available Cash:</b> <code>${cash:,.2f}</code>\n"
            f"🔧 <b>Capital Deployed:</b> <code>${deployed:,.2f}</code>\n"
            f"<code>[{bar}] {deployed_ratio:.0%}</code>\n\n"
            f"📊 <b>Active Positions:</b> <code>{pos_count}</code>\n"
            f"🤖 <b>Bot Mode:</b> <code>{config.AGENT_MODE.upper()}</code>\n"
            f"⏱ <b>Cycle:</b> <code>{config.AGENT_INTERVAL_MINUTES}min</code>"
        )
        
        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]])
        
        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                target = update.effective_message or update.effective_chat
                await (target.send_message(text, parse_mode='HTML', reply_markup=reply_markup) if hasattr(target, 'send_message') else target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup))
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def pnl_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get PnL summary with advanced metrics."""
        query = update.callback_query
        if query:
            await query.answer()

        trades = self.db.get_all_trades(limit=100)
        if trades.empty or 'pnl' not in trades.columns or trades['pnl'].isnull().all():
            text = "🟡 <b>No closed trades found yet.</b>"
        else:
            # Filter only closed trades with PnL
            closed_trades = trades[trades['pnl'].notnull()]
            total_pnl = closed_trades['pnl'].sum()
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            losing_trades = closed_trades[closed_trades['pnl'] < 0]
            
            wins = len(winning_trades)
            losses = len(losing_trades)
            count = len(closed_trades)
            win_rate = (wins / count) * 100 if count > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if wins > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if losses > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losses > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

            text = (
                "💰 <b>Performance Analytics</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"💵 <b>Net Profit:</b> <code>{total_pnl:+.4f}</code>\n"
                f"🎯 <b>Win Rate:</b>   <code>{win_rate:.1f}%</code>\n"
                f"📊 <b>Trade Count:</b> <code>{count}</code>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"✅ <b>Wins:</b>   <code>{wins}</code> (Avg: <code>{avg_win:+.4f}</code>)\n"
                f"❌ <b>Losses:</b> <code>{losses}</code> (Avg: <code>{avg_loss:+.4f}</code>)\n"
                f"📈 <b>Profit Factor:</b> <code>{profit_factor:.2f}</code>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"<i>Based on last {count} closed trades.</i>"
            )

        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]])
        
        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                target = update.effective_message or update.effective_chat
                await (target.send_message(text, parse_mode='HTML', reply_markup=reply_markup) if hasattr(target, 'send_message') else target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup))
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def positions_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List open positions with unrealized PnL and risk metrics."""
        query = update.callback_query
        if query:
            await query.answer()

        df = self.db.get_all_positions()
        if df.empty:
            text = "⚪️ <b>No active positions at the moment.</b>"
        else:
            val = self.db.get_portfolio_value() or config.AGENT_INITIAL_CAPITAL
            rows = []
            total_unrealized = 0
            
            for _, r in df.iterrows():
                symbol = r['symbol']
                # Try to get latest price for mark-to-market
                try:
                    price_df = self.db.load_data(symbol)
                    current_price = price_df.tail(1)['close'].iloc[0] if not price_df.empty else r['entry_price']
                except:
                    current_price = r['entry_price']
                
                unrealized = (current_price - r['entry_price']) * r['position_size']
                unrealized_pct = ((current_price / r['entry_price']) - 1) * 100
                total_unrealized += unrealized
                
                risk_pct = (r['cash_deployed'] / val) * 100 if val > 0 else 0
                emoji = "🟢" if unrealized >= 0 else "🔴"
                
                rows.append(
                    f"{emoji} <b>{symbol}</b>\n"
                    f"   Size: <code>{r['position_size']:.4f}</code> | Entry: <code>${r['entry_price']:.2f}</code>\n"
                    f"   Current: <code>${current_price:.2f}</code>\n"
                    f"   PnL: <code>{unrealized:+.2f} ({unrealized_pct:+.2f}%)</code>\n"
                    f"   Portfolio Risk: <code>{risk_pct:.1f}%</code>"
                )
            
            summary = (
                f"📂 <b>Active Positions ({len(df)})</b>\n"
                f"Total Unrealized: <code>{total_unrealized:+.2f}</code>\n"
                "━━━━━━━━━━━━━━━━━━\n"
            )
            text = summary + "\n\n".join(rows)

        keyboard = [
            [InlineKeyboardButton("⬅️ Back", callback_data='start')]
        ]
        if not df.empty:
            keyboard.insert(0, [InlineKeyboardButton("🗑 Clear All Positions", callback_data='confirm_clear')])
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                target = update.effective_message or update.effective_chat
                await (target.send_message(text, parse_mode='HTML', reply_markup=reply_markup) if hasattr(target, 'send_message') else target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup))
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def analyze_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str = None):
        """Trigger manual analysis for a symbol with improved visibility."""
        query = update.callback_query
        
        if not symbol:
            if query:
                await query.answer()
                if query.data.startswith('analyze_'):
                    symbol = query.data.split('_')[1]
                elif query.data.startswith('sel_analyze_'):
                    symbol = query.data.split('_')[2]
            elif context.args:
                symbol = context.args[0].upper()

        if not symbol:
            target = update.effective_message or update.effective_chat
            await target.reply_text("❓ <b>Please specify a symbol, e.g., /analyze BTC-USD</b>", parse_mode='HTML')
            return

        msg_ref = await (query.message.reply_text if query else update.message.reply_text)(f"🔍 <b>Scrutinizing {symbol}...</b>", parse_mode='HTML')

        try:
            # We can run a fresh analysis now that we have the bots
            res = await self.perform_combined_analysis(symbol, "1h")
            if res:
                await msg_ref.edit_text(res['text'], parse_mode='HTML')
            else:
                await msg_ref.edit_text(f"🟡 <b>Could not perform fresh analysis for {symbol}.</b>")
        except Exception as e:
            await msg_ref.edit_text(f"❌ <b>Error during analysis:</b>\n<code>{e}</code>", parse_mode='HTML')

    async def perform_combined_analysis(self, symbol: str, interval: str) -> dict:
        """Fetch prediction and perform CIO analysis for a symbol."""
        try:
            from predict import predict_next_movement
            # 1. Technical Prediction
            safe_name = symbol.lower().replace('/', '_').replace('=', '_')
            tf_suffix = f"_{interval}" if interval != "1d" else ""
            m_path = f"models/{safe_name}{tf_suffix}_model.pkl"
            s_path = f"models/{safe_name}{tf_suffix}_scaler.pkl"
            
            pred_res = await asyncio.to_thread(predict_next_movement, symbol, m_path, s_path, interval=interval, tp_mult=config.ATR_TP_MULT, sl_mult=config.ATR_SL_MULT)
            
            # 2. Sentiment Analysis
            sent_context = {"symbols": [symbol]}
            sent_res = await asyncio.to_thread(self.sentiment_bot.run, sent_context)
            sentiment = sent_res.get("sentiment", {}).get(symbol, {})
            
            # 3. CIO Deliberation
            cio_context = {
                "symbols": [symbol],
                "quant": {symbol: {"reasoning": pred_res.get('reasoning', "Technical signal generated.") if pred_res else "No technical data."}},
                "sentiment": {symbol: sentiment}
            }
            cio_res = await asyncio.to_thread(self.cio.run, cio_context)
            cio_data = cio_res.get("cio", {}).get(symbol, {})
            
            # 4. Format Output
            header = f"🔔 <b>MONITOR ALERT: {symbol} ({interval})</b>\n━━━━━━━━━━━━━━━━━━\n"
            footer = f"\n━━━━━━━━━━━━━━━━━━\n⏰ <i>Report generated at: {datetime.now().strftime('%H:%M:%S')}</i>"
            
            sig_emoji = {"BULLISH": "🚀", "BEARISH": "📉", "NEUTRAL": "⚖️"}.get(cio_data.get('cio_signal'), "❓")
            
            # Actionable Intelligence
            intel_text = (
                f"🧠 <b>Intelligence:</b> {sig_emoji} <code>{cio_data.get('cio_signal', 'NEUTRAL')}</code>\n"
                f"📝 <b>CIO Memo:</b>\n<blockquote>{cio_data.get('memo', 'No analysis available.')}</blockquote>"
            )

            # Contradiction Detection
            if pred_res:
                tech_dir = pred_res['prediction']
                sent_dir = sentiment.get('sentiment_signal', 'NEUTRAL')
                if (tech_dir == "UP" and sent_dir == "BEARISH") or (tech_dir == "DOWN" and sent_dir == "BULLISH"):
                    header = f"⚠️ <b>DIVERGENCE ALERT: {symbol}</b>\n━━━━━━━━━━━━━━━━━━\n"
                    intel_text = "🚩 <b>CONTRADICTION DETECTED:</b> Tech & Sentiment are misaligned.\n" + intel_text

            # Technical details
            if pred_res:
                emoji = "🚀" if pred_res['prediction'] == "UP" else "📉"
                rsi_val = pred_res.get('rsi', 50)
                rsi_desc = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                
                # Position Sizing Logic (Adapted from RiskManager)
                val = self.db.get_portfolio_value() or config.AGENT_INITIAL_CAPITAL
                atr = pred_res.get('atr', 0)
                risk_pct = config.RISK_PER_TRADE_PCT # e.g. 0.01
                sl_mult = config.ATR_SL_MULT
                
                rec_size = (val * risk_pct) / (atr * sl_mult) if atr > 0 else 0
                
                tech_text = (
                    f"🔮 <b>Forecast:</b> {emoji} <code>{pred_res['prediction']}</code> ({pred_res['confidence']:.1%})\n"
                    f"📊 <b>Technicals:</b> RSI: <code>{rsi_val:.1f}</code> ({rsi_desc}) | ATR: <code>{atr:.2f}</code>\n"
                    f"📈 <b>Levels:</b> Entry: <code>${pred_res['current_price']:.2f}</code> | TP: <code>${pred_res['tp']:.2f}</code>\n"
                    f"💰 <b>Rec. Size:</b> <code>{rec_size:.4f}</code> (1% Risk)\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                )
                
                # News Highlight
                news_text = ""
                news_list = self.db.get_news(symbol, limit=2)
                if news_list:
                    news_text = "📰 <b>Recent Headlines:</b>\n"
                    for n in news_list:
                        news_text += f"• <i>{n['title'][:60]}...</i>\n"
                    news_text += "━━━━━━━━━━━━━━━━━━\n"

                return {
                    'text': header + tech_text + news_text + intel_text + footer, 
                    'signal': pred_res['prediction'], 
                    'confidence': pred_res['confidence']
                }
            else:
                tech_text = "🔮 <b>Forecast:</b> ⚠️ <code>TECHNICAL DATA UNAVAILABLE</code>\n"
                tech_text += "<i>Possible causes: Insufficient liquidity or API rate limit.</i>\n━━━━━━━━━━━━━━━━━━\n"
                return {
                    'text': header + tech_text + intel_text + footer,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0
                }
        except Exception as e:
            logger.error(f"Error in perform_combined_analysis for {symbol}: {e}")
            return None

    async def _monitor_loop(self, bot_app: Application):
        """Background loop to process monitors."""
        logger.info("📡 Monitoring background loop STARTED.")
        while True:
            try:
                now = datetime.now()
                # Use a copy of keys to safely iterate
                monitor_keys = list(self.active_monitors.keys())
                
                if not monitor_keys:
                    await asyncio.sleep(30)
                    continue

                for key in monitor_keys:
                    chat_id, symbol = key
                    config_data = self.active_monitors.get(key)
                    if not config_data: continue
                    
                    interval_str = config_data['interval']
                    last_run = config_data.get('last_run')
                    
                    # Convert interval string to minutes
                    minutes = 60 # Default 1h
                    if interval_str == "1m": minutes = 1
                    elif interval_str == "5m": minutes = 5
                    elif interval_str == "15m": minutes = 15
                    elif interval_str == "30m": minutes = 30
                    elif interval_str == "1h": minutes = 60
                    elif interval_str == "4h": minutes = 240
                    elif interval_str == "1d": minutes = 1440
                    
                    should_run = False
                    if last_run is None:
                        should_run = True
                    else:
                        diff = (now - last_run).total_seconds() / 60
                        if diff >= minutes:
                            should_run = True
                    
                    if should_run:
                        logger.info(f"🔄 Running monitor task: {symbol} ({interval_str}) for chat {chat_id}")
                        # Update last_run BEFORE starting to prevent overlaps if analysis is slow
                        self.active_monitors[key]['last_run'] = now
                        self.db.save_monitor(chat_id, symbol, interval_str, now)
                        
                        try:
                            # Run analysis with timeout to prevent loop hangs
                            res = await asyncio.wait_for(self.perform_combined_analysis(symbol, interval_str), timeout=120)
                            if res:
                                await bot_app.bot.send_message(chat_id=chat_id, text=res['text'], parse_mode='HTML')
                                logger.info(f"✅ Monitor alert sent: {symbol} to {chat_id}")
                        except asyncio.TimeoutError:
                            logger.error(f"⌛ Monitor timeout for {symbol}")
                        except Exception as e:
                            logger.error(f"❌ Error processing monitor for {symbol}: {e}")
                
                # Global check interval
                await asyncio.sleep(15)
                    
            except asyncio.CancelledError:
                logger.info("Monitoring loop STOPPED.")
                break
            except Exception as e:
                logger.error(f"🔥 Critical error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(60)

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
            if custom_action in ('train', 'predict', 'tp', 'analyze'):
                if custom_action == 'analyze':
                    # For analyze, we just run the command directly
                    await self.analyze_cmd(update, context, symbol=symbol)
                else:
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

    async def run_monitor(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Start monitoring a symbol."""
        query = update.callback_query
        target = query.message if query else update.message
        chat_id = update.effective_chat.id
        
        self.active_monitors[(chat_id, symbol)] = {
            'interval': interval,
            'last_run': None
        }
        self.db.save_monitor(chat_id, symbol, interval, None)
        
        # Start the background task if not running
        if self._monitor_task is None or self._monitor_task[0].done():
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._monitor_loop(context.application))
            self._monitor_task = (task, loop)
            
        await target.reply_text(
            f"📡 <b>Monitoring Activated!</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"<b>Asset:</b> <code>{symbol}</code>\n"
            f"<b>Interval:</b> <code>{interval}</code>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"<i>I will send you Intelligence & Forecast reports every {interval}.</i>",
            parse_mode='HTML'
        )

    async def monitors_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show active monitors with options to stop them."""
        query = update.callback_query
        if query: await query.answer()
        
        chat_id = update.effective_chat.id
        my_monitors = [(sym, data['interval']) for (cid, sym), data in self.active_monitors.items() if cid == chat_id]
        
        if not my_monitors:
            text = "📴 <b>No active monitors.</b>\nUse the monitoring menu to start tracking assets."
            keyboard = [[InlineKeyboardButton("📡 Monitor New Asset", callback_data='menu_monitor')],
                        [InlineKeyboardButton("⬅️ Back", callback_data='start')]]
        else:
            text = "📋 <b>Active Market Intelligence Tracks:</b>\nSelect an asset to manage its surveillance."
            keyboard = []
            for sym, interval in my_monitors:
                keyboard.append([InlineKeyboardButton(f"🛑 Stop {sym} ({interval})", callback_data=f"stop_mon_{sym}")])
            
            keyboard.append([InlineKeyboardButton("🛑 Stop ALL Monitors", callback_data="stop_mon_ALL")])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data='start')])
            
        markup = InlineKeyboardMarkup(keyboard)
        if query:
            await query.edit_message_text(text, reply_markup=markup, parse_mode='HTML')
        else:
            await update.message.reply_text(text, reply_markup=markup, parse_mode='HTML')

    async def stopmonitor_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop monitoring a symbol."""
        if not context.args:
            await update.message.reply_text("Usage: /stopmonitor <symbol> or /stopmonitor ALL")
            return
        
        symbol = context.args[0].upper()
        chat_id = update.effective_chat.id
        
        if symbol == "ALL":
            keys_to_del = [k for k in self.active_monitors.keys() if k[0] == chat_id]
            for k in keys_to_del:
                del self.active_monitors[k]
            await update.message.reply_text(f"🛑 <b>Stopped all monitors for this chat.</b>", parse_mode='HTML')
        elif (chat_id, symbol) in self.active_monitors:
            del self.active_monitors[(chat_id, symbol)]
            await update.message.reply_text(f"🛑 <b>Stopped monitoring {symbol}.</b>", parse_mode='HTML')
        else:
            await update.message.reply_text(f"❓ <b>Not monitoring {symbol}.</b>", parse_mode='HTML')

    async def clear_positions_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ask for confirmation to clear all positions."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Liquidate All", callback_data='exec_clear'),
                InlineKeyboardButton("❌ No, Keep Them", callback_data='cancel_clear')
            ]
        ]
        text = (
            "⚠️ <b>DANGER: CLEAR ALL POSITIONS</b>\n\n"
            "This will liquidate all open trades in the fund "
            "and reset the database state. This action is irreversible.\n\n"
            "<b>Are you sure?</b>"
        )
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')
        else:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def run_clear_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Execute the liquidation process."""
        query = update.callback_query
        await query.edit_message_text("⏳ <b>Liquidating everything... please wait.</b>", parse_mode='HTML')
        
        try:
            # Instantiate Portfolio Manager
            pm = PortfolioManager()
            
            # Form a minimal context for liquidation
            # We need current prices to log "Close" PnL accurately.
            # We'll pull latest from DB.
            symbols = [r['symbol'] for _, r in self.db.get_all_positions().iterrows()]
            market_data = {}
            for s in symbols:
                df = self.db.load_data(s)
                if not df.empty:
                    market_data[s] = {"latest_close": df.iloc[-1]['close']}
            
            ctx = {
                "symbols": symbols,
                "mode": config.AGENT_MODE,
                "market_data": market_data,
                "_agents": discover_agents()
            }
            
            results = pm.liquidate_all_positions(config.AGENT_MODE, ctx)
            
            if not results:
                # If no results but maybe DB was out of sync, ensure DB is cleared
                self.db.clear_all_positions()
                await query.edit_message_text("✅ <b>Database state cleared.</b> (No open broker positions detected)", parse_mode='HTML', 
                                             reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
                return

            closed_count = len(results)
            await query.edit_message_text(f"✅ <b>Successfully liquidated {closed_count} positions.</b>", 
                                         parse_mode='HTML',
                                         reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
            
        except Exception as e:
            logger.error(f"Error liquidating positions: {e}", exc_info=True)
            await query.edit_message_text(f"❌ <b>Error during liquidation:</b> {str(e)}", parse_mode='HTML',
                                         reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]))
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
        for label, pairs in AGENT_PAIR_GROUPS:
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
        elif data == 'menu_analyze': await self.category_menu(update, context, 'analyze')
        elif data == 'menu_monitor': await self.monitor_menu(update, context)
        elif data == 'menu_monitors': await self.monitors_menu(update, context)
        elif data.startswith('stop_mon_'):
            symbol = data.replace('stop_mon_', '')
            chat_id = update.effective_chat.id
            if symbol == "ALL":
                keys_to_del = [k for k in self.active_monitors.keys() if k[0] == chat_id]
                for k in keys_to_del: 
                    del self.active_monitors[k]
                self.db.delete_monitor(chat_id, "ALL")
                await query.edit_message_text("🛑 <b>All monitors stopped.</b>", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]), parse_mode='HTML')
            else:
                if (chat_id, symbol) in self.active_monitors:
                    del self.active_monitors[(chat_id, symbol)]
                    self.db.delete_monitor(chat_id, symbol)
                    await query.edit_message_text(f"🛑 <b>Stopped monitoring {symbol}.</b>", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='start')]]), parse_mode='HTML')
                else:
                    await query.answer(f"Already stopped: {symbol}")
                    await self.monitors_menu(update, context)
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
            if action in ('train', 'predict', 'tp', 'monitor'):
                await self.interval_menu(update, context, action, symbol)
            elif action == 'analyze':
                await self.analyze_cmd(update, context)
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
            elif action == 'monitor': await self.run_monitor(update, context, symbol, interval)
        elif data.startswith('analyze_'): await self.analyze_cmd(update, context)
        elif data.startswith('conf_set_'):
            key = data.replace('conf_set_', '')
            await self.config_set_prompt(update, context, key)
        elif data == 'confirm_clear':
            await self.clear_positions_cmd(update, context)
        elif data == 'exec_clear':
            await self.run_clear_positions(update, context)
        elif data == 'cancel_clear':
            await self.positions_cmd(update, context)

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display help message."""
        query = update.callback_query
        if query:
            await query.answer()

        text = (
            "🤖 <b>AI Hedge Fund Command Reference</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "<b>General Operations</b>\n"
            "• /start - Access unified command center\n"
            "• /status - Real-time portfolio valuation\n"
            "• /positions - Active trade surveillance\n"
            "• /pnl - Performance & win-rate metrics\n"
            "• /clear - Wipe database position state\n\n"
            "<b>Market Intelligence</b>\n"
            "• /analyze <code>&lt;sym&gt;</code> - Deep CIO deliberation\n"
            "• /predict - ML-driven market forecasting\n"
            "• /tp - Automated Train-to-Predict pipeline\n"
            "• /monitor - Set periodic intelligence alerts\n\n"
            "<b>Fund Management</b>\n"
            "• /monitors - List your active monitors\n"
            "• /stopmonitor <code>&lt;sym&gt;</code> - End surveillance\n"
            "• /config - Adjust risk & reward params\n"
            "• /retrain - Force immediate AI recalibration\n"
            "• /dashboard - Launch Streamlit Analytics\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "<i>Interact with the bot using natural language for AI chat assistance.</i>"
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

def setup_application():
    """Build the Telegram Application object without starting it."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found.")
        return None

    bot = AlgoTelegramBot()
    
    # Increase timeouts for flaky networks
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).request(request).build()

    # Commands
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("status", bot.status_cmd))
    app.add_handler(CommandHandler("pnl", bot.pnl_cmd))
    app.add_handler(CommandHandler("positions", bot.positions_cmd))
    # Note: ensure bot has clear_positions_cmd defined
    if hasattr(bot, 'clear_positions_cmd'):
        app.add_handler(CommandHandler("clear", bot.clear_positions_cmd))
    app.add_handler(CommandHandler("analyze", bot.analyze_cmd))
    app.add_handler(CommandHandler("retrain", bot.retrain_cmd))
    app.add_handler(CommandHandler("settings", bot.settings_cmd))
    app.add_handler(CommandHandler("train", bot.train_menu))
    app.add_handler(CommandHandler("predict", bot.predict_menu))
    app.add_handler(CommandHandler("backtest", bot.backtest_menu))
    app.add_handler(CommandHandler("paper", bot.paper_menu))
    app.add_handler(CommandHandler("config", bot.config_menu))
    if hasattr(bot, 'train_predict_menu'):
        app.add_handler(CommandHandler("tp", bot.train_predict_menu))
    app.add_handler(CommandHandler("monitor", bot.monitor_menu))
    app.add_handler(CommandHandler("monitors", bot.monitors_menu))
    if hasattr(bot, 'stopmonitor_cmd'):
        app.add_handler(CommandHandler("stopmonitor", bot.stopmonitor_cmd))
    if hasattr(bot, 'dashboard_cmd'):
        app.add_handler(CommandHandler("dashboard", bot.dashboard_cmd))
    if hasattr(bot, 'list_symbols_cmd'):
        app.add_handler(CommandHandler("list_symbols", bot.list_symbols_cmd))
    app.add_handler(CommandHandler("help", bot.help_cmd))

    # Setup background monitoring loop
    async def post_init(application: Application):
        if bot.active_monitors:
            task = asyncio.create_task(bot._monitor_loop(application))
            bot._monitor_task = (task, asyncio.get_running_loop())
            logger.info("Background monitoring task created via post_init.")

    # Button Callbacks
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    # Non-command messages (Chat / Custom Input)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.chat_handler))

    # Build application with post_init
    app.post_init = post_init
    
    return app, bot

def main():
    app, bot = setup_application()
    if not app:
        return

    print("🤖 Telegram Bot is running (Polling)...")
    # Add bootstrap retries to handle initial connection issues
    app.run_polling(bootstrap_retries=5)

if __name__ == "__main__":
    main()
