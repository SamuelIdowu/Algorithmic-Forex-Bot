import logging
import asyncio
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.request import HTTPXRequest

import utils.config as config
from data.db_manager import DatabaseManager
from services.prediction_tracker_service import PredictionTrackerService, format_symbol
from agents.chief_investment_officer import ChiefInvestmentOfficer
from agents.sentiment_analyst import SentimentAnalyst
from agents.registry import discover_agents
from cli import COMMON_SYMBOLS, AGENT_PAIR_GROUPS

# Notional baseline for system calculations
NOTIONAL_CAPITAL = 10000.0

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
        self.prediction_tracker_service = PredictionTrackerService(self.db)
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
                InlineKeyboardButton("\U0001F4CA Status", callback_data='status'),
                InlineKeyboardButton("\U0001F4B0 Predictions", callback_data='predictions'),
                InlineKeyboardButton("\U0001F4CA Recent Signals", callback_data='recent_signals'),
            ],
            [
                InlineKeyboardButton("\U0001F50D Analyze", callback_data='menu_analyze'),
                InlineKeyboardButton("\U0001F52E Predict", callback_data='menu_predict'),
                InlineKeyboardButton("\U0001F4CB Trackers", callback_data='menu_monitors'),
            ],
            [
                InlineKeyboardButton("\U0001F4C8 Train", callback_data='menu_train'),
                InlineKeyboardButton("\U0001F4E1 Monitor", callback_data='menu_monitor'),
                InlineKeyboardButton("\u2699\uFE0F Config", callback_data='config_menu'),
            ],
            [
                InlineKeyboardButton("\u2753 Help", callback_data='help'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        target = update.effective_message or update.effective_chat
        send_func = target.reply_text if hasattr(target, 'reply_text') else target.send_message
        await send_func(
            "\U0001F680 <b>ENSOTRADE Insights Terminal</b>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            "Welcome, Commander. I am your AI market analysis engine. "
            "I'm currently scanning global markets and generating institutional-grade predictions.\n\n"
            "<b>AI-powered market intelligence</b> is at your fingertips. Use the dashboard below to access your insights.",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

    async def category_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str):
        """Show categories for symbol selection."""
        query = update.callback_query
        if query: await query.answer()

        keyboard = []
        cat_items = list(SYMBOL_CATEGORIES.items())
        for i in range(0, len(cat_items), 2):
            row = [InlineKeyboardButton(cat_items[i][1], callback_data=f"cat_{action}_{cat_items[i][0]}")]
            if i + 1 < len(cat_items):
                row.append(InlineKeyboardButton(cat_items[i+1][1], callback_data=f"cat_{action}_{cat_items[i+1][0]}"))
            keyboard.append(row)

        keyboard.append([InlineKeyboardButton("\u2795 Custom Ticker", callback_data=f"cat_{action}_custom")])
        keyboard.append([InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')])

        text = f"\U0001F50D <b>Select Category for {action.capitalize()}:</b>"
        await (query.edit_message_text if query else update.message.reply_text)(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML'
        )

    async def symbol_picker(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, category: str, page: int = 0):
        """Show symbols in a category."""
        query = update.callback_query
        await query.answer()

        target_label = SYMBOL_CATEGORIES.get(category, "")
        found_pairs = []
        for label, pairs in AGENT_PAIR_GROUPS:
            if target_label.lower() in label.lower():
                found_pairs = pairs
                break

        if not found_pairs:
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
            nav_row.append(InlineKeyboardButton("\u2B05\uFE0F Prev", callback_data=f"cat_{action}_{category}_{page-1}"))
        if end_idx < len(found_pairs):
            nav_row.append(InlineKeyboardButton("Next \u27A1\uFE0F", callback_data=f"cat_{action}_{category}_{page+1}"))
        if nav_row: keyboard.append(nav_row)

        keyboard.append([InlineKeyboardButton("\U0001F519 Back to Categories", callback_data=f"menu_{action}")])

        await query.edit_message_text(f"\U0001F4CB <b>{target_label} Symbols:</b>",
                                      reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def custom_ticker_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str):
        """Prompt for manual ticker entry."""
        query = update.callback_query
        await query.answer()
        context.user_data['custom_ticker_action'] = action
        await query.edit_message_text("\u2328\uFE0F <b>Enter Custom Ticker:</b>\n<i>(e.g., AAPL, NVDA, ETH-USD)</i>", parse_mode='HTML')

    async def train_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.category_menu(update, context, "train")

    async def predict_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args:
            symbol = context.args[0].upper()
            await self.interval_menu(update, context, "predict", symbol)
            return
        await self.category_menu(update, context, "predict")

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

        keyboard.append([InlineKeyboardButton("\u2B05\uFE0F Back", callback_data=f"menu_{action}")])
        text = f"\u23F1 <b>Select Interval for {symbol}:</b>"
        await (query.edit_message_text if query else update.message.reply_text)(
            text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML'
        )

    async def run_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run training in a thread."""
        query = update.callback_query
        target = query.message if query else update.message
        msg = await target.reply_text(f"\U0001F69C Training model for {symbol} ({interval})...")

        try:
            from train_model import train_model
            _, _, acc = await asyncio.to_thread(train_model, symbol=symbol, interval=interval)
            await msg.edit_text(
                f"\u2705 <b>Model Optimized!</b>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"<b>Asset:</b> <code>{symbol}</code>\n"
                f"<b>Timeframe:</b> <code>{interval}</code>\n"
                f"<b>Backtest Accuracy:</b> <code>{acc:.2f}</code>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"<i>The AI has updated its internal weights for this asset.</i>",
                parse_mode='HTML'
            )
        except Exception as e:
            await msg.edit_text(f"\u274C Training failed: {e}")

    async def run_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run prediction in a thread."""
        query = update.callback_query
        target = query.message if query else update.message
        msg = await target.reply_text(f"\U0001F52E Fetching prediction for {symbol} ({interval})...")

        try:
            from predict import predict_next_movement
            safe_name = symbol.lower().replace('/', '_').replace('=', '_')
            tf_suffix = f"_{interval}" if interval != "1d" else ""
            m_path = f"models/{safe_name}{tf_suffix}_model.pkl"
            s_path = f"models/{safe_name}{tf_suffix}_scaler.pkl"

            res = await asyncio.to_thread(predict_next_movement, symbol=symbol, interval=interval, model_path=m_path, scaler_path=s_path, tp_mult=config.ATR_TP_MULT, sl_mult=config.ATR_SL_MULT, include_insights=True)
            if res:
                emoji = "\U0001F680" if res['prediction'] == "UP" else "\U0001F4C9"
                rr_ratio = abs(res['tp'] - res['current_price']) / abs(res['current_price'] - res['sl']) if abs(res['current_price'] - res['sl']) != 0 else 0

                text = (
                    f"\U0001F52E <b>{symbol} Forecast ({interval})</b>\n"
                    "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                    f"<b>Direction:</b> {emoji} <code>{res['prediction']}</code>\n"
                    f"<b>Confidence:</b> <code>{res['confidence']:.1%}</code>\n"
                    "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                    f"<b>Entry Price:</b> <code>{config.format_price(res['current_price'])}</code>\n"
                    f"<b>Target (TP):</b> <code>{config.format_price(res['tp'])}</code>\n"
                    f"<b>Stop (SL):</b>   <code>{config.format_price(res['sl'])}</code>\n"
                    f"<b>R/R Ratio:</b>  <code>{rr_ratio:.2f}</code>\n"
                )

                # Add agent insights if available
                insights = res.get('agent_insights')
                if insights:
                    text += "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                    text += "\U0001F916 <b>AI Agent Insights</b>\n"

                    # Quant
                    quant = insights.get("quant", {})
                    q_signal = quant.get("signal", "N/A")
                    q_conf = quant.get("confidence", 0)
                    text += f"\u2022 <b>Quant:</b> <code>{q_signal}</code> ({q_conf:.1%})\n"

                    # Sentiment
                    sentiment = insights.get("sentiment", {})
                    s_signal = sentiment.get("signal", "N/A")
                    s_score = sentiment.get("score", 0)
                    text += f"\u2022 <b>Sentiment:</b> <code>{s_signal}</code> ({s_score:.3f})\n"

                    # Fundamentals
                    fundamentals = insights.get("fundamentals", {})
                    f_signal = fundamentals.get("signal", "N/A")
                    text += f"\u2022 <b>Fundamentals:</b> <code>{f_signal}</code>\n"

                    # CIO
                    cio = insights.get("cio", {})
                    cio_signal = cio.get("signal", "N/A")
                    text += f"\u2022 <b>CIO:</b> <code>{cio_signal}</code>\n"

                    # Risk Manager
                    risk = insights.get("risk_manager", {})
                    r_action = risk.get("action", "HOLD")
                    r_score = risk.get("vote_score", 0)
                    weights = risk.get("weights", {})
                    text += f"\u2022 <b>Risk Mgr:</b> <code>{r_action}</code> ({r_score:.3f})\n"
                    if weights:
                        text += f"   <i>Weights: Q:{weights.get('quant', 0):.0%} S:{weights.get('sentiment', 0):.0%} F:{weights.get('fundamentals', 0):.0%}</i>\n"

                    # CIO Memo (truncated)
                    cio_memo = cio.get("memo", "")
                    if cio_memo:
                        text += f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                        text += f"<i>\u201C{cio_memo[:150]}{'...' if len(cio_memo) > 150 else ''}</i>\n"

                text += "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                text += "<i>Forecast generated based on latest ML model + multi-agent AI analysis.</i>"

                await msg.edit_text(text, parse_mode='HTML')
            else:
                await msg.edit_text(f"\u274C <b>Could not generate prediction for {symbol}.</b>\nIs the model trained? Checked: <code>{m_path}</code>", parse_mode='HTML')
        except Exception as e:
            await msg.edit_text(f"\u274C <b>Prediction failed:</b>\n<code>{e}</code>", parse_mode='HTML')

    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get system status with prediction metrics."""
        query = update.callback_query
        if query:
            await query.answer()

        val = self.db.get_prediction_baseline() or NOTIONAL_CAPITAL

        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_trades WHERE created_at >= datetime('now', '-24 hours')")
            recent_trades = cursor.fetchone()[0] or 0
        except:
            recent_trades = 0

        text = (
            "\U0001F916 <b>ENSOTRADE System Status</b>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"\U0001F4CA <b>Baseline Value:</b>  <code>${val:,.2f}</code>\n"
            f"\U0001F504 <b>24h Trades:</b>      <code>{recent_trades}</code>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"\U0001F916 <b>Engine:</b> <code>ENSOTRADE v3.0</code>\n"
            f"\u23F1 <b>Cycle:</b> <code>{config.AGENT_INTERVAL_MINUTES}min</code>"
        )

        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]])

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                target = update.effective_message or update.effective_chat
                await (target.send_message(text, parse_mode='HTML', reply_markup=reply_markup) if hasattr(target, 'send_message') else target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup))
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def predictions_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show prediction accuracy and analyst performance metrics."""
        query = update.callback_query
        if query:
            await query.answer()

        preds = self.db.get_recent_predictions(limit=500)
        if preds.empty:
            text = "\U0001F7E1 <b>No predictions recorded yet.</b>\nRun /analyze or /predict to start generating signals."
        else:
            total = len(preds)
            buy_signals = preds[preds['action'] == 'BUY']
            sell_signals = preds[preds['action'] == 'SELL']
            hold_signals = preds[preds['action'] == 'HOLD']

            avg_confidence = preds['quant_confidence'].mean() if 'quant_confidence' in preds.columns else 0
            avg_score = preds['vote_score'].mean() if 'vote_score' in preds.columns else 0

            # Check prediction accuracy against current prices
            correct = 0
            checked = 0
            for _, p in preds.iterrows():
                if p.get('current_price') and p.get('entry_price') and p['entry_price'] > 0:
                    actual_move = "UP" if p['current_price'] > p['entry_price'] else "DOWN"
                    predicted_dir = "UP" if p['action'] == 'BUY' else ("DOWN" if p['action'] == 'SELL' else None)
                    if predicted_dir:
                        checked += 1
                        if actual_move == predicted_dir:
                            correct += 1

            accuracy = (correct / checked * 100) if checked > 0 else 0

            # Per-symbol breakdown
            symbol_stats = []
            for sym in preds['symbol'].unique():
                sym_preds = preds[preds['symbol'] == sym]
                sym_count = len(sym_preds)
                sym_avg_conf = sym_preds['quant_confidence'].mean() if 'quant_confidence' in sym_preds.columns else 0
                sym_buy = len(sym_preds[sym_preds['action'] == 'BUY'])
                sym_sell = len(sym_preds[sym_preds['action'] == 'SELL'])
                symbol_stats.append(f"  \u2022 <code>{sym}</code>: {sym_count} predictions (B:{sym_buy}/S:{sym_sell}) | Avg conf: {sym_avg_conf:.1%}")

            text = (
                "\U0001F4B0 <b>Prediction Analytics</b>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"\U0001F52E <b>Total Predictions:</b> <code>{total}</code>\n"
                f"\U0001F3AF <b>Directional Accuracy:</b> <code>{accuracy:.1f}%</code> ({checked} checked)\n"
                f"\U0001F4CA <b>Avg Confidence:</b> <code>{avg_confidence:.1%}</code>\n"
                f"\U0001F4C8 <b>Avg Vote Score:</b> <code>{avg_score:.3f}</code>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"\U0001F680 <b>BUY Signals:</b> <code>{len(buy_signals)}</code>\n"
                f"\U0001F4C9 <b>SELL Signals:</b> <code>{len(sell_signals)}</code>\n"
                f"\u2696\uFE0F <b>HOLD Signals:</b> <code>{len(hold_signals)}</code>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"<b>By Symbol:</b>\n" + "\n".join(symbol_stats[:10])
            )

        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]])

        if query:
            try:
                await query.edit_message_text(text, parse_mode='HTML', reply_markup=reply_markup)
            except:
                target = update.effective_message or update.effective_chat
                await (target.send_message(text, parse_mode='HTML', reply_markup=reply_markup) if hasattr(target, 'send_message') else target.reply_text(text, parse_mode='HTML', reply_markup=reply_markup))
        else:
            await update.message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def recent_signals_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show the most recent prediction signals with TP/SL levels."""
        query = update.callback_query
        if query:
            await query.answer()

        preds = self.db.get_recent_predictions(limit=15)
        if preds.empty:
            text = "\u26AA\uFE0F <b>No recent signals recorded.</b>\nStart analyzing assets to build a signal history."
        else:
            rows = []
            for _, r in preds.iterrows():
                symbol = r['symbol']
                action = r.get('action', 'HOLD')
                entry = r.get('entry_price', 0)
                tp = r.get('take_profit', 0)
                sl = r.get('stop_loss', 0)
                score = r.get('vote_score', 0)
                conf = r.get('quant_confidence', 0)
                current = r.get('current_price', 0)
                memo = r.get('cio_memo', '')[:80]

                if action == 'BUY':
                    sig_emoji = "\U0001F680"
                elif action == 'SELL':
                    sig_emoji = "\U0001F4C9"
                else:
                    sig_emoji = "\u2696\uFE0F"

                pnl_str = ""
                if current and entry and entry > 0:
                    pnl_pct = ((current - entry) / entry) * 100
                    pnl_str = f" | PnL: {pnl_pct:+.2f}%"

                ts = r.get('timestamp', '')
                time_str = ts[:16] if ts else ''

                rows.append(
                    f"{sig_emoji} <b>{symbol}</b> <code>{action}</code> <i>({time_str})</i>\n"
                    f"   Entry: <code>{config.format_price(entry)}</code> | TP: <code>{config.format_price(tp)}</code> | SL: <code>{config.format_price(sl)}</code>{pnl_str}\n"
                    f"   Score: <code>{score:.3f}</code> | Conf: <code>{conf:.1%}</code>\n"
                    f"   <i>{memo}...</i>" if memo else ""
                )

            text = (
                "\U0001F4CA <b>Recent Prediction Signals</b>\n"
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                + "\n\n".join(rows)
            )

        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]])

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
            await target.reply_text("\u2753 <b>Please specify a symbol, e.g., /analyze BTC-USD</b>", parse_mode='HTML')
            return

        msg_ref = await (query.message.reply_text if query else update.message.reply_text)(f"\U0001F50D <b>Scrutinizing {symbol}...</b>", parse_mode='HTML')

        try:
            res = await self.perform_combined_analysis(symbol, "1h")
            if res:
                await msg_ref.edit_text(res['text'], parse_mode='HTML')
            else:
                await msg_ref.edit_text(f"\U0001F7E1 <b>Could not perform fresh analysis for {symbol}.</b>")
        except Exception as e:
            await msg_ref.edit_text(f"\u274C <b>Error during analysis:</b>\n<code>{e}</code>", parse_mode='HTML')

    async def perform_combined_analysis(self, symbol: str, interval: str) -> dict:
        """Fetch prediction and perform CIO analysis for a symbol."""
        try:
            from predict import predict_next_movement
            safe_name = symbol.lower().replace('/', '_').replace('=', '_')
            tf_suffix = f"_{interval}" if interval != "1d" else ""
            m_path = f"models/{safe_name}{tf_suffix}_model.pkl"
            s_path = f"models/{safe_name}{tf_suffix}_scaler.pkl"

            pred_res = await asyncio.to_thread(predict_next_movement, symbol, m_path, s_path, interval=interval, tp_mult=config.ATR_TP_MULT, sl_mult=config.ATR_SL_MULT)

            sent_context = {"symbols": [symbol]}
            sent_res = await asyncio.to_thread(self.sentiment_bot.run, sent_context)
            sentiment = sent_res.get("sentiment", {}).get(symbol, {})

            cio_context = {
                "symbols": [symbol],
                "quant": {symbol: {"reasoning": pred_res.get('reasoning', "Technical signal generated.") if pred_res else "No technical data."}},
                "sentiment": {symbol: sentiment}
            }
            cio_res = await asyncio.to_thread(self.cio.run, cio_context)
            cio_data = cio_res.get("cio", {}).get(symbol, {})

            header = f"\U0001F514 <b>MONITOR ALERT: {symbol} ({interval})</b>\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            footer = f"\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\u23F0 <i>Report generated at: {datetime.now().strftime('%H:%M:%S')}</i>"

            sig_emoji = {"BULLISH": "\U0001F680", "BEARISH": "\U0001F4C9", "NEUTRAL": "\u2696\uFE0F"}.get(cio_data.get('cio_signal'), "\u2753")

            intel_text = (
                f"\U0001F9E0 <b>Intelligence:</b> {sig_emoji} <code>{cio_data.get('cio_signal', 'NEUTRAL')}</code>\n"
                f"\U0001F4DD <b>CIO Memo:</b>\n<blockquote>{cio_data.get('memo', 'No analysis available.')}</blockquote>"
            )

            if pred_res:
                tech_dir = pred_res['prediction']
                sent_dir = sentiment.get('sentiment_signal', 'NEUTRAL')
                if (tech_dir == "UP" and sent_dir == "BEARISH") or (tech_dir == "DOWN" and sent_dir == "BULLISH"):
                    header = f"\u26A0\uFE0F <b>DIVERGENCE ALERT: {symbol}</b>\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                    intel_text = "\U0001F6A9 <b>CONTRADICTION DETECTED:</b> Tech & Sentiment are misaligned.\n" + intel_text

            if pred_res:
                emoji = "\U0001F680" if pred_res['prediction'] == "UP" else "\U0001F4C9"
                rsi_val = pred_res.get('rsi', 50)
                rsi_desc = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"

                val = self.db.get_prediction_baseline() or NOTIONAL_CAPITAL
                atr = pred_res.get('atr', 0)
                risk_pct = config.RISK_PER_TRADE_PCT
                sl_mult = config.ATR_SL_MULT
                rec_size = (val * risk_pct) / (atr * sl_mult) if atr > 0 else 0

                tech_text = (
                    f"🔮 <b>Forecast:</b> {emoji} <code>{pred_res['prediction']}</code> ({pred_res['confidence']:.1%})\n"
                    f"📊 <b>Technicals:</b> RSI: <code>{rsi_val:.1f}</code> ({rsi_desc}) | ATR: <code>{atr:.4f}</code>\n"
                    f"📈 <b>Levels:</b> Entry: <code>{config.format_price(pred_res['current_price'])}</code> | TP: <code>{config.format_price(pred_res['tp'])}</code>\n"
                    f"💰 <b>Rec. Size:</b> <code>{rec_size:.4f}</code> (1% Risk)\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                )

                news_text = ""
                news_list = self.db.get_news(symbol, limit=2)
                if news_list:
                    news_text = "\U0001F4F0 <b>Recent Headlines:</b>\n"
                    for n in news_list:
                        news_text += f"\u2022 <i>{n['title'][:60]}...</i>\n"
                    news_text += "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"

                return {
                    'text': header + tech_text + news_text + intel_text + footer,
                    'signal': pred_res['prediction'],
                    'confidence': pred_res['confidence']
                }
            else:
                tech_text = "\U0001F52E <b>Forecast:</b> \u26A0\uFE0F <code>TECHNICAL DATA UNAVAILABLE</code>\n"
                tech_text += "<i>Possible causes: Insufficient liquidity or API rate limit.</i>\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
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
        logger.info("\U0001F4E1 Monitoring background loop STARTED.")
        while True:
            try:
                now = datetime.now()
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

                    minutes = 60
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
                        logger.info(f"\U0001F504 Running monitor task: {symbol} ({interval_str}) for chat {chat_id}")
                        self.active_monitors[key]['last_run'] = now
                        self.db.save_monitor(chat_id, symbol, interval_str, now)

                        try:
                            res = await asyncio.wait_for(self.perform_combined_analysis(symbol, interval_str), timeout=120)
                            if res:
                                await bot_app.bot.send_message(chat_id=chat_id, text=res['text'], parse_mode='HTML')
                                logger.info(f"\u2705 Monitor alert sent: {symbol} to {chat_id}")
                        except asyncio.TimeoutError:
                            logger.error(f"\u231B Monitor timeout for {symbol}")
                        except Exception as e:
                            logger.error(f"\u274C Error processing monitor for {symbol}: {e}")

                await asyncio.sleep(15)

            except asyncio.CancelledError:
                logger.info("Monitoring loop STOPPED.")
                break
            except Exception as e:
                logger.error(f"\U0001F525 Critical error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def retrain_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger manual retraining."""
        if not context.args:
            await update.message.reply_text("Usage: /retrain <symbol>")
            return

        symbol = context.args[0].upper()
        msg = await update.message.reply_text(f"\U0001F69C Triggering retrain for {symbol}...")

        try:
            from run_agent import trigger_retrain
            trigger_retrain(symbol)
            await msg.edit_text(f"\u2705 Retrain cycle initiated for {symbol}. Check logs for progress.")
        except Exception as e:
            await msg.edit_text(f"\u274C Failed to trigger retrain: {e}")

    async def settings_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot settings."""
        query = update.callback_query
        if query:
            await query.answer()

        text = (
            "\u2699\uFE0F <b>Current Settings</b>\n"
            f"Symbols: {', '.join(config.AGENT_SYMBOLS)}\n"
            f"Interval: {config.AGENT_INTERVAL_MINUTES}m\n"
            f"ATR Stop Loss: {config.ATR_SL_MULT}x\n"
            f"ATR Take Profit: {config.ATR_TP_MULT}x\n"
            f"Risk per Trade: {config.RISK_PER_TRADE_PCT*100}%\n"
            f"Required Vote Score: {config.REQUIRED_VOTE_SCORE}"
        )
        keyboard = [[InlineKeyboardButton("\u270F\uFE0F Edit Config", callback_data='config_menu')],
                    [InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]]
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

        # Check IF user is currently updating a config setting
        setting_key = context.user_data.get('setting_to_change')
        if setting_key:
            try:
                val = float(user_text)
                from utils.config import write_config
                write_config({setting_key: str(val)})

                context.user_data['setting_to_change'] = None
                await update.message.reply_text(f"\u2705 <b>Updated {setting_key} to {val}</b>\n(Note: Restart the bot or agent loop to apply some deep changes).", parse_mode='HTML')
                await self.config_menu(update, context)
                return
            except ValueError:
                await update.message.reply_text("\u274C Invalid value. Please enter a number (e.g. 1.5). Type /cancel to stop.")
                return

        # Check IF user is providing a custom ticker
        custom_action = context.user_data.get('custom_ticker_action')
        if custom_action:
            symbol = user_text.strip().upper()
            context.user_data['custom_ticker_action'] = None
            if custom_action in ('train', 'predict', 'tp', 'analyze'):
                if custom_action == 'analyze':
                    await self.analyze_cmd(update, context, symbol=symbol)
                else:
                    await self.interval_menu(update, context, custom_action, symbol)
            return

        # Normal AI Chat
        client = self.groq_client
        if not client:
            await update.message.reply_text("Groq API key missing or library not installed. AI chat disabled.")
            return

        system_prompt = (
            "You are a helpful AI Market Analysis Assistant. You monitor an algorithmic prediction engine. "
            "Answer the user's questions about the bot, trading strategies, or prediction performance. "
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

        if self._monitor_task is None or self._monitor_task[0].done():
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._monitor_loop(context.application))
            self._monitor_task = (task, loop)

        await target.reply_text(
            f"\U0001F4E1 <b>Monitoring Activated!</b>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"<b>Asset:</b> <code>{symbol}</code>\n"
            f"<b>Interval:</b> <code>{interval}</code>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
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
            text = "\U0001F4F4 <b>No active monitors.</b>\nUse the monitoring menu to start tracking assets."
            keyboard = [[InlineKeyboardButton("\U0001F4E1 Monitor New Asset", callback_data='menu_monitor')],
                        [InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]]
        else:
            text = "\U0001F4CB <b>Active Market Intelligence Tracks:</b>\nSelect an asset to manage its surveillance."
            keyboard = []
            for sym, interval in my_monitors:
                keyboard.append([InlineKeyboardButton(f"\U0001F6D1 Stop {sym} ({interval})", callback_data=f"stop_mon_{sym}")])

            keyboard.append([InlineKeyboardButton("\U0001F6D1 Stop ALL Monitors", callback_data="stop_mon_ALL")])
            keyboard.append([InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')])

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
            await update.message.reply_text(f"\U0001F6D1 <b>Stopped all monitors for this chat.</b>", parse_mode='HTML')
        elif (chat_id, symbol) in self.active_monitors:
            del self.active_monitors[(chat_id, symbol)]
            await update.message.reply_text(f"\U0001F6D1 <b>Stopped monitoring {symbol}.</b>", parse_mode='HTML')
        else:
            await update.message.reply_text(f"\u2753 <b>Not monitoring {symbol}.</b>", parse_mode='HTML')

    async def config_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show configuration menu."""
        query = update.callback_query
        if query: await query.answer()

        keyboard = [
            [InlineKeyboardButton(f"Stop Loss: {config.ATR_SL_MULT}x", callback_data='conf_set_ATR_SL_MULT')],
            [InlineKeyboardButton(f"Take Profit: {config.ATR_TP_MULT}x", callback_data='conf_set_ATR_TP_MULT')],
            [InlineKeyboardButton(f"Risk/Trade: {config.RISK_PER_TRADE_PCT*100}%", callback_data='conf_set_RISK_PER_TRADE_PCT')],
            [InlineKeyboardButton(f"Vote Score: {config.REQUIRED_VOTE_SCORE}", callback_data='conf_set_REQUIRED_VOTE_SCORE')],
            [InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]
        ]
        text = "\u2699\uFE0F <b>Interactive Configuration</b>\nSelect a setting to modify it:"
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
        await query.edit_message_text(f"\U0001F4DD <b>Updating {key}</b>\nPlease type the new value (e.g., 2.5):", parse_mode='HTML')

    async def dashboard_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send dashboard link."""
        await update.message.reply_text("\U0001F4CA <b>AI Hedge Fund Dashboard</b>\nLink: <a href='http://localhost:8501'>Open Dashboard</a>\n<i>Note: Dashboard must be running on your host machine.</i>", parse_mode='HTML')

    async def list_symbols_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all supported symbols grouped by category."""
        lines = ["\U0001F4CB <b>Supported Symbols:</b>\n"]
        for label, pairs in AGENT_PAIR_GROUPS:
            lines.append(f"<b>{label}</b>")
            for display, ticker in pairs:
                lines.append(f"\u2022 {display}")
            lines.append("")

        await update.message.reply_text("\n".join(lines), parse_mode='HTML')

    async def train_predict_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show options for training then predicting."""
        query = update.callback_query
        await query.answer()
        keyboard = [
            [InlineKeyboardButton("BTC-USD (1h)", callback_data='tp_run_BTC-USD_1h')],
            [InlineKeyboardButton("Gold (1d)", callback_data='tp_run_GC=F_1d')],
            [InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]
        ]
        await query.edit_message_text("\U0001F4C8\U0001F52E <b>Train \u2192 Predict Chain:</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def run_train_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str):
        """Run train then predict."""
        await self.run_train(update, context, symbol, interval)
        await self.run_predict(update, context, symbol, interval)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses."""
        query = update.callback_query
        data = query.data

        if data == 'start': await self.start(update, context)
        elif data == 'status': await self.status_cmd(update, context)
        elif data == 'predictions': await self.predictions_cmd(update, context)
        elif data == 'recent_signals': await self.recent_signals_cmd(update, context)
        elif data == 'settings': await self.settings_cmd(update, context)
        elif data == 'config_menu': await self.config_menu(update, context)
        elif data == 'help': await self.help_cmd(update, context)
        elif data == 'menu_train': await self.train_menu(update, context)
        elif data == 'menu_predict': await self.predict_menu(update, context)
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
                await query.edit_message_text("\U0001F6D1 <b>All monitors stopped.</b>", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]]), parse_mode='HTML')
            else:
                if (chat_id, symbol) in self.active_monitors:
                    del self.active_monitors[(chat_id, symbol)]
                    self.db.delete_monitor(chat_id, symbol)
                    await query.edit_message_text(f"\U0001F6D1 <b>Stopped monitoring {symbol}.</b>", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]]), parse_mode='HTML')
                else:
                    await query.answer(f"Already stopped: {symbol}")
                    await self.monitors_menu(update, context)
        elif data.startswith('cat_'):
            parts = data.split('_')
            action = parts[1]
            category = parts[2]
            page = int(parts[3]) if len(parts) > 3 else 0
            if category == 'custom':
                await self.custom_ticker_prompt(update, context, action)
            else:
                await self.symbol_picker(update, context, action, category, page)
        elif data.startswith('sel_'):
            _, action, symbol = data.split('_', 2)
            if action in ('train', 'predict', 'tp', 'monitor'):
                await self.interval_menu(update, context, action, symbol)
            elif action == 'analyze':
                await self.analyze_cmd(update, context)
        elif data.startswith('run_'):
            _, action, symbol, interval = data.split('_', 3)
            if action == 'train': await self.run_train(update, context, symbol, interval)
            elif action == 'predict': await self.run_predict(update, context, symbol, interval)
            elif action == 'tp': await self.run_train_predict(update, context, symbol, interval)
            elif action == 'monitor': await self.run_monitor(update, context, symbol, interval)
        elif data.startswith('analyze_'): await self.analyze_cmd(update, context)
        elif data.startswith('conf_set_'):
            key = data.replace('conf_set_', '')
            await self.config_set_prompt(update, context, key)

    async def my_predictions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active predictions."""
        symbol = None
        if context.args:
            symbol = " ".join(context.args).strip().upper()

        try:
            summary = self.prediction_tracker_service.get_active_predictions_summary(symbol)
            if not summary or "No active predictions" in summary:
                await update.message.reply_text(
                    "\U0001F4C8 <b>No active predictions found.</b>\n"
                    "Use /predict to create a new prediction tracker."
                )
            else:
                await update.message.reply_text(summary, parse_mode=None)
        except Exception as e:
            logger.error(f"Error in /my_predictions: {e}")
            await update.message.reply_text("\u274C <b>Failed to fetch predictions.</b>\nPlease try again later.")

    async def check_prediction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check current price for a symbol's prediction."""
        if not context.args:
            await update.message.reply_text(
                "\U0001F4DD <b>Usage:</b> <code>/check EUR/USD</code>\n"
                "Shows the most recent active prediction for that symbol with live price.",
                parse_mode='HTML'
            )
            return

        symbol = " ".join(context.args).strip().upper()

        try:
            trackers_df = self.db.get_active_trackers(symbol)
            if trackers_df.empty:
                await update.message.reply_text(
                    f"\U0001F4C8 <b>No active predictions for {symbol}.</b>\n"
                    "Use /predict to create a new prediction."
                )
                return

            most_recent = trackers_df.iloc[0]
            tracker_id = most_recent["id"]

            result = self.prediction_tracker_service.check_prediction(tracker_id)
            status_text = self.prediction_tracker_service.get_prediction_status_text(tracker_id)

            if result.get("current_price") is None and result.get("outcome") == "PRICE_FETCH_FAILED":
                await update.message.reply_text(
                    f"\u2753 <b>Could not fetch live price for {symbol}.</b>\n"
                    "The market may be closed or the API is unavailable."
                )
                return

            await update.message.reply_text(status_text)
        except Exception as e:
            logger.error(f"Error in /check: {e}")
            await update.message.reply_text("\u274C <b>Failed to check prediction.</b>\nPlease try again later.")

    async def prediction_detail(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show details of a specific prediction tracker."""
        if not context.args:
            await update.message.reply_text(
                "\U0001F4DD <b>Usage:</b> <code>/prediction_detail &lt;id&gt;</code>\n"
                "Example: <code>/prediction_detail 42</code>",
                parse_mode='HTML'
            )
            return

        try:
            tracker_id = int(context.args[0])
        except (ValueError, IndexError):
            await update.message.reply_text("\u274C <b>Invalid tracker ID.</b>\nPlease provide a numeric ID.")
            return

        try:
            tracker = self.db.get_tracker_by_id(tracker_id)
            if not tracker:
                await update.message.reply_text(f"\u274C <b>Tracker #{tracker_id} not found.</b>")
                return

            symbol = tracker["symbol"]
            status = tracker["status"]
            direction = tracker["direction"]
            timeframe = tracker["timeframe"]
            entry = tracker["entry_price"]
            tp = tracker.get("take_profit")
            sl = tracker.get("stop_loss")
            current = tracker.get("current_price")
            pnl = tracker.get("pnl_percent")
            max_profit = tracker.get("max_profit_reached")
            max_loss = tracker.get("max_loss_reached")
            created_at = tracker.get("created_at", "")
            expires_at_str = tracker.get("expires_at")
            rsi = tracker.get("rsi")
            atr = tracker.get("atr")
            bb_pos = tracker.get("bb_pos")
            ml_conf = tracker.get("ml_confidence")
            quant_conf = tracker.get("quant_confidence")
            vote_score = tracker.get("vote_score")
            reasoning = tracker.get("reasoning", "")
            agent_insights_json = tracker.get("agent_insights_json")
            direction_result = tracker.get("direction_result")

            dir_emoji = "\U0001F680" if direction in ("UP", "BUY") else "\U0001F4C9"

            lines = [
                f"{dir_emoji} <b>Tracker #{tracker_id} Detail</b>",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                f"<b>Symbol:</b> <code>{symbol}</code>",
                f"<b>Status:</b> <code>{status}</code>",
                f"<b>Direction:</b> {direction} | <b>Timeframe:</b> <code>{timeframe}</code>",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                f"<b>Entry:</b> <code>{config.format_price(entry)}</code>",
            ]

            if tp is not None:
                lines.append(f"<b>Take Profit:</b> <code>{config.format_price(tp)}</code>")
            if sl is not None:
                lines.append(f"<b>Stop Loss:</b> <code>{config.format_price(sl)}</code>")

            if current is not None:
                pnl_sign = "+" if pnl and pnl >= 0 else ""
                pnl_str = f"{pnl_sign}{pnl:.2f}%" if pnl is not None else "N/A"
                lines.append(f"<b>Current Price:</b> <code>{config.format_price(current)}</code>")
                lines.append(f"<b>PnL:</b> <code>{pnl_str}</code>")
            else:
                lines.append("<b>Current Price:</b> N/A")

            if max_profit is not None:
                lines.append(f"<b>Max Profit Reached:</b> <code>${max_profit:.4f}</code>")
            if max_loss is not None:
                lines.append(f"<b>Max Loss Reached:</b> <code>${max_loss:.4f}</code>")

            if direction_result:
                lines.append(f"<b>Direction Result:</b> <code>{direction_result}</code>")

            # Time info
            if created_at:
                lines.append(f"<b>Created:</b> <code>{created_at[:19]}</code>")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    delta = expires_at - datetime.now()
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        minutes = int((delta.total_seconds() % 3600) // 60)
                        if hours > 0:
                            lines.append(f"<b>Expires in:</b> <code>{hours}h {minutes}m</code>")
                        else:
                            lines.append(f"<b>Expires in:</b> <code>{minutes}m</code>")
                    else:
                        lines.append("<b>Status:</b> \u23F0 <code>Expired</code>")
                except (ValueError, TypeError):
                    pass

            # Technical indicators at prediction time
            tech_lines = []
            if rsi is not None:
                rsi_desc = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                tech_lines.append(f"RSI: <code>{rsi:.1f}</code> ({rsi_desc})")
            if atr is not None:
                tech_lines.append(f"ATR: <code>{atr:.2f}</code>")
            if bb_pos is not None:
                tech_lines.append(f"BB Pos: <code>{bb_pos:.2f}</code>")

            if tech_lines:
                lines.append("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
                lines.append("\U0001F4CA <b>Technicals at Prediction:</b>")
                lines.extend(tech_lines)

            # Confidence scores
            conf_lines = []
            if ml_conf is not None:
                conf_lines.append(f"ML Confidence: <code>{ml_conf:.1%}</code>")
            if quant_conf is not None:
                conf_lines.append(f"Quant Confidence: <code>{quant_conf:.1%}</code>")
            if vote_score is not None:
                conf_lines.append(f"Vote Score: <code>{vote_score:.3f}</code>")

            if conf_lines:
                lines.append("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
                lines.append("\U0001F9E0 <b>Confidence Scores:</b>")
                lines.extend(conf_lines)

            # Agent insights summary
            if agent_insights_json:
                try:
                    insights = json.loads(agent_insights_json)
                    insight_lines = []
                    quant = insights.get("quant", {})
                    if quant:
                        insight_lines.append(
                            f"Quant: <code>{quant.get('signal', 'N/A')}</code> ({quant.get('confidence', 0):.1%})"
                        )
                    sentiment = insights.get("sentiment", {})
                    if sentiment:
                        insight_lines.append(
                            f"Sentiment: <code>{sentiment.get('signal', 'N/A')}</code> ({sentiment.get('score', 0):.3f})"
                        )
                    fundamentals = insights.get("fundamentals", {})
                    if fundamentals:
                        insight_lines.append(
                            f"Fundamentals: <code>{fundamentals.get('signal', 'N/A')}</code>"
                        )
                    cio = insights.get("cio", {})
                    if cio:
                        insight_lines.append(
                            f"CIO: <code>{cio.get('signal', 'N/A')}</code>"
                        )
                    risk = insights.get("risk_manager", {})
                    if risk:
                        weights = risk.get("weights", {})
                        w_str = ""
                        if weights:
                            w_str = f" | Weights: Q:{weights.get('quant', 0):.0%} S:{weights.get('sentiment', 0):.0%} F:{weights.get('fundamentals', 0):.0%}"
                        insight_lines.append(
                            f"Risk Mgr: <code>{risk.get('action', 'HOLD')}</code> ({risk.get('vote_score', 0):.3f}){w_str}"
                        )

                    if insight_lines:
                        lines.append("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
                        lines.append("\U0001F916 <b>Agent Insights:</b>")
                        lines.extend(insight_lines)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Reasoning / CIO Memo
            if reasoning:
                lines.append("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
                memo_preview = reasoning[:200] + ("..." if len(reasoning) > 200 else "")
                lines.append(f"\U0001F4DD <b>Reasoning:</b>\n<i>{memo_preview}</i>")

            text = "\n".join(lines)
            await update.message.reply_text(text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error in /prediction_detail: {e}")
            await update.message.reply_text("\u274C <b>Failed to fetch tracker details.</b>\nPlease try again later.")

    async def win_rate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show win rate statistics."""
        symbol = None
        if context.args:
            symbol = " ".join(context.args).strip().upper()

        try:
            stats = self.db.get_tracker_stats(symbol, time_range_days=30)

            if stats["total_predictions"] == 0:
                msg = f"\U0001F4C8 <b>No prediction data found.</b>"
                if symbol:
                    msg += f"\nNo trackers for {symbol}."
                else:
                    msg += "\nNo prediction trackers exist yet."
                await update.message.reply_text(msg)
                return

            total = stats["total_predictions"]
            active = stats["active_count"]
            completed = stats["completed_count"]
            tp_rate = stats["tp_win_rate"]
            dir_acc = stats["directional_accuracy"]
            avg_pnl = stats["avg_pnl_at_outcome"]
            avg_time = stats["avg_time_to_outcome_hours"]

            tp_hits = stats["tp_hits"]
            dir_wins = stats["directional_wins"]

            lines = [
                "\U0001F4CA <b>Win Rate Statistics</b> (Last 30 Days)",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            ]

            if symbol:
                lines.append(f"<b>Symbol:</b> <code>{symbol}</code>")

            lines.append("")
            lines.append("<b>Overall:</b>")
            lines.append(f"Total Predictions: <code>{total}</code>")
            lines.append(f"Active: <code>{active}</code>")
            lines.append(f"Completed: <code>{completed}</code>")
            lines.append("")

            if completed > 0:
                lines.append(f"TP Hit Rate: <code>{tp_rate:.0f}%</code> ({tp_hits}/{completed})")
            if stats.get("directional_accuracy") is not None:
                lines.append(f"Directional Accuracy: <code>{dir_acc:.0f}%</code> ({dir_wins} wins)")

            if avg_pnl is not None:
                pnl_sign = "+" if avg_pnl >= 0 else ""
                lines.append(f"Avg P&L at Outcome: <code>{pnl_sign}{avg_pnl:.1f}%</code>")
            if avg_time is not None:
                lines.append(f"Avg Time to Outcome: <code>{avg_time:.1f} hours</code>")

            text = "\n".join(lines)
            await update.message.reply_text(text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error in /win_rate: {e}")
            await update.message.reply_text("\u274C <b>Failed to fetch win rate statistics.</b>\nPlease try again later.")

    async def trackers_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show tracker dashboard summary."""
        try:
            overall_stats = self.db.get_tracker_stats()
            active_df = self.db.get_active_trackers()
            active_count = len(active_df) if not active_df.empty else 0

            # This week stats (7 days)
            week_stats = self.db.get_tracker_stats(time_range_days=7)
            week_completed = week_stats["completed_count"]
            week_tp = week_stats["tp_hits"]
            week_dir_wins = week_stats["directional_wins"]
            week_losses = week_completed - week_tp - week_dir_wins
            week_win_rate = 0
            if week_completed > 0:
                week_win_rate = ((week_tp + week_dir_wins) / week_completed) * 100

            # Per-symbol breakdown for last 7 days
            try:
                cursor = self.db.conn.cursor()
                cursor.execute('''
                    SELECT symbol,
                           COUNT(*) as total,
                           SUM(CASE WHEN status IN ('WON_TP', 'WON_DIRECTION') THEN 1 ELSE 0 END) as wins
                    FROM prediction_tracker
                    WHERE created_at >= datetime('now', '-7 days')
                      AND status IN ('WON_TP', 'WON_DIRECTION', 'LOST_SL', 'EXPIRED')
                    GROUP BY symbol
                    ORDER BY wins * 1.0 / NULLIF(COUNT(*), 0) DESC
                    LIMIT 10
                ''')
                symbol_rows = cursor.fetchall()
                symbol_stats_list = []
                for row in symbol_rows:
                    sym = row["symbol"]
                    sym_total = row["total"]
                    sym_wins = row["wins"]
                    sym_rate = (sym_wins / sym_total * 100) if sym_total > 0 else 0
                    symbol_stats_list.append((sym, sym_rate, sym_wins, sym_total))
            except Exception as e:
                logger.error(f"Error fetching per-symbol stats: {e}")
                symbol_stats_list = []

            # Recent outcomes
            try:
                cursor.execute('''
                    SELECT id, symbol, status, pnl_percent
                    FROM prediction_tracker
                    WHERE status IN ('WON_TP', 'WON_DIRECTION', 'LOST_SL', 'EXPIRED')
                    ORDER BY COALESCE(tp_hit_at, sl_hit_at, direction_check_at, created_at) DESC
                    LIMIT 5
                ''')
                recent_rows = cursor.fetchall()
                recent_lines = []
                for row in recent_rows:
                    tid = row["id"]
                    sym = row["symbol"]
                    status = row["status"]
                    pnl = row["pnl_percent"]
                    if status in ("WON_TP", "WON_DIRECTION"):
                        emoji = "\u2705"
                        status_label = status.replace("_", "_")
                    else:
                        emoji = "\u274C"
                        status_label = status.replace("_", "_")
                    pnl_str = f" {pnl:+.1f}%" if pnl is not None else ""
                    recent_lines.append(f"{emoji} #{tid} {sym} <code>{status_label}</code>{pnl_str}")
            except Exception as e:
                logger.error(f"Error fetching recent outcomes: {e}")
                recent_lines = ["<i>No recent outcomes available.</i>"]

            # Build dashboard text
            medals = ["\U0001F947", "\U0001F948", "\U0001F949"]
            top_lines = []
            for i, (sym, rate, wins, total) in enumerate(symbol_stats_list[:3]):
                medal = medals[i] if i < 3 else "  "
                top_lines.append(f"{medal} {sym}: <code>{rate:.0f}%</code> ({wins}/{total})")

            if not top_lines:
                top_lines.append("<i>Insufficient data for rankings.</i>")

            lines = [
                "\U0001F4C8 <b>Prediction Tracker Dashboard</b>",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                f"Active Predictions: <code>{active_count}</code>",
                f"Won This Week: <code>{week_tp + week_dir_wins}</code>",
                f"Lost This Week: <code>{max(0, week_losses)}</code>",
                f"Win Rate (7d): <code>{week_win_rate:.0f}%</code>",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                "<b>Top Performers (7d):</b>",
            ]
            lines.extend(top_lines)

            lines.append("")
            lines.append("<b>Recent Outcomes:</b>")
            lines.extend(recent_lines)

            text = "\n".join(lines)
            await update.message.reply_text(text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error in /trackers_summary: {e}")
            await update.message.reply_text("\u274C <b>Failed to fetch dashboard summary.</b>\nPlease try again later.")

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display help message."""
        query = update.callback_query
        if query:
            await query.answer()

        text = (
            "\U0001F916 <b>ENSOTRADE Command Reference</b>\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            "<b>General Operations</b>\n"
            "\u2022 /start - Access insights command center\n"
            "\u2022 /status - System status & metrics\n"
            "\u2022 /predictions - Prediction accuracy & metrics\n"
            "\u2022 /signals - Recent prediction signals\n\n"
            "<b>Market Intelligence</b>\n"
            "\u2022 /analyze <code>&lt;sym&gt;</code> - Deep CIO deliberation\n"
            "\u2022 /predict - ML-driven market forecasting\n"
            "\u2022 /tp - Automated Train-to-Predict pipeline\n"
            "\u2022 /monitor - Set periodic intelligence alerts\n\n"
            "<b>Prediction Tracking</b>\n"
            "\u2022 /my_predictions [<code>&lt;sym&gt;</code>] - View active predictions\n"
            "\u2022 /check <code>&lt;sym&gt;</code> - Live price vs prediction\n"
            "\u2022 /prediction_detail <code>&lt;id&gt;</code> - Full tracker details\n"
            "\u2022 /win_rate [<code>&lt;sym&gt;</code>] - Win rate statistics\n"
            "\u2022 /trackers_summary - Dashboard overview\n\n"
            "<b>Fund Management</b>\n"
            "\u2022 /monitors - List your active monitors\n"
            "\u2022 /stopmonitor <code>&lt;sym&gt;</code> - End surveillance\n"
            "\u2022 /config - Adjust risk & reward params\n"
            "\u2022 /retrain - Force immediate AI recalibration\n"
            "\u2022 /dashboard - Launch Streamlit Analytics\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            "<i>Interact with the bot using natural language for AI chat assistance.</i>"
        )
        keyboard = [[InlineKeyboardButton("\u2B05\uFE0F Back", callback_data='start')]]
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

    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).request(request).build()

    # Commands
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("status", bot.status_cmd))
    app.add_handler(CommandHandler("predictions", bot.predictions_cmd))
    app.add_handler(CommandHandler("signals", bot.recent_signals_cmd))
    app.add_handler(CommandHandler("analyze", bot.analyze_cmd))
    app.add_handler(CommandHandler("retrain", bot.retrain_cmd))
    app.add_handler(CommandHandler("settings", bot.settings_cmd))
    app.add_handler(CommandHandler("train", bot.train_menu))
    app.add_handler(CommandHandler("predict", bot.predict_menu))
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

    # New tracker commands
    app.add_handler(CommandHandler("my_predictions", bot.my_predictions))
    app.add_handler(CommandHandler("check", bot.check_prediction))
    app.add_handler(CommandHandler("prediction_detail", bot.prediction_detail))
    app.add_handler(CommandHandler("win_rate", bot.win_rate))
    app.add_handler(CommandHandler("trackers_summary", bot.trackers_summary))

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

    print("\U0001F916 Telegram Bot is running (Polling)...")
    app.run_polling(bootstrap_retries=5)


if __name__ == "__main__":
    main()
