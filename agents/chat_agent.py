"""
Chat Agent — Phase 11
A utility agent for direct user interaction through the dashboard.

It doesn't participate in the trade cycle directly but provides 
on-demand consultation using LLM (Groq) and data from the database.
"""
import logging
import json
from agents.base_agent import BaseAgent
from data.db_manager import DatabaseManager
from utils.config import GROQ_MODEL_70B

logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    name = "ChatAgent"
    role = "assistant"
    priority = 10  # Low priority, utility agent

    def __init__(self):
        super().__init__()
        self._db = DatabaseManager()

    def _extract_symbol_from_message(self, message: str) -> str:
        """
        Try to extract a symbol from user message.
        Checks for known symbols: EUR/USD, GBP/USD, XAU/USD, etc.
        Returns None if not found.
        """
        known_symbols = [
            "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD",
            "USD/CAD", "NZD/USD", "EUR/GBP", "GC=F", "AAPL", "TSLA", "BTC", "ETH"
        ]
        msg_upper = message.upper()
        for sym in known_symbols:
            if sym in msg_upper or sym.replace("/", "") in msg_upper:
                return sym
        return None

    def _fetch_tracker_context(self, symbol: str = None) -> str:
        """
        Fetch active prediction trackers and format as text context.
        Optionally filtered by symbol.
        """
        trackers = self._db.get_active_trackers(symbol)

        if not trackers or trackers.empty:
            return "No active prediction trackers."

        lines = []
        for _, row in trackers.iterrows():
            tracker_info = (
                f"Tracker #{row['id']} - {row['symbol']}\n"
                f"  Direction: {row['direction']} | Status: {row['status']}\n"
                f"  Entry: {row['entry_price']} | Current: {row['current_price']} ({row['pnl_percent']}%)\n"
                f"  TP: {row['take_profit']} | SL: {row['stop_loss']}\n"
                f"  Confidence: {row['ml_confidence']:.1%} | Timeframe: {row['timeframe']}\n"
                f"  Expires: {row['expires_at']}"
            )
            lines.append(tracker_info)

        return "\n\n".join(lines)

    def _fetch_tracker_stats_context(self) -> str:
        """
        Fetch win rate statistics and format as text context.
        """
        stats = self._db.get_tracker_stats()

        if not stats:
            return "No prediction tracker statistics available."

        stats_text = (
            "PREDICTION TRACKER STATISTICS:\n"
            f"  Total Predictions: {stats['total']}\n"
            f"  Active: {stats['active']}\n"
            f"  Completed: {stats['completed']}\n"
            f"  TP Hit Rate: {stats['tp_hit_rate']:.1%}\n"
            f"  Directional Accuracy: {stats['dir_accuracy']:.1%}\n"
            f"  Avg P&L: {stats['avg_pnl']:.2%}"
        )
        return stats_text

    def ask(self, user_message: str, history: list[dict] = None) -> str:
        """
        Answer a user query with context from the trading logs and prediction trackers.
        """
        # 1. Fetch context from DB
        recent_trades = self._db.get_all_trades(limit=5)

        # 2. Format context for LLM
        context_str = "RECENT TRADE HISTORY:\n"
        if recent_trades.empty:
            context_str += "No trade history yet.\n"
        else:
            context_str += recent_trades[["timestamp", "symbol", "action", "price", "cio_memo"]].to_string(index=False) + "\n"

        # 3. Detect if user is asking about predictions/trackers
        user_lower = user_message.lower()
        tracker_keywords = [
            "prediction", "tracker", "active", "win rate", "performance",
            "how is", "how are", "my prediction", "check", "status"
        ]

        if any(keyword in user_lower for keyword in tracker_keywords):
            # Fetch tracker context
            if any(word in user_lower for word in ["win rate", "statistics", "stats", "performance", "overall"]):
                context_str += "\n" + self._fetch_tracker_stats_context()
            else:
                # Extract symbol if mentioned
                symbol = self._extract_symbol_from_message(user_message)
                context_str += "\nACTIVE PREDICTION TRACKERS:\n" + self._fetch_tracker_context(symbol)

        # 4. Build Prompt
        system_prompt = (
            "You are the Lead Assistant at an AI Hedge Fund. You are talking to the Fund Manager. "
            "Use the provided context to answer their questions professionally and concisely. "
            "You have access to trade history and active prediction tracker data. "
            "If the user asks about predictions, use the tracker context provided. "
            "If you don't know something based on the data, say so. "
            "Focus on logic, risk, and the data provided."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {user_message}"
        })

        client = self.llm_client
        if not client:
            return "I apologize, but my LLM connection (Groq) is not configured. I cannot consult without my brain!"

        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL_70B,
                messages=messages,
                max_tokens=500,
                temperature=0.5,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[ChatAgent] consultation failed: {e}")
            return f"I encountered an error while consulting: {e}"

    def run(self, context: dict) -> dict:
        # This agent doesn't run in the main loop cycle
        return context
