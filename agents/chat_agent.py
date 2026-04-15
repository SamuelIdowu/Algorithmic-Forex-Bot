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

    def ask(self, user_message: str, history: list[dict] = None) -> str:
        """
        Answer a user query with context from the trading logs.
        """
        # 1. Fetch context from DB
        recent_trades = self._db.get_all_trades(limit=5)

        # 2. Format context for LLM
        context_str = "RECENT TRADE HISTORY:\n"
        if recent_trades.empty:
            context_str += "No trade history yet.\n"
        else:
            context_str += recent_trades[["timestamp", "symbol", "action", "price", "cio_memo"]].to_string(index=False) + "\n"

        # 3. Build Prompt
        system_prompt = (
            "You are the Lead Assistant at an AI Hedge Fund. You are talking to the Fund Manager. "
            "Use the provided context to answer their questions professionally and concisely. "
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
