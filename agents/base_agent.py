"""
Base Agent — Abstract class all hedge fund agents must implement.

To create a new agent:
  1. Create agents/my_analyst.py
  2. Inherit from BaseAgent and implement run(context) -> dict
  3. Done — the registry auto-discovers it.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all hedge fund agents.

    Attributes:
        name (str): Human-readable display name.
        role (str): One of "analyst", "manager", "monitor".
        priority (int): Execution order — lower value runs earlier.
        enabled (bool): Toggle agent on/off without deleting the file.
    """

    name: str = "BaseAgent"
    role: str = "analyst"       # "analyst" | "manager" | "monitor"
    priority: int = 0           # lower = runs earlier
    enabled: bool = True        # set to False to skip this agent
    reasoning: str = ""         # Qualitative explanation of the agent's decision

    def __init__(self):
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy access to LLM client (Groq by default)."""
        if self._llm_client is None:
            try:
                from groq import Groq
                from utils.config import GROQ_API_KEY
                if GROQ_API_KEY:
                    self._llm_client = Groq(api_key=GROQ_API_KEY)
            except ImportError:
                logger.warning(f"[{self.name}] Groq library not found for LLM features.")
        return self._llm_client

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Execute the agent's logic.

        Args:
            context (dict): Shared context dict passed through all agents.
                            Always contains at minimum:
                              - "symbols": list[str]
                              - "mode": str  (backtest | paper | live)

        Returns:
            dict: Updated context with this agent's results merged in.
        """

    def on_trade_result(self, symbol: str, pnl: float) -> None:
        """
        Optional hook called after a trade closes.
        Agents can override this for self-improvement logic.

        Args:
            symbol (str): The traded symbol.
            pnl (float): Realised profit/loss of the closed trade.
        """

    def __repr__(self) -> str:
        status = "✅" if self.enabled else "⏸️"
        return f"{status} [{self.priority}] {self.name} ({self.role})"
