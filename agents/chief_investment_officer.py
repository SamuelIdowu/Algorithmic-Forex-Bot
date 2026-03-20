"""
Chief Investment Officer (CIO) — Phase 10
The "C-Suite" LLM agent that synthesizes quantitative and qualitative signals.

It reads reasoning strings from:
  - QuantAnalyst (ML predictions & technicals)
  - SentimentAnalyst (News & headlines)
  
And uses an LLM (Groq/Llama-3) to:
  1. Detect contradictions (e.g. Bullish tech, Bearish news).
  2. Formulate a logical "Trade Memo".
  3. Output a final cio_signal and confidence.

Priority = 5 (Runs after all analysts, before RiskManager).
"""
import logging
import json
from agents.base_agent import BaseAgent
from utils.config import GROQ_MODEL_8B

logger = logging.getLogger(__name__)

class ChiefInvestmentOfficer(BaseAgent):
    name = "ChiefInvestmentOfficer"
    role = "manager"
    priority = 5  # Runs after analysts (2 and 3), before RiskManager (6)

    def __init__(self):
        super().__init__()

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        cio_results = {}

        for symbol in symbols:
            try:
                result = self._deliberate(symbol, context)
                cio_results[symbol] = result
                logger.info(f"[CIO] {symbol}: {result['cio_signal']} - {result['memo'][:60]}...")
            except Exception as exc:
                logger.error(f"[CIO] Error for {symbol}: {exc}")
                cio_results[symbol] = {
                    "cio_signal": "NEUTRAL",
                    "cio_confidence": 0.0,
                    "memo": f"Failed to deliberate: {exc}"
                }

        context["cio"] = cio_results
        return context

    def _deliberate(self, symbol: str, context: dict) -> dict:
        """Use LLM to synthesize data into a trade memo."""
        quant = context.get("quant", {}).get(symbol, {})
        sentiment = context.get("sentiment", {}).get(symbol, {})
        
        q_reason = quant.get("reasoning", "No technical reasoning provided.")
        s_reason = sentiment.get("reasoning", "No sentiment reasoning provided.")
        
        # Build prompt for the CIO
        prompt = (
            f"You are the Chief Investment Officer of an AI Hedge Fund. "
            f"Review the following reports for {symbol} and decide on a final signal.\n\n"
            f"QUANT REPORT:\n{q_reason}\n\n"
            f"SENTIMENT REPORT:\n{s_reason}\n\n"
            f"Provide your response as ONLY a JSON object with this structure:\n"
            f"{{\"cio_signal\": \"BULLISH\"|\"BEARISH\"|\"NEUTRAL\", "
            f"\"cio_confidence\": <float 0-1>, "
            f"\"memo\": \"<1-2 sentence logical justification>\"}}"
        )

        client = self.llm_client
        if not client:
            return {
                "cio_signal": "NEUTRAL",
                "cio_confidence": 0.0,
                "memo": "LLM client not configured. Defaulting to Neutral."
            }

        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL_8B,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            raw = chat.choices[0].message.content.strip()
            
            # Basic JSON cleanup in case model adds fluff
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            
            parsed = json.loads(raw)
            return {
                "cio_signal": parsed.get("cio_signal", "NEUTRAL"),
                "cio_confidence": float(parsed.get("cio_confidence", 0.0)),
                "memo": parsed.get("memo", "No memo provided.")
            }
        except Exception as e:
            logger.warning(f"[CIO] LLM call failed: {e}")
            return {
                "cio_signal": "NEUTRAL",
                "cio_confidence": 0.0,
                "memo": f"LLM reasoning failed: {e}"
            }
