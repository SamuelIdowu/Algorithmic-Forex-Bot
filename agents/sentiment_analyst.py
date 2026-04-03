"""
Sentiment Analyst — Phase 4

Two-tier NLP sentiment engine:
  - Tier 1 (default): VADER (offline, free) — works out of the box
  - Tier 2 (upgrade): Groq API + llama-3 — set SENTIMENT_ENGINE=groq in .env

Headlines are fetched from Alpha Vantage News API (uses existing AV key).
Falls back to NEUTRAL gracefully if any step fails.

Output in context["sentiment"][symbol]:
    {
        "sentiment_signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "sentiment_score":  float (-1 to 1),
        "engine":           "vader" | "groq",
        "headlines_used":   int,
    }
"""
import logging
import os
import requests

from agents.base_agent import BaseAgent
from utils.config import (
    ALPHA_VANTAGE_API_KEY,
    NEWS_API_KEY,
    FINNHUB_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL_8B,
    SENTIMENT_ENGINE,
)

logger = logging.getLogger(__name__)

# Map symbols → news search terms
_SYMBOL_KEYWORDS: dict[str, str] = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "EURUSD=X": "Euro Dollar forex",
    "GC=F":     "Gold commodity",
    "EUR/USD":  "Euro Dollar forex",
    "GBP/USD":  "British Pound forex",
    "USD/JPY":  "Dollar Yen forex",
    "TSLA":     "Tesla",
    "AAPL":     "Apple",
    "NVDA":     "Nvidia",
}


def _fetch_headlines_av(symbol: str, limit: int = 10) -> list[dict]:
    """Fetch headlines from Alpha Vantage News Sentiment endpoint."""
    keyword = _SYMBOL_KEYWORDS.get(symbol, symbol.replace("-", " ").replace("=X", ""))
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&keywords={keyword}"
        f"&limit={limit}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])
        results = []
        for item in feed:
            if item.get("title"):
                results.append({
                    "title":     item.get("title"),
                    "source":    item.get("source", "Alpha Vantage"),
                    "url":       item.get("url", ""),
                    "sentiment": item.get("overall_sentiment_label", "NEUTRAL"),
                })
        return results
    except Exception as exc:
        logger.warning(f"[SentimentAnalyst] AV news fetch failed for {symbol}: {exc}")
        return []


def _fetch_headlines_newsapi(symbol: str, limit: int = 10) -> list[dict]:
    """Fetch headlines from NewsAPI.org (optional upgrade, requires NEWS_API_KEY)."""
    if not NEWS_API_KEY:
        return []
    keyword = _SYMBOL_KEYWORDS.get(symbol, symbol)
    url = (
        f"https://newsapi.org/v2/everything?q={keyword}"
        f"&pageSize={limit}&sortBy=publishedAt"
        f"&apiKey={NEWS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results = []
        for a in articles:
             if a.get("title"):
                 results.append({
                     "title":  a.get("title"),
                     "source": a.get("source", {}).get("name", "NewsAPI"),
                     "url":    a.get("url", ""),
                     "sentiment": "NEUTRAL",  # NewsAPI doesn't provide sentiment
                 })
        return results
    except Exception as exc:
        logger.warning(f"[SentimentAnalyst] NewsAPI fetch failed for {symbol}: {exc}")
        return []


def _fetch_headlines_finnhub(symbol: str, limit: int = 15) -> list[dict]:
    """Fetch headlines from Finnhub (requires FINNHUB_API_KEY)."""
    if not FINNHUB_API_KEY:
        return []
    
    # Finnhub likes clean symbols (e.g. AAPL)
    clean_sym = symbol.split("-")[0].split("/")[0].replace("=X", "").upper()
    
    url = (
        f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        if clean_sym in ("GENERAL", "MARKET") else
        f"https://finnhub.io/api/v1/company-news?symbol={clean_sym}&from=2024-01-01&to=2030-01-01&token={FINNHUB_API_KEY}"
    )
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # If company-news is empty, try 'general' news
        if not data and clean_sym not in ("GENERAL", "MARKET"):
             return _fetch_headlines_finnhub("GENERAL", limit=limit)
             
        results = []
        for item in data[:limit]:
            if item.get("headline"):
                results.append({
                    "title":     item.get("headline"),
                    "source":    item.get("source", "Finnhub"),
                    "url":       item.get("url", ""),
                    "sentiment": "NEUTRAL", # Finnhub doesn't provide sentiment here
                })
        return results
    except Exception as exc:
        logger.warning(f"[SentimentAnalyst] Finnhub fetch failed for {symbol}: {exc}")
        return []


def _score_vader(headlines: list[str]) -> float:
    """Score headlines with VADER. Returns compound score mean (-1 to 1)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.error("[SentimentAnalyst] vaderSentiment not installed. Run: pip install vaderSentiment")
        return 0.0

    if not headlines:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)


def _score_groq(headlines: list[str], symbol: str) -> float:
    """
    Score headlines using Groq API (llama-3).
    Returns a float in [-1, 1] parsed from the model's JSON reply.
    Falls back to 0.0 on any failure.
    """
    if not GROQ_API_KEY:
        logger.warning("[SentimentAnalyst] GROQ_API_KEY not set, falling back to VADER")
        return _score_vader(headlines)

    try:
        from groq import Groq  # pip install groq
        client = Groq(api_key=GROQ_API_KEY)

        combined = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = (
            f"You are a financial sentiment analyst. Analyse the following headlines "
            f"about {symbol} and reply with ONLY a JSON object: "
            f'{{\"score\": <float between -1.0 (very bearish) and 1.0 (very bullish)>}}\n\n'
            f"Headlines:\n{combined}"
        )

        chat = client.chat.completions.create(
            model=GROQ_MODEL_8B,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        raw = chat.choices[0].message.content.strip()

        import json
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        return max(-1.0, min(1.0, score))   # clamp

    except Exception as exc:
        logger.error(f"[SentimentAnalyst] Groq scoring failed: {exc}")
        return _score_vader(headlines)   # graceful fallback


class SentimentAnalyst(BaseAgent):
    """
    NLP sentiment analyst. Fetches news headlines and scores them with
    VADER (default) or Groq llama-3 (upgrade).
    """

    name = "SentimentAnalyst"
    role = "analyst"
    priority = 3      # After QuantAnalyst, before RiskManager

    def __init__(self):
        super().__init__()
        from data.db_manager import DatabaseManager
        self._db = DatabaseManager()

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        engine  = SENTIMENT_ENGINE.lower()
        sentiment_results: dict = {}

        for symbol in symbols:
            try:
                result = self._analyse(symbol, engine)
                sentiment_results[symbol] = result
                logger.info(
                    f"[SentimentAnalyst] {symbol}: {result['sentiment_signal']} "
                    f"(score={result['sentiment_score']:.3f}, engine={result['engine']})"
                )
            except Exception as exc:
                logger.error(f"[SentimentAnalyst] Error for {symbol}: {exc}", exc_info=True)
                sentiment_results[symbol] = self._neutral(engine, reason=str(exc))

        context["sentiment"] = sentiment_results
        return context

    # ──────────────────────────────────────────────────────────────────────────

    def _analyse(self, symbol: str, engine: str) -> dict:
        """Fetch headlines + score for a single symbol."""
        # 1. Aggregate headlines from all sources
        headlines = []
        
        # Alpha Vantage (Keywords)
        av_list = _fetch_headlines_av(symbol, limit=10)
        headlines.extend(av_list)
        
        # NewsAPI (Keywords)
        if len(headlines) < 10:
            headlines.extend(_fetch_headlines_newsapi(symbol, limit=10))
            
        # Finnhub (Symbol-specific)
        if len(headlines) < 15:
            headlines.extend(_fetch_headlines_finnhub(symbol, limit=10))

        # Remove duplicates while preserving order
        seen = set()
        unique_headlines = []
        for h in headlines:
            if h["title"] not in seen:
                seen.add(h["title"])
                unique_headlines.append(h)
        headlines = unique_headlines

        if not headlines:
            logger.warning(f"[SentimentAnalyst] No news found for {symbol} on any source.")
            return self._neutral(engine)

        # 3. Persist headlines to DB for the dashboard
        self._db.save_news(symbol, headlines[:15])

        # 4. Extract just text for scoring
        headline_texts = [h["title"] for h in headlines]

        # Force Groq if key is available and engine isn't explicitly 'vader'
        final_engine = "groq" if (GROQ_API_KEY and engine != "vader") else "vader"

        if final_engine == "groq":
            score = _score_groq(headline_texts, symbol)
        else:
            score = _score_vader(headline_texts)

        # Convert score → signal
        if score > 0.05:
            signal = "BULLISH"
        elif score < -0.05:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # Reasoning: summary of headlines
        reasoning = f"Processed {len(headlines)} headlines. Top: '{headlines[0]['title'][:60]}...'"

        return {
            "sentiment_signal": signal,
            "sentiment_score":  round(score, 4),
            "engine":           engine,
            "headlines_used":   len(headlines),
            "reasoning":        reasoning,
        }

    @staticmethod
    def _neutral(engine: str, reason: str = "No headlines found") -> dict:
        return {
            "sentiment_signal": "NEUTRAL",
            "sentiment_score":  0.0,
            "engine":           engine,
            "headlines_used":   0,
            "reasoning":        reason,
        }
