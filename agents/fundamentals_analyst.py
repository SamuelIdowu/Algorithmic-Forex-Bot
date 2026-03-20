"""
Fundamentals Analyst — Phase 5

Determines asset class from the symbol format, then calls the appropriate
data source:
  • Crypto  (e.g. BTC-USD)  → CoinGecko /simple/price (no API key)
  • Forex   (e.g. EURUSD=X) → returns NEUTRAL (no reliable free fundamental)
  • Stocks  (e.g. AAPL)     → yfinance.Ticker.info (P/E, earnings growth)
  • Futures (e.g. GC=F)     → CoinGecko for gold equivalent, else NEUTRAL

Always returns NEUTRAL on error — never crashes the loop.

Output in context["fundamentals"][symbol]:
    {
        "fundamentals_signal": "BULLISH" | "BEARISH" | "NEUTRAL",
        "reason":              str,
        "asset_class":         "crypto" | "forex" | "stock" | "futures",
    }
"""
import logging
import requests
import yfinance as yf

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# CoinGecko id mapping for crypto symbols
_COINGECKO_ID: dict[str, str] = {
    "BTC-USD":   "bitcoin",
    "ETH-USD":   "ethereum",
    "BNB-USD":   "binancecoin",
    "SOL-USD":   "solana",
    "XRP-USD":   "ripple",
    "ADA-USD":   "cardano",
}

# Map futures tickers to CoinGecko equivalent (gold commodity)
_FUTURES_TO_CG: dict[str, str] = {
    "GC=F": "gold",   # XAUUSD proxy
}

_NEUTRAL = {
    "fundamentals_signal": "NEUTRAL",
    "reason":              "No fundamental data available",
}


def _classify_symbol(symbol: str) -> str:
    """Return asset class string for a symbol."""
    s = symbol.upper()
    if s.endswith("=X"):
        return "forex"
    if "-USD" in s or "-BTC" in s or "-ETH" in s:
        return "crypto"
    if s.endswith("=F") or s.endswith("=E"):
        return "futures"
    # Anything else is treated as stock
    return "stock"


def _analyse_crypto(symbol: str) -> dict:
    """CoinGecko fundamentals for crypto assets."""
    cg_id = _COINGECKO_ID.get(symbol, "")
    if not cg_id:
        return {**_NEUTRAL, "reason": f"No CoinGecko mapping for {symbol}"}

    url = (
        f"https://api.coingecko.com/api/v3/coins/{cg_id}"
        "?localization=false&tickers=false&market_data=true"
        "&community_data=false&developer_data=false"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        market = data.get("market_data", {})

        price_change_7d  = market.get("price_change_percentage_7d", 0) or 0
        market_cap_rank  = data.get("market_cap_rank", 999)
        dominance_signal = "BULLISH" if market_cap_rank <= 3 else "NEUTRAL"

        if price_change_7d > 5 and dominance_signal == "BULLISH":
            signal = "BULLISH"
            reason = f"7d change +{price_change_7d:.1f}%, market cap rank #{market_cap_rank}"
        elif price_change_7d < -5:
            signal = "BEARISH"
            reason = f"7d change {price_change_7d:.1f}%, selling pressure"
        else:
            signal = "NEUTRAL"
            reason = f"7d change {price_change_7d:.1f}%, no strong fundamental signal"

        return {"fundamentals_signal": signal, "reason": reason, "asset_class": "crypto"}

    except Exception as exc:
        logger.warning(f"[FundamentalsAnalyst] CoinGecko error for {symbol}: {exc}")
        return {**_NEUTRAL, "asset_class": "crypto"}


def _analyse_futures(symbol: str) -> dict:
    """Lightweight fundamentals for futures (gold uses CoinGecko proxy)."""
    cg_id = _FUTURES_TO_CG.get(symbol)
    if not cg_id:
        return {**_NEUTRAL, "asset_class": "futures"}

    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={cg_id}&vs_currencies=usd&include_24hr_change=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get(cg_id, {})
        change_24h = data.get("usd_24h_change", 0) or 0

        if change_24h > 1.0:
            signal, reason = "BULLISH", f"Gold +{change_24h:.2f}% in 24h (risk-off demand)"
        elif change_24h < -1.0:
            signal, reason = "BEARISH", f"Gold {change_24h:.2f}% in 24h (risk-on environment)"
        else:
            signal, reason = "NEUTRAL", f"Gold flat ({change_24h:.2f}% 24h change)"

        return {"fundamentals_signal": signal, "reason": reason, "asset_class": "futures"}

    except Exception as exc:
        logger.warning(f"[FundamentalsAnalyst] Futures fundamentals error for {symbol}: {exc}")
        return {**_NEUTRAL, "asset_class": "futures"}


def _analyse_stock(symbol: str) -> dict:
    """yfinance-based fundamentals for equities."""
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}

        pe_ratio        = info.get("trailingPE")
        earnings_growth = info.get("earningsQuarterlyGrowth")
        short_ratio     = info.get("shortRatio", 0) or 0

        signals: list[int] = []   # +1 = bullish, -1 = bearish

        reasons: list[str] = []

        if pe_ratio is not None:
            if pe_ratio < 20:
                signals.append(1)
                reasons.append(f"P/E {pe_ratio:.1f} (undervalued)")
            elif pe_ratio > 40:
                signals.append(-1)
                reasons.append(f"P/E {pe_ratio:.1f} (overvalued)")

        if earnings_growth is not None:
            if earnings_growth > 0.10:
                signals.append(1)
                reasons.append(f"Earnings growth +{earnings_growth*100:.1f}%")
            elif earnings_growth < -0.10:
                signals.append(-1)
                reasons.append(f"Earnings decline {earnings_growth*100:.1f}%")

        if short_ratio > 5:
            signals.append(-1)
            reasons.append(f"High short ratio {short_ratio:.1f}")

        if not signals:
            return {"fundamentals_signal": "NEUTRAL", "reason": "Insufficient fundamental data", "asset_class": "stock"}

        score = sum(signals)
        if score >= 1:
            final = "BULLISH"
        elif score <= -1:
            final = "BEARISH"
        else:
            final = "NEUTRAL"

        return {
            "fundamentals_signal": final,
            "reason":              "; ".join(reasons),
            "asset_class":         "stock",
        }

    except Exception as exc:
        logger.warning(f"[FundamentalsAnalyst] yfinance error for {symbol}: {exc}")
        return {**_NEUTRAL, "asset_class": "stock"}


class FundamentalsAnalyst(BaseAgent):
    """
    Fundamental analysis agent. Auto-detects asset class and applies
    the appropriate analytical framework.
    """

    name = "FundamentalsAnalyst"
    role = "analyst"
    priority = 4   # After SentimentAnalyst, before RiskManager

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        fundamentals_results: dict = {}

        for symbol in symbols:
            try:
                asset_class = _classify_symbol(symbol)
                if asset_class == "crypto":
                    result = _analyse_crypto(symbol)
                elif asset_class == "futures":
                    result = _analyse_futures(symbol)
                elif asset_class == "forex":
                    result = {**_NEUTRAL, "asset_class": "forex",
                              "reason": "Forex fundamentals not implemented (returned NEUTRAL)"}
                else:
                    result = _analyse_stock(symbol)

                fundamentals_results[symbol] = result
                logger.info(
                    f"[FundamentalsAnalyst] {symbol} [{asset_class}]: "
                    f"{result['fundamentals_signal']} — {result['reason']}"
                )

            except Exception as exc:
                logger.error(f"[FundamentalsAnalyst] Error for {symbol}: {exc}", exc_info=True)
                fundamentals_results[symbol] = {**_NEUTRAL, "asset_class": "unknown"}

        context["fundamentals"] = fundamentals_results
        return context
