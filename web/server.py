import os
import sys
import json
import logging
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add current directory to path so we can import models/agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db_manager import DatabaseManager
from utils.config import AGENT_SYMBOLS, ALL_SUPPORTED_SYMBOLS, AGENT_PAIR_GROUPS
import telegram_bot

import yfinance as yf
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Telegram Integration ---
tg_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app
    logger.info("Starting up EnsoTrade Web Server...")
    
    # Initialize Telegram Application
    tg_app, _ = telegram_bot.setup_application()
    if tg_app:
        await tg_app.initialize()
        await tg_app.start()
        
        # Set Webhook if URL provided
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            logger.info(f"Setting Telegram Webhook to: {webhook_url}")
            # Ensure URL ends correctly
            if not webhook_url.endswith("/webhook"):
                webhook_url = webhook_url.rstrip("/") + "/webhook"
            await tg_app.bot.set_webhook(url=webhook_url)
            
        logger.info("Telegram component initialized.")
    
    # Start Agent background loop if requested
    # Note: On Render Free tier, this will only run when service is awake.
    mode = "insights"
    if mode in ("paper", "live", "insights"):
        logger.info(f"Agent running in {mode} mode (Background).")
        # In a real setup, you might start the agent loop here as a background task
    
    yield
    
    # Shutdown
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()
    logger.info("Shutdown complete.")

app = FastAPI(title="ENSOTRADE Insights Portal", lifespan=lifespan)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Manager
db = DatabaseManager()

# --- Models ---
# ... (existing models ConsensusSignal, ReliabilityScore, Mover, TopMovers)
class ConsensusSignal(BaseModel):
    symbol: str
    category: str
    quant: str
    sentiment: str
    fundamentals: str
    consensus: str
    confidence: float
    timestamp: str

class ReliabilityScore(BaseModel):
    symbol: str
    analyst: str
    accuracy: float
    weight: float

class Mover(BaseModel):
    symbol: str
    change: float

class TopMovers(BaseModel):
    gainers: List[Mover]
    losers: List[Mover]

# --- WebSocket Manager ---
# ... (existing ConnectionManager)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)


    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Handle incoming Telegram updates via Webhook."""
    if not tg_app:
        return {"status": "Telegram app not initialized"}
    
    from telegram import Update
    try:
        data = await request.json()
        update = Update.de_json(data, tg_app.bot)
        await tg_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/consensus", response_model=List[ConsensusSignal])
async def get_consensus():
# ... (rest of get_consensus remains same)
    """Fetch the latest consensus signals for all monitored symbols."""
    # Combine symbols from database, config defaults, and master CLI groups
    db_symbols = set(db.get_available_symbols())
    config_symbols = set(AGENT_SYMBOLS)
    master_symbols = set(ALL_SUPPORTED_SYMBOLS)
    
    # Union of all unique symbols to ensure full coverage
    symbols = sorted(list(db_symbols | config_symbols | master_symbols))
    
    results = []
    
    # Helper to find category
    def find_category(sym: str):
        for group_label, pairs in AGENT_PAIR_GROUPS:
            # Clean group labels (remove decorative marks)
            clean_label = group_label.replace("── ", "").split(" (")[0].strip()
            for _, ticker in pairs:
                if ticker == sym:
                    return clean_label
        return "Miscellaneous"

    for symbol in symbols:
        category = find_category(symbol)
        # Get the latest analyst cycle for this symbol (even if HOLD)
        trades = db.get_latest_consensus(symbol)
        if not trades.empty:
            row = trades.iloc[0]
            results.append(ConsensusSignal(
                symbol=symbol,
                category=category,
                quant=row.get("quant_signal", "NEUTRAL"),
                sentiment=row.get("sentiment_signal", "NEUTRAL"),
                fundamentals=row.get("fundamentals_signal", "NEUTRAL"),
                consensus=row.get("action", "HOLD"),
                confidence=row.get("vote_score", 0.0),
                timestamp=str(row.get("timestamp", datetime.now()))
            ))
        else:
            # Fallback if no trade data yet for this specific symbol
            results.append(ConsensusSignal(
                symbol=symbol,
                category=category,
                quant="WAITING",
                sentiment="WAITING",
                fundamentals="WAITING",
                consensus="INIT",
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            ))
    return results

@app.get("/api/top_movers", response_model=TopMovers)
async def get_top_movers():
# ... (rest of get_top_movers remains same)
    """Fetch top gainers and losers using yfinance bulk download with short caching."""
    global top_movers_cache
    
    now = datetime.now()
    # Reduced cache for "realtime" feel
    if top_movers_cache["last_fetch"] and (now - top_movers_cache["last_fetch"]) < timedelta(seconds=10):
        return top_movers_cache

    try:
        # Fetch data for all supported symbols in one batch
        symbols = ALL_SUPPORTED_SYMBOLS
        
        # We fetch the last 2 days of data to calculate the current day's change
        # This is MUCH faster than individual .info calls for 50+ tickers
        data = yf.download(
            tickers=symbols,
            period="2d",
            interval="1d",
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        
        movers = []
        for sym in symbols:
            try:
                # Handle both Single ticker (Series) and Multiple tickers (DataFrame)
                if len(symbols) > 1:
                    ticker_data = data[sym]
                else:
                    ticker_data = data
                
                if ticker_data.empty or len(ticker_data) < 2:
                    continue
                
                # Get the last two close prices
                # yfinance returns lowercase columns via our DataProvider, but yf.download returns TitleCase by default
                current_price = ticker_data['Close'].iloc[-1]
                prev_price = ticker_data['Close'].iloc[-2]
                
                if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
                    continue
                    
                change_pct = ((current_price / prev_price) - 1) * 100
                movers.append({"symbol": sym, "change": float(change_pct)})
            except Exception:
                continue
        
        if not movers:
            return top_movers_cache
 
        # Sort by change
        sorted_movers = sorted(movers, key=lambda x: x["change"], reverse=True)
        
        top_movers_cache = {
            "gainers": sorted_movers[:5],
            "losers": sorted_movers[-5:][::-1], # Top 5 losers, most negative first
            "last_fetch": now
        }
        return top_movers_cache
    except Exception as e:
        logger.error(f"Error fetching top movers: {e}")
        return top_movers_cache # Return last known or empty

@app.get("/api/reliability", response_model=List[ReliabilityScore])
async def get_reliability():
    """Fetch analyst reliability metrics (accuracy and current weights)."""
    df = db.get_performance_table()
    results = []
    if not df.empty:
        for _, row in df.iterrows():
            results.append(ReliabilityScore(
                symbol=row["symbol"],
                analyst=row["analyst"],
                accuracy=row["accuracy"] if row["accuracy"] is not None else 0.0,
                weight=row["weight"]
            ))
    return results

@app.get("/api/news")
async def get_news():
# ... (rest remains same)
    """Fetch recent news headlines and sentiment scores from the database."""
    news = db.get_recent_news(limit=20)
    if not news:
        # Fallback to placeholders only if DB is empty
        return [
            {"title": "Intelligence Terminal starting up...", "source": "System", "sentiment": "neutral", "time": "Just now"},
            {"title": "Waiting for first analyst cycle to fetch news...", "source": "System", "sentiment": "neutral", "time": "Recent"}
        ]
    return news

@app.get("/api/freshness")
async def get_freshness():
# ... (rest remains same)
    """Check how recently each ML model was trained."""
    model_dir = "models"
    results = []
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith("_model.pkl"):
                symbol = f.replace("_model.pkl", "").upper()
                mtime = os.path.getmtime(os.path.join(model_dir, f))
                results.append({
                    "symbol": symbol,
                    "last_trained": datetime.fromtimestamp(mtime).isoformat()
                })
    return results

@app.get("/api/status")
async def get_status():
    """General system status."""
    return {
        "status": "ONLINE",
        "mode": "insights",
        "last_heartbeat": datetime.now().isoformat(),
        "monitored_symbols": AGENT_SYMBOLS
    }

@app.post("/api/analyze/{symbol}")
async def trigger_analysis(symbol: str):
# ... (rest remains same)
    """Trigger a manual one-shot analysis for a symbol via the Intelligence Terminal."""
    symbol = symbol.upper()
    logger.info(f"Triggering manual analysis for {symbol}")
    
    # Broadcast to websocket that analysis is starting
    await manager.broadcast(json.dumps({
        "type": "thought",
        "message": f"Analyzing {symbol}... It may take up to 240 seconds."
    }))
    
    # Run the agent in a non-blocking way (Simulated for this implementation, 
    # normally we'd trigger run_agent.py --once --symbol symbol)
    try:
        # Note: In a real production scenario, we'd use a task queue like Celery.
        # Here we'll just run it as a subprocess to keep it simple.
        cmd = [sys.executable, "run_agent.py", "--mode", "backtest", "--symbols", symbol, "--once"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # We don't await finish here to avoid blocking the API response
        # But we could stream the output to the websocket
        return {"status": "Analysis Triggered", "symbol": symbol}
    except Exception as e:
        logger.error(f"Failed to trigger analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/details/{symbol}")
async def get_symbol_details(symbol: str):
# ... (rest remains same)
    """Fetch comprehensive detail package for a specific symbol."""
    symbol = symbol.upper()
    try:
        # 1. Fetch recent price history (last 100 days)
        df = yf.download(symbol, period="100d", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df[symbol] if symbol in df.columns.levels[0] else df.iloc[:, :6]
        
        history = []
        if not df.empty:
            for idx, row in df.iterrows():
                history.append({
                    "date": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })

        # 2. Fetch Consensus & Signal History
        # We fetch last 50 cycles to show signal history
        query = "SELECT * FROM agent_trades WHERE symbol=? ORDER BY id DESC LIMIT 50"
        trades_history_df = pd.read_sql_query(query, db.conn, params=(symbol,))
        
        latest_consensus = {}
        signal_history = []
        bias = 0.0 # -1 to 1 scale

        if not trades_history_df.empty:
            # Latest for summary
            row = trades_history_df.iloc[0]
            latest_consensus = {
                "action": row.get("action", "HOLD"),
                "confidence": row.get("vote_score", 0.0),
                "quant": row.get("quant_signal", "NEUTRAL"),
                "sentiment": row.get("sentiment_signal", "NEUTRAL"),
                "fundamentals": row.get("fundamentals_signal", "NEUTRAL"),
                "memo": row.get("cio_memo", ""),
                "timestamp": str(row.get("timestamp", ""))
            }
            
            # Map history for chart
            for _, r in trades_history_df.iloc[::-1].iterrows(): # chronological order
                signal_history.append({
                    "timestamp": str(r["timestamp"]),
                    "score": float(r["vote_score"]) if r["vote_score"] is not None else 0.0,
                    "action": r["action"]
                })
            
            # BIAS Calculation (last 20)
            recent_20 = trades_history_df.head(20)
            buys = len(recent_20[recent_20["action"] == "BUY"])
            sells = len(recent_20[recent_20["action"] == "SELL"])
            if len(recent_20) > 0:
                bias = (buys - sells) / len(recent_20)

        # 3. Recent Trades with SETUP
        recent_trades = []
        # Re-fetch or reuse trades_history_df but filter for actual trades (not just consensus)
        actual_trades = trades_history_df[trades_history_df["pnl"].notnull()].head(10)
        for _, row in actual_trades.iterrows():
            recent_trades.append({
                "id": int(row["id"]),
                "timestamp": str(row["timestamp"]),
                "action": row["action"],
                "price": float(row["price"]) if row["price"] else None,
                "pnl": float(row["pnl"]) if row["pnl"] else None,
                "setup": {
                    "quant": row.get("quant_signal"),
                    "sentiment": row.get("sentiment_signal"),
                    "fundamentals": row.get("fundamentals_signal")
                }
            })

        # 4. Symbol reliability
        reliability_df = db.get_performance_table()
        symbol_rel = []
        if not reliability_df.empty:
            filtered_rel = reliability_df[reliability_df["symbol"] == symbol]
            for _, row in filtered_rel.iterrows():
                symbol_rel.append({
                    "analyst": row["analyst"],
                    "accuracy": float(row["accuracy"]) if row["accuracy"] is not None else 0.5,
                    "weight": float(row["weight"])
                })

        # 5. Technical Indicators
        indicators = {}
        if not df.empty and len(df) >= 20:
            from utils.indicators import calculate_rsi, calculate_sma
            close_prices = df["Close"]
            indicators = {
                "rsi": float(calculate_rsi(close_prices).iloc[-1]),
                "sma20": float(calculate_sma(close_prices, 20).iloc[-1])
            }

        return {
            "symbol": symbol,
            "current_price": history[-1]["close"] if history else None,
            "history": history,
            "signal_history": signal_history,
            "consensus": latest_consensus,
            "trades": recent_trades,
            "reliability": symbol_rel,
            "indicators": indicators,
            "bias": bias
        }
    except Exception as e:
        logger.error(f"Error fetching details for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- WebSocket ---

@app.websocket("/ws/intelligence")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive, can handle client commands here too
            data = await websocket.receive_text()
            logger.info(f"Received from WS: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Static Files ---

# Mount static files *after* API routes so they don't override them
# Ensure the directory exists
os.makedirs("web/static", exist_ok=True)
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
