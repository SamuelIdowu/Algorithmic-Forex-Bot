#!/bin/bash

# Start the AI Trading Agent in the background
echo "🚀 Starting AI Trading Agent..."
uv run python run_agent.py &

# Start the Web Server (FastAPI + Telegram Bot)
echo "🌐 Starting EnsoTrade Web Server on port ${PORT:-8000}..."
uv run uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8000}
