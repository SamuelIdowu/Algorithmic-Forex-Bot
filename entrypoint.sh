#!/bin/bash

# Start the AI Trading Agent in the background
echo "🚀 Starting AI Trading Agent..."
python run_agent.py &

# Start the Telegram Bot in the foreground
echo "🤖 Starting Telegram Bot..."
python telegram_bot.py

# Wait for all background processes to finish (though telegram_bot should keep it alive)
wait
