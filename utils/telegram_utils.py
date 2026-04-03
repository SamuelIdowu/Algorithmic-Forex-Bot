import logging
import httpx
from utils.config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

async def send_telegram_message(text: str):
    """
    Sends a message to the configured Telegram chat.
    Non-blocking/Async.
    """
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

def send_telegram_message_sync(text: str):
    """
    Synchronous version of the Telegram message sender.
    Useful for existing sync agent code.
    """
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        resp = httpx.post(url, json=payload, timeout=10.0)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message (sync): {e}")
