"""
alerts.py
---------
Notification dispatch module for Telegram Bot API & Discord Webhooks.
Allows CurrencyKartz to push real-time BUY/SELL alerts directly to user devices.
"""

import requests
from datetime import datetime


def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> dict:
    """
    Sends a message via Telegram Bot API (100% free).

    Required:
      bot_token: Telegram Bot Token (from @BotFather)
      chat_id:   Chat ID or Channel ID
      message:   Text message (Markdown supported)
    """
    if not bot_token or not chat_id:
        return {"success": False, "error": "Bot token and Chat ID are required."}

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return {"success": True, "response": res.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_discord_alert(webhook_url: str, message: str, username: str = "CurrencyKartz Bot") -> dict:
    """
    Sends a notification via Discord Webhook (100% free).

    Required:
      webhook_url: Discord Webhook URL
      message:     Message content
    """
    if not webhook_url:
        return {"success": False, "error": "Discord Webhook URL is required."}

    payload = {
        "username": username,
        "content": message,
    }
    try:
        res = requests.post(webhook_url, json=payload, timeout=10)
        res.raise_for_status()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def format_signal_alert(symbol: str, signal: str, price: float, strategy: str = "MA Crossover", timeframe: str = "1h") -> str:
    """Formats a neat Markdown alert message for Telegram/Discord."""
    emoji = "🟢 BUY" if signal == "BUY" else ("🔴 SELL" if signal == "SELL" else "⚪ HOLD")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"⚡ *CurrencyKartz Trading Alert*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• *Symbol:* `{symbol}`\n"
        f"• *Signal:* {emoji}\n"
        f"• *Price:* `${price:,.4f}`\n"
        f"• *Strategy:* `{strategy}`\n"
        f"• *Timeframe:* `{timeframe}`\n"
        f"• *Time:* `{timestamp}`\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
