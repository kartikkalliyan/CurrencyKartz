"""
live_bot.py
-----------
Live signal monitor for CurrencyKartz.

Supports two modes:
  - Polling mode  (default): REST API, checks every N minutes (original behaviour)
  - WebSocket mode (--ws flag): real-time kline stream, processes each closed candle

All data comes from Binance free public endpoints — no API key required.

Usage:
    python live_bot.py                          # polling mode, BTCUSDT 1h
    python live_bot.py --ws                     # WebSocket mode, BTCUSDT 1m
    python live_bot.py --symbol ETHUSDT --ws    # WebSocket mode, ETHUSDT 1m
    python live_bot.py --symbol BTCUSDT --symbols ETHUSDT SOLUSDT  # multi-symbol polling
"""

import argparse
import os
import time
import threading
import requests
import pandas as pd
from datetime import datetime

from binance_market import get_ticker_24h
from binance_ws import stream_live_candle
from alerts import send_telegram_alert, send_discord_alert, format_signal_alert

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_LOOKBACK = 200
CHECK_INTERVAL_MINUTES = 15
LOG_FILE = "live_signals_log.csv"


# ---------------------------------------------------------------------------
# Data fetching (free REST)
# ---------------------------------------------------------------------------

def fetch_latest_candles(symbol: str = DEFAULT_SYMBOL,
                          interval: str = DEFAULT_INTERVAL,
                          limit: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"open_time": "timestamp"})
    df = df.iloc[:-1].reset_index(drop=True)   # drop still-forming candle
    return df


# ---------------------------------------------------------------------------
# Signal logic (MA20 / MA50 crossover — no API key, no paid service)
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame):
    df = df.copy()
    df["MA_20"] = df["close"].rolling(window=20).mean()
    df["MA_50"] = df["close"].rolling(window=50).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    golden_cross = (prev["MA_20"] <= prev["MA_50"]) and (latest["MA_20"] > latest["MA_50"])
    death_cross  = (prev["MA_20"] >= prev["MA_50"]) and (latest["MA_20"] < latest["MA_50"])

    if golden_cross:
        signal = "BUY"
    elif death_cross:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, latest


# ---------------------------------------------------------------------------
# Volume context (free 24h ticker — enriches log without any cost)
# ---------------------------------------------------------------------------

def _volume_context(symbol: str) -> str:
    """Returns a short volume/change string from the free 24h ticker."""
    try:
        t = get_ticker_24h(symbol)
        change = t["price_change_pct"]
        vol    = t["quote_volume_24h"]
        sign   = "+" if change >= 0 else ""
        return f"{sign}{change:.2f}% | 24h vol: ${vol:,.0f}"
    except Exception:
        return "volume context unavailable"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_signal(signal: str, row: pd.Series, symbol: str, interval: str):
    file_exists = os.path.isfile(LOG_FILE)
    entry = pd.DataFrame([{
        "checked_at":   datetime.now(),
        "symbol":       symbol,
        "interval":     interval,
        "candle_time":  row["timestamp"],
        "close_price":  row["close"],
        "MA_20":        row["MA_20"],
        "MA_50":        row["MA_50"],
        "signal":       signal,
    }])
    entry.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)


# ---------------------------------------------------------------------------
# Polling mode (original behaviour, now multi-symbol capable)
# ---------------------------------------------------------------------------

def _poll_once(symbol: str, interval: str):
    """Run one polling cycle for a single symbol."""
    df = fetch_latest_candles(symbol, interval)
    signal, latest_row = compute_signal(df)
    log_signal(signal, latest_row, symbol, interval)

    vol_ctx = _volume_context(symbol)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts}] {symbol} ({interval}) | "
        f"candle: {latest_row['timestamp']} | "
        f"price: {latest_row['close']:.4f} | "
        f"MA20: {latest_row['MA_20']:.4f} | MA50: {latest_row['MA_50']:.4f} | "
        f"{vol_ctx} "
        f"-> SUGGESTION: {signal}"
    )

    if signal != "HOLD":
        print(f"  >>> New {signal} signal for {symbol}! Check {LOG_FILE} <<<")


def run_polling(symbols: list[str], interval: str = DEFAULT_INTERVAL):
    print(f"Starting polling monitor | symbols: {', '.join(symbols)} | "
          f"interval: {interval} | checking every {CHECK_INTERVAL_MINUTES} min\n")

    while True:
        for sym in symbols:
            try:
                _poll_once(sym, interval)
            except Exception as e:
                print(f"[{sym}] Error: {e}. Will retry next cycle.")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


# ---------------------------------------------------------------------------
# WebSocket mode — processes only fully closed candles (is_closed=True)
# ---------------------------------------------------------------------------

def run_websocket(symbols: list[str], interval: str = "1m"):
    """
    Streams live kline data for each symbol via free Binance WebSocket.
    Signal is evaluated only when a candle fully closes (is_closed=True).
    """
    print(f"Starting WebSocket monitor | symbols: {', '.join(symbols)} | "
          f"interval: {interval}\nPress Ctrl+C to stop.\n")

    # Keep a small rolling buffer of closed candles per symbol
    buffers: dict[str, list] = {s: [] for s in symbols}
    locks:   dict[str, threading.Lock] = {s: threading.Lock() for s in symbols}
    handles = []

    def make_handler(sym: str):
        def handler(candle: dict):
            if not candle["is_closed"]:
                return   # only act on fully closed candles

            row = {
                "timestamp": candle["open_time"],
                "open":      candle["open"],
                "high":      candle["high"],
                "low":       candle["low"],
                "close":     candle["close"],
                "volume":    candle["volume"],
            }

            with locks[sym]:
                buffers[sym].append(row)
                if len(buffers[sym]) > DEFAULT_LOOKBACK:
                    buffers[sym].pop(0)

                if len(buffers[sym]) < 51:   # need at least 51 rows for MA50
                    return

                df = pd.DataFrame(buffers[sym])
                signal, latest_row = compute_signal(df)

            log_signal(signal, latest_row, sym, interval)
            vol_ctx = _volume_context(sym)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}] WS {sym} ({interval}) candle closed | "
                f"price: {latest_row['close']:.4f} | "
                f"{vol_ctx} -> {signal}"
            )
            if signal != "HOLD":
                print(f"  >>> {signal} signal for {sym}! <<<")

        return handler

    for sym in symbols:
        handle = stream_live_candle(sym, interval, make_handler(sym))
        handles.append(handle)
        print(f"  Streaming {sym}@{interval} via WebSocket...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streams...")
        for h in handles:
            h["stop"]()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CurrencyKartz live signal monitor")
    parser.add_argument("--symbol",  default=DEFAULT_SYMBOL,
                        help="Primary trading symbol (default: BTCUSDT)")
    parser.add_argument("--symbols", nargs="+", default=[],
                        help="Additional symbols to monitor simultaneously")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        help="Kline interval (default: 1h). "
                             "WebSocket mode defaults to 1m for responsiveness.")
    parser.add_argument("--ws", action="store_true",
                        help="Use WebSocket streaming instead of polling")
    parser.add_argument("--telegram-token", default="", help="Telegram Bot Token for alerts")
    parser.add_argument("--telegram-chat", default="", help="Telegram Chat ID for alerts")
    parser.add_argument("--discord-webhook", default="", help="Discord Webhook URL for alerts")
    args = parser.parse_args()

    def dispatch_alert_if_needed(symbol: str, signal: str, price: float, interval: str):
        if signal != "HOLD":
            msg = format_signal_alert(symbol, signal, price, timeframe=interval)
            if args.telegram_token and args.telegram_chat:
                send_telegram_alert(args.telegram_token, args.telegram_chat, msg)
            if args.discord_webhook:
                send_discord_alert(args.discord_webhook, msg)

    all_symbols = list(dict.fromkeys([args.symbol.upper()] +
                                      [s.upper() for s in args.symbols]))

    if args.ws:
        ws_interval = "1m" if args.interval == DEFAULT_INTERVAL else args.interval
        run_websocket(all_symbols, ws_interval)
    else:
        run_polling(all_symbols, args.interval)
        run_polling(all_symbols, args.interval)