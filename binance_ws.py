"""
binance_ws.py
-------------
Lightweight WebSocket client for free Binance real-time market streams.
No API key. No Binance SDK. Uses only the `websockets` library.

Free streams used:
  wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>
  wss://stream.binance.com:9443/ws/<symbol>@ticker
  wss://stream.binance.com:9443/ws/<symbol>@depth5

Usage (standalone, blocking):
    from binance_ws import stream_live_candle

    def on_candle(candle):
        print(candle)

    stream_live_candle("BTCUSDT", "1m", on_candle)   # runs forever

Usage (background thread):
    import threading
    t = threading.Thread(target=stream_live_candle,
                         args=("BTCUSDT", "1m", on_candle),
                         daemon=True)
    t.start()
"""

import asyncio
import json
import threading
from datetime import datetime

WS_BASE = "wss://stream.binance.com:9443/ws"


# ---------------------------------------------------------------------------
# Internal async helpers
# ---------------------------------------------------------------------------

async def _run_stream(url: str, callback, stop_event: asyncio.Event):
    """
    Core async loop — connects to a WebSocket URL, parses each JSON frame
    and passes it to callback(). Reconnects automatically on disconnect.
    Stops when stop_event is set.
    """
    try:
        import websockets  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "The 'websockets' package is required for WebSocket streaming.\n"
            "Install it with:  pip install websockets"
        )

    while not stop_event.is_set():
        try:
            async with websockets.connect(url) as ws:
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(raw)
                        callback(data)
                    except asyncio.TimeoutError:
                        # send a ping to keep the connection alive
                        await ws.ping()
        except Exception:
            # brief pause before reconnect
            if not stop_event.is_set():
                await asyncio.sleep(3)


def _start_stream(url: str, callback):
    """
    Runs the async stream in a new event loop on the current thread.
    Returns a stop_handle dict with a 'stop()' method.
    """
    loop = asyncio.new_event_loop()
    stop_event = asyncio.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_stream(url, callback, stop_event))
        loop.close()

    def _stop():
        loop.call_soon_threadsafe(stop_event.set)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"thread": thread, "stop": _stop}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_live_candle(symbol: str, interval: str, callback):
    """
    Streams real-time kline (candle) updates for a symbol/interval.

    Each callback invocation receives a dict:
      {
        "event_type":   "kline",
        "event_time":   datetime,
        "symbol":       str,
        "interval":     str,
        "open_time":    datetime,
        "open":         float,
        "high":         float,
        "low":          float,
        "close":        float,
        "volume":       float,
        "is_closed":    bool,   # True only when candle is finalized
        "num_trades":   int,
      }

    Args:
        symbol:   e.g. "BTCUSDT"
        interval: e.g. "1m", "5m", "15m", "1h", "4h", "1d"
        callback: callable(candle_dict) — called on each update

    Returns a handle with a .stop() method to terminate the stream.

    Example:
        handle = stream_live_candle("BTCUSDT", "1m", print)
        # ... later ...
        handle["stop"]()
    """
    url = f"{WS_BASE}/{symbol.lower()}@kline_{interval}"

    def _parse_and_forward(raw: dict):
        k = raw.get("k", {})
        callback({
            "event_type": raw.get("e"),
            "event_time": datetime.fromtimestamp(raw["E"] / 1000),
            "symbol":     raw["s"],
            "interval":   k.get("i"),
            "open_time":  datetime.fromtimestamp(k["t"] / 1000),
            "open":       float(k["o"]),
            "high":       float(k["h"]),
            "low":        float(k["l"]),
            "close":      float(k["c"]),
            "volume":     float(k["v"]),
            "is_closed":  k.get("x", False),
            "num_trades": k.get("n", 0),
        })

    return _start_stream(url, _parse_and_forward)


def stream_ticker(symbol: str, callback):
    """
    Streams real-time 24h rolling ticker statistics.

    Each callback invocation receives a dict:
      {
        "event_time":        datetime,
        "symbol":            str,
        "price_change":      float,
        "price_change_pct":  float,
        "last_price":        float,
        "high_24h":          float,
        "low_24h":           float,
        "volume_24h":        float,
        "quote_volume_24h":  float,
      }

    Returns a handle with a .stop() method.
    """
    url = f"{WS_BASE}/{symbol.lower()}@ticker"

    def _parse_and_forward(raw: dict):
        callback({
            "event_time":       datetime.fromtimestamp(raw["E"] / 1000),
            "symbol":           raw["s"],
            "price_change":     float(raw["p"]),
            "price_change_pct": float(raw["P"]),
            "last_price":       float(raw["c"]),
            "high_24h":         float(raw["h"]),
            "low_24h":          float(raw["l"]),
            "volume_24h":       float(raw["v"]),
            "quote_volume_24h": float(raw["q"]),
        })

    return _start_stream(url, _parse_and_forward)


def stream_order_book(symbol: str, callback, depth: int = 5):
    """
    Streams real-time partial order book depth (top N bids/asks).
    depth: 5, 10, or 20 (all free).

    Each callback invocation receives a dict:
      {
        "symbol": str,
        "bids": [{"price": float, "qty": float}, ...],
        "asks": [{"price": float, "qty": float}, ...],
      }

    Returns a handle with a .stop() method.
    """
    valid_depths = {5, 10, 20}
    if depth not in valid_depths:
        depth = 5

    url = f"{WS_BASE}/{symbol.lower()}@depth{depth}"

    def _parse_and_forward(raw: dict):
        callback({
            "symbol": symbol.upper(),
            "bids": [{"price": float(b[0]), "qty": float(b[1])} for b in raw.get("bids", [])],
            "asks": [{"price": float(a[0]), "qty": float(a[1])} for a in raw.get("asks", [])],
        })

    return _start_stream(url, _parse_and_forward)
