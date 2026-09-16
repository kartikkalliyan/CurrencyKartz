"""
binance_market.py
-----------------
All free, public Binance REST market-data calls live here.
No API key, no authentication, no paid endpoints.

Endpoints used (all 100% free):
  GET /api/v3/ticker/24hr     — 24h price stats
  GET /api/v3/ticker/price    — latest spot price
  GET /api/v3/ticker/bookTicker — best bid/ask
  GET /api/v3/depth           — order book depth
  GET /api/v3/trades          — recent public trades
  GET /api/v3/exchangeInfo    — all active trading pairs
"""

import requests
from datetime import datetime

BASE_URL = "https://api.binance.com/api/v3"
TIMEOUT = 10


def _get(endpoint: str, params: dict = None) -> dict | list:
    """Generic GET helper — raises on HTTP errors."""
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 1. 24-hour ticker stats
# ---------------------------------------------------------------------------

def get_ticker_24h(symbol: str) -> dict:
    """
    Returns 24-hour price change statistics for a symbol.

    Keys of interest in the returned dict:
      symbol, priceChange, priceChangePercent, weightedAvgPrice,
      prevClosePrice, lastPrice, highPrice, lowPrice,
      volume, quoteVolume, openTime, closeTime
    """
    data = _get("ticker/24hr", params={"symbol": symbol.upper()})
    return {
        "symbol":              data["symbol"],
        "last_price":          float(data["lastPrice"]),
        "price_change":        float(data["priceChange"]),
        "price_change_pct":    float(data["priceChangePercent"]),
        "high_24h":            float(data["highPrice"]),
        "low_24h":             float(data["lowPrice"]),
        "volume_24h":          float(data["volume"]),
        "quote_volume_24h":    float(data["quoteVolume"]),
        "weighted_avg_price":  float(data["weightedAvgPrice"]),
        "open_time":           datetime.fromtimestamp(data["openTime"] / 1000),
        "close_time":          datetime.fromtimestamp(data["closeTime"] / 1000),
    }


# ---------------------------------------------------------------------------
# 2. Simple latest price
# ---------------------------------------------------------------------------

def get_latest_price(symbol: str) -> float:
    """Returns the latest trade price for a symbol as a float."""
    data = _get("ticker/price", params={"symbol": symbol.upper()})
    return float(data["price"])


# ---------------------------------------------------------------------------
# 3. Best bid / ask spread
# ---------------------------------------------------------------------------

def get_best_bid_ask(symbol: str) -> dict:
    """
    Returns the best bid price/qty and ask price/qty.

    Keys: symbol, bid_price, bid_qty, ask_price, ask_qty, spread, spread_pct
    """
    data = _get("ticker/bookTicker", params={"symbol": symbol.upper()})
    bid = float(data["bidPrice"])
    ask = float(data["askPrice"])
    spread = ask - bid
    spread_pct = (spread / bid * 100) if bid > 0 else 0.0
    return {
        "symbol":     data["symbol"],
        "bid_price":  bid,
        "bid_qty":    float(data["bidQty"]),
        "ask_price":  ask,
        "ask_qty":    float(data["askQty"]),
        "spread":     round(spread, 8),
        "spread_pct": round(spread_pct, 4),
    }


# ---------------------------------------------------------------------------
# 4. Order book depth
# ---------------------------------------------------------------------------

def get_order_book(symbol: str, limit: int = 10) -> dict:
    """
    Returns the top N bids and asks from the order book.

    limit: 5, 10, 20, 50, 100, 500, 1000 (all free)

    Returns:
      {
        "bids": [{"price": float, "qty": float}, ...],   # highest bid first
        "asks": [{"price": float, "qty": float}, ...],   # lowest ask first
        "last_update_id": int
      }
    """
    data = _get("depth", params={"symbol": symbol.upper(), "limit": limit})
    return {
        "last_update_id": data["lastUpdateId"],
        "bids": [{"price": float(b[0]), "qty": float(b[1])} for b in data["bids"]],
        "asks": [{"price": float(a[0]), "qty": float(a[1])} for a in data["asks"]],
    }


# ---------------------------------------------------------------------------
# 5. Recent public trades
# ---------------------------------------------------------------------------

def get_recent_trades(symbol: str, limit: int = 20) -> list[dict]:
    """
    Returns the most recent public trades (max 1000, free endpoint).

    Each trade dict:
      id, price, qty, quote_qty, time, is_buyer_maker
    """
    raw = _get("trades", params={"symbol": symbol.upper(), "limit": limit})
    return [
        {
            "id":             t["id"],
            "price":          float(t["price"]),
            "qty":            float(t["qty"]),
            "quote_qty":      float(t["quoteQty"]),
            "time":           datetime.fromtimestamp(t["time"] / 1000),
            "is_buyer_maker": t["isBuyerMaker"],
            "side":           "SELL" if t["isBuyerMaker"] else "BUY",
        }
        for t in raw
    ]


# ---------------------------------------------------------------------------
# 6. All active USDT trading pairs
# ---------------------------------------------------------------------------

def get_all_usdt_symbols() -> list[str]:
    """
    Returns a sorted list of all active *USDT spot trading pairs
    from /api/v3/exchangeInfo. No API key needed.

    Example: ['1000BONKUSDT', 'AAVEUSDT', 'ADAUSDT', 'ARBUSDT', ...]
    """
    data = _get("exchangeInfo")
    symbols = [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
        and s["isSpotTradingAllowed"]
    ]
    return sorted(symbols)


# ---------------------------------------------------------------------------
# 7. Convenience: quick market summary for a symbol
# ---------------------------------------------------------------------------

def get_market_summary(symbol: str) -> dict:
    """
    One call that bundles ticker + best bid/ask into a single dict.
    Useful for displaying a full market snapshot panel.
    """
    ticker = get_ticker_24h(symbol)
    spread = get_best_bid_ask(symbol)
    return {**ticker, **{k: v for k, v in spread.items() if k not in ticker}}
