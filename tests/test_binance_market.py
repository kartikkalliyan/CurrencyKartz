"""
tests/test_binance_market.py
----------------------------
Unit tests for binance_market.py.

All Binance HTTP calls are mocked — no real network requests are made.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binance_market


# ---------------------------------------------------------------------------
# Fixtures: sample Binance API responses
# ---------------------------------------------------------------------------

TICKER_24H_RESPONSE = {
    "symbol":              "BTCUSDT",
    "priceChange":         "1234.56",
    "priceChangePercent":  "2.35",
    "weightedAvgPrice":    "53500.00",
    "prevClosePrice":      "52500.00",
    "lastPrice":           "53734.56",
    "highPrice":           "54200.00",
    "lowPrice":            "52100.00",
    "volume":              "18500.00",
    "quoteVolume":         "990000000.00",
    "openTime":            1700000000000,
    "closeTime":           1700086400000,
}

BOOK_TICKER_RESPONSE = {
    "symbol":   "BTCUSDT",
    "bidPrice": "53730.00",
    "bidQty":   "0.500",
    "askPrice": "53732.00",
    "askQty":   "0.250",
}

PRICE_RESPONSE = {"symbol": "BTCUSDT", "price": "53734.56"}

DEPTH_RESPONSE = {
    "lastUpdateId": 123456789,
    "bids": [["53730.00", "0.500"], ["53720.00", "1.200"]],
    "asks": [["53732.00", "0.250"], ["53740.00", "0.800"]],
}

TRADES_RESPONSE = [
    {
        "id":            9001,
        "price":         "53734.56",
        "qty":           "0.01",
        "quoteQty":      "537.35",
        "time":          1700086400000,
        "isBuyerMaker":  False,
    },
    {
        "id":            9002,
        "price":         "53730.00",
        "qty":           "0.02",
        "quoteQty":      "1074.60",
        "time":          1700086401000,
        "isBuyerMaker":  True,
    },
]

EXCHANGE_INFO_RESPONSE = {
    "symbols": [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING",  "isSpotTradingAllowed": True},
        {"symbol": "ETHUSDT", "quoteAsset": "USDT", "status": "TRADING",  "isSpotTradingAllowed": True},
        {"symbol": "BTCBUSD", "quoteAsset": "BUSD", "status": "TRADING",  "isSpotTradingAllowed": True},
        {"symbol": "XRPUSDT", "quoteAsset": "USDT", "status": "BREAK",    "isSpotTradingAllowed": True},
        {"symbol": "BNBUSDT", "quoteAsset": "USDT", "status": "TRADING",  "isSpotTradingAllowed": False},
    ]
}


def _mock_get(response_data):
    """Returns a mock requests.get that yields response_data as .json()."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: get_ticker_24h
# ---------------------------------------------------------------------------

class TestGetTicker24h:
    def test_returns_expected_keys(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TICKER_24H_RESPONSE)):
            result = binance_market.get_ticker_24h("BTCUSDT")

        assert result["symbol"] == "BTCUSDT"
        assert result["last_price"] == pytest.approx(53734.56)
        assert result["price_change_pct"] == pytest.approx(2.35)
        assert result["high_24h"] == pytest.approx(54200.00)
        assert result["low_24h"] == pytest.approx(52100.00)
        assert result["volume_24h"] == pytest.approx(18500.00)
        assert result["quote_volume_24h"] == pytest.approx(990_000_000.00)

    def test_timestamps_are_datetime(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TICKER_24H_RESPONSE)):
            result = binance_market.get_ticker_24h("BTCUSDT")

        assert isinstance(result["open_time"], datetime)
        assert isinstance(result["close_time"], datetime)

    def test_symbol_uppercased(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TICKER_24H_RESPONSE)) as mock_req:
            binance_market.get_ticker_24h("btcusdt")
            call_params = mock_req.call_args[1]["params"]
            assert call_params["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# Tests: get_latest_price
# ---------------------------------------------------------------------------

class TestGetLatestPrice:
    def test_returns_float(self):
        with patch("binance_market.requests.get", return_value=_mock_get(PRICE_RESPONSE)):
            price = binance_market.get_latest_price("BTCUSDT")
        assert isinstance(price, float)
        assert price == pytest.approx(53734.56)


# ---------------------------------------------------------------------------
# Tests: get_best_bid_ask
# ---------------------------------------------------------------------------

class TestGetBestBidAsk:
    def test_spread_calculation(self):
        with patch("binance_market.requests.get", return_value=_mock_get(BOOK_TICKER_RESPONSE)):
            result = binance_market.get_best_bid_ask("BTCUSDT")

        assert result["bid_price"] == pytest.approx(53730.00)
        assert result["ask_price"] == pytest.approx(53732.00)
        assert result["spread"]    == pytest.approx(2.00)
        assert result["spread_pct"] > 0

    def test_all_keys_present(self):
        with patch("binance_market.requests.get", return_value=_mock_get(BOOK_TICKER_RESPONSE)):
            result = binance_market.get_best_bid_ask("BTCUSDT")

        for key in ("symbol", "bid_price", "bid_qty", "ask_price", "ask_qty", "spread", "spread_pct"):
            assert key in result


# ---------------------------------------------------------------------------
# Tests: get_order_book
# ---------------------------------------------------------------------------

class TestGetOrderBook:
    def test_structure(self):
        with patch("binance_market.requests.get", return_value=_mock_get(DEPTH_RESPONSE)):
            result = binance_market.get_order_book("BTCUSDT", limit=10)

        assert "bids" in result and "asks" in result
        assert "last_update_id" in result
        assert isinstance(result["bids"], list)
        assert isinstance(result["asks"], list)

    def test_bid_ask_are_dicts_with_price_qty(self):
        with patch("binance_market.requests.get", return_value=_mock_get(DEPTH_RESPONSE)):
            result = binance_market.get_order_book("BTCUSDT")

        for entry in result["bids"] + result["asks"]:
            assert "price" in entry and "qty" in entry
            assert isinstance(entry["price"], float)
            assert isinstance(entry["qty"], float)

    def test_bids_highest_first(self):
        with patch("binance_market.requests.get", return_value=_mock_get(DEPTH_RESPONSE)):
            result = binance_market.get_order_book("BTCUSDT")

        prices = [b["price"] for b in result["bids"]]
        assert prices == sorted(prices, reverse=True)

    def test_asks_lowest_first(self):
        with patch("binance_market.requests.get", return_value=_mock_get(DEPTH_RESPONSE)):
            result = binance_market.get_order_book("BTCUSDT")

        prices = [a["price"] for a in result["asks"]]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# Tests: get_recent_trades
# ---------------------------------------------------------------------------

class TestGetRecentTrades:
    def test_returns_list_of_dicts(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TRADES_RESPONSE)):
            result = binance_market.get_recent_trades("BTCUSDT", limit=2)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_side_derived_from_buyer_maker(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TRADES_RESPONSE)):
            result = binance_market.get_recent_trades("BTCUSDT")

        assert result[0]["side"] == "BUY"   # isBuyerMaker=False → BUY
        assert result[1]["side"] == "SELL"  # isBuyerMaker=True  → SELL

    def test_time_is_datetime(self):
        with patch("binance_market.requests.get", return_value=_mock_get(TRADES_RESPONSE)):
            result = binance_market.get_recent_trades("BTCUSDT")

        assert isinstance(result[0]["time"], datetime)


# ---------------------------------------------------------------------------
# Tests: get_all_usdt_symbols
# ---------------------------------------------------------------------------

class TestGetAllUsdtSymbols:
    def test_filters_only_active_usdt_spot(self):
        with patch("binance_market.requests.get", return_value=_mock_get(EXCHANGE_INFO_RESPONSE)):
            result = binance_market.get_all_usdt_symbols()

        # BTCBUSD  → excluded (quoteAsset != USDT)
        # XRPUSDT  → excluded (status != TRADING)
        # BNBUSDT  → excluded (isSpotTradingAllowed=False)
        assert set(result) == {"BTCUSDT", "ETHUSDT"}

    def test_returns_sorted_list(self):
        with patch("binance_market.requests.get", return_value=_mock_get(EXCHANGE_INFO_RESPONSE)):
            result = binance_market.get_all_usdt_symbols()

        assert result == sorted(result)


# ---------------------------------------------------------------------------
# Tests: get_market_summary
# ---------------------------------------------------------------------------

class TestGetMarketSummary:
    def test_merges_ticker_and_spread(self):
        with patch("binance_market.get_ticker_24h", return_value={"symbol": "BTCUSDT", "last_price": 53734.56}), \
             patch("binance_market.get_best_bid_ask", return_value={"symbol": "BTCUSDT", "spread": 2.0}):
            result = binance_market.get_market_summary("BTCUSDT")

        assert "last_price" in result
        assert "spread" in result
