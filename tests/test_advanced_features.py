"""
tests/test_advanced_features.py
-------------------------------
Unit tests for Advanced Features:
  - Feature 1: Advanced Risk & Performance Analytics (Equity Drawdown Chart)
  - Feature 3: Telegram & Discord Notification Alerts (alerts.py)
  - Feature 4: Paper Trading Engine (paper_trading.py)
  - Feature 5: VWAP & Supertrend Indicators & Strategies
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alerts
import paper_trading
import indicators
import strategy_supertrend
import strategy_vwap
from display_helpers import make_equity_drawdown_chart


# ---------------------------------------------------------------------------
# Fixture: Sample OHLCV DataFrame
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ohlcv_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100.0 + np.cumsum(np.random.randn(n) * 2.0)
    high = price + np.random.rand(n) * 3.0
    low = price - np.random.rand(n) * 3.0
    volume = np.random.randint(1000, 10000, size=n).astype(float)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    })
    return df


# ---------------------------------------------------------------------------
# Feature 3 Tests: Alerts
# ---------------------------------------------------------------------------
class TestAlerts:
    def test_format_signal_alert(self):
        msg = alerts.format_signal_alert("BTCUSDT", "BUY", 50000.0, "MA Crossover", "1h")
        assert "BTCUSDT" in msg
        assert "BUY" in msg
        assert "$50,000.0000" in msg

    @patch("alerts.requests.post")
    def test_send_telegram_alert_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        res = alerts.send_telegram_alert("mock_token", "12345", "Test alert")
        assert res["success"] is True

    @patch("alerts.requests.post")
    def test_send_discord_alert_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        res = alerts.send_discord_alert("https://discord.com/api/webhooks/test", "Test alert")
        assert res["success"] is True


# ---------------------------------------------------------------------------
# Feature 4 Tests: Paper Trading Engine
# ---------------------------------------------------------------------------
class TestPaperTrading:
    def test_paper_trading_init_and_reset(self):
        paper_trading.reset_paper_account(10000.0)
        summary = paper_trading.get_account_summary()
        assert summary["cash_balance"] == pytest.approx(10000.0)
        assert summary["open_positions_count"] == 0

    def test_paper_buy_order_executes(self):
        paper_trading.reset_paper_account(10000.0)
        res = paper_trading.execute_paper_order("BTCUSDT", "BUY", 50000.0, amount_usd=2000.0)
        assert res["success"] is True

        summary = paper_trading.get_account_summary()
        assert summary["cash_balance"] == pytest.approx(8000.0)
        assert summary["open_positions_count"] == 1

    def test_paper_sell_order_closes_position(self):
        paper_trading.reset_paper_account(10000.0)
        paper_trading.execute_paper_order("BTCUSDT", "BUY", 50000.0, amount_usd=2000.0) # 0.04 BTC

        # Sell at higher price ($60,000 -> 20% gain on $2,000 = $400 gain)
        res = paper_trading.execute_paper_order("BTCUSDT", "SELL", 60000.0)
        assert res["success"] is True

        summary = paper_trading.get_account_summary()
        assert summary["cash_balance"] == pytest.approx(10400.0)
        assert summary["realized_pnl"] == pytest.approx(400.0)
        assert summary["open_positions_count"] == 0

    def test_insufficient_funds_fails(self):
        paper_trading.reset_paper_account(10000.0)
        res = paper_trading.execute_paper_order("BTCUSDT", "BUY", 50000.0, amount_usd=50000.0)
        assert res["success"] is False
        assert "Insufficient cash balance" in res["error"]


# ---------------------------------------------------------------------------
# Feature 5 Tests: VWAP & Supertrend Indicators / Strategies
# ---------------------------------------------------------------------------
class TestIndicatorsAndStrategies:
    def test_add_vwap(self, sample_ohlcv_df):
        df = indicators.add_vwap(sample_ohlcv_df)
        assert "VWAP" in df.columns
        assert not df["VWAP"].isnull().all()
        assert (df["VWAP"] > 0).all()

    def test_add_supertrend(self, sample_ohlcv_df):
        df = indicators.add_supertrend(sample_ohlcv_df)
        assert "Supertrend" in df.columns
        assert "Supertrend_dir" in df.columns
        assert set(df["Supertrend_dir"].unique()).issubset({1, -1})

    def test_strategy_supertrend(self, sample_ohlcv_df):
        sig_df = strategy_supertrend.generate_signals(sample_ohlcv_df)
        assert "signal" in sig_df.columns
        assert set(sig_df["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})

    def test_strategy_vwap(self, sample_ohlcv_df):
        sig_df = strategy_vwap.generate_signals(sample_ohlcv_df)
        assert "signal" in sig_df.columns
        assert set(sig_df["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})


# ---------------------------------------------------------------------------
# Feature 1 Tests: Equity Drawdown Chart Helper
# ---------------------------------------------------------------------------
class TestChartHelper:
    def test_make_equity_drawdown_chart(self, sample_ohlcv_df):
        mock_results = {
            "df": pd.DataFrame({
                "timestamp": sample_ohlcv_df["timestamp"],
                "equity": 1000.0 + np.cumsum(np.random.randn(100) * 10),
                "buy_hold_equity": 1000.0 + np.cumsum(np.random.randn(100) * 8),
            })
        }
        fig = make_equity_drawdown_chart(mock_results)
        assert fig is not None
        assert len(fig.data) >= 2
