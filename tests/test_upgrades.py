"""
tests/test_upgrades.py
-----------------------
Unit tests for the 5 Free Platform Upgrades:
  1. Monte Carlo Strategy Stress-Tester & VaR (monte_carlo.py)
  2. Crypto Fear & Greed Index & Sentiment Strategy (fear_greed.py, strategy_sentiment.py)
  3. Walk-Forward Optimization & Overfitting Guard (walk_forward.py)
  4. Cross-Asset Correlation Matrix (display_helpers.py)
  5. Weighted Custom Strategy Ensemble Builder (strategy_custom_ensemble.py)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import monte_carlo
import fear_greed
import strategy_sentiment
import walk_forward
import strategy_custom_ensemble
from display_helpers import make_monte_carlo_chart, make_correlation_heatmap


@pytest.fixture
def sample_ohlcv_df():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100.0 + np.cumsum(np.random.randn(n) * 2.0)
    high = price + np.random.rand(n) * 3.0
    low = price - np.random.rand(n) * 3.0
    volume = np.random.randint(1000, 10000, size=n).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    })


# ---------------------------------------------------------------------------
# Test 1: Monte Carlo Simulation & VaR
# ---------------------------------------------------------------------------
class TestMonteCarlo:
    def test_run_monte_carlo_simulation(self, sample_ohlcv_df):
        mock_backtest = {
            "df": pd.DataFrame({
                "timestamp": sample_ohlcv_df["timestamp"],
                "equity": 1000.0 + np.cumsum(np.random.randn(150) * 10)
            }),
            "initial_capital": 1000.0
        }
        res = monte_carlo.run_monte_carlo_simulation(mock_backtest, num_simulations=50, num_days=30)
        assert "paths" in res
        assert "percentiles" in res
        assert "var_95" in res
        assert "var_99" in res
        assert res["var_95"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: Fear & Greed Index & Sentiment Strategy
# ---------------------------------------------------------------------------
class TestFearGreed:
    @patch("fear_greed.requests.get")
    def test_fetch_fear_and_greed_index(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"value": "25", "value_classification": "Extreme Fear", "timestamp": "1700000000"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        res = fear_greed.fetch_fear_and_greed_index(limit=1)
        assert res["current_value"] == 25
        assert res["current_classification"] == "Extreme Fear"

    def test_strategy_sentiment(self, sample_ohlcv_df):
        sig_df = strategy_sentiment.generate_signals(sample_ohlcv_df)
        assert "signal" in sig_df.columns
        assert set(sig_df["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})


# ---------------------------------------------------------------------------
# Test 3: Walk-Forward Optimization
# ---------------------------------------------------------------------------
class TestWalkForward:
    def test_run_walk_forward_analysis(self, sample_ohlcv_df):
        from strategy_v4_filtered_ma import generate_signals as ma_signals
        res = walk_forward.run_walk_forward_analysis(sample_ohlcv_df, ma_signals, train_days=60, test_days=30)
        assert "windows_df" in res
        assert "efficiency_ratio" in res
        assert isinstance(res["is_robust"], bool)


# ---------------------------------------------------------------------------
# Test 4: Cross-Asset Correlation Heatmap
# ---------------------------------------------------------------------------
class TestCorrelationHeatmap:
    def test_make_correlation_heatmap(self):
        corr_df = pd.DataFrame({"BTC": [1.0, 0.8], "ETH": [0.8, 1.0]}, index=["BTC", "ETH"])
        fig = make_correlation_heatmap(corr_df)
        assert fig is not None
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# Test 5: Weighted Custom Strategy Ensemble
# ---------------------------------------------------------------------------
class TestWeightedEnsemble:
    def test_generate_weighted_ensemble_signals(self, sample_ohlcv_df):
        sig_df = strategy_custom_ensemble.generate_weighted_ensemble_signals(sample_ohlcv_df)
        assert "signal" in sig_df.columns
        assert "ensemble_score" in sig_df.columns
        assert set(sig_df["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})


# ---------------------------------------------------------------------------
# Test 6: Multi-Strategy Comparative Backtest
# ---------------------------------------------------------------------------
class TestComparativeBacktest:
    def test_run_comparative_backtest_returns_expected_keys(self, sample_ohlcv_df):
        from core_logic import run_comparative_backtest, prep_data
        df = prep_data(sample_ohlcv_df.copy())
        result = run_comparative_backtest(df, ["RSI Mean-Reversion", "MA Crossover (regime-filtered)"])
        assert "equity_df" in result
        assert "summary" in result

    def test_equity_df_has_strategy_columns(self, sample_ohlcv_df):
        from core_logic import run_comparative_backtest, prep_data
        df = prep_data(sample_ohlcv_df.copy())
        result = run_comparative_backtest(df, ["RSI Mean-Reversion", "MA Crossover (regime-filtered)"])
        strat_cols = [c for c in result["equity_df"].columns if c != "timestamp"]
        assert len(strat_cols) >= 1

    def test_summary_has_return_column(self, sample_ohlcv_df):
        from core_logic import run_comparative_backtest, prep_data
        df = prep_data(sample_ohlcv_df.copy())
        result = run_comparative_backtest(df, ["RSI Mean-Reversion", "MA Crossover (regime-filtered)"])
        assert "Return %" in result["summary"].columns

    def test_make_comparative_equity_chart_renders(self, sample_ohlcv_df):
        from core_logic import run_comparative_backtest, prep_data
        from display_helpers import make_comparative_equity_chart
        df = prep_data(sample_ohlcv_df.copy())
        result = run_comparative_backtest(df, ["RSI Mean-Reversion", "MA Crossover (regime-filtered)"])
        fig = make_comparative_equity_chart(result["equity_df"], title="Test Chart")
        assert fig is not None
        assert len(fig.data) >= 1
