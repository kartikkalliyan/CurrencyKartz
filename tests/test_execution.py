"""
tests/test_execution.py
------------------------
Unit tests for Institutional Execution Slicer (execution_slicer.py).
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution_slicer import InstitutionalExecutionSlicer


@pytest.fixture
def sample_kline_df():
    """Generates synthetic OHLCV kline data."""
    dates = pd.date_range("2024-01-01", periods=10, freq="15min")
    prices = [100.0, 100.2, 100.1, 100.5, 100.4, 100.8, 101.0, 100.9, 101.2, 101.5]
    volumes = [1000, 2000, 1500, 5000, 3000, 2500, 4000, 1800, 3500, 6000]
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": volumes
    })


class TestInstitutionalExecutionSlicer:
    def test_twap_equal_slicing(self, sample_kline_df):
        slicer = InstitutionalExecutionSlicer(sample_kline_df, parent_order_usd=10000.0, num_slices=5, side="BUY")
        res = slicer.simulate_twap()

        assert res["algo"] == "TWAP"
        assert res["total_executed_usd"] == 10000.0
        assert len(res["fills_df"]) == 5
        # Each slice should be equal USD size ($2000)
        assert res["fills_df"]["child_usd"].nunique() == 1
        assert res["fills_df"]["child_usd"].iloc[0] == 2000.0

    def test_vwap_volume_proportional_slicing(self, sample_kline_df):
        slicer = InstitutionalExecutionSlicer(sample_kline_df, parent_order_usd=10000.0, num_slices=5, side="BUY")
        res = slicer.simulate_vwap()

        assert res["algo"] == "VWAP"
        assert pytest.approx(res["total_executed_usd"], abs=0.01) == 10000.0
        assert len(res["fills_df"]) == 5

        # Bar with highest volume in last 5 rows should receive highest child USD
        vols = sample_kline_df.tail(5)["volume"].values
        max_vol_idx = int(np.argmax(vols))
        max_child_idx = int(res["fills_df"]["child_usd"].idxmax())
        assert max_vol_idx == max_child_idx

    def test_shortfall_calculation_buy(self, sample_kline_df):
        slicer = InstitutionalExecutionSlicer(sample_kline_df, parent_order_usd=5000.0, num_slices=5, side="BUY")
        res = slicer.simulate_twap()
        
        assert "shortfall_bps" in res
        assert "arrival_price" in res
        assert "avg_exec_price" in res
        assert res["arrival_price"] > 0.0

    def test_sell_order_execution(self, sample_kline_df):
        slicer = InstitutionalExecutionSlicer(sample_kline_df, parent_order_usd=5000.0, num_slices=5, side="SELL")
        res_twap = slicer.simulate_twap()
        res_vwap = slicer.simulate_vwap()

        assert res_twap["total_executed_usd"] == 5000.0
        assert res_vwap["total_executed_usd"] == 5000.0
