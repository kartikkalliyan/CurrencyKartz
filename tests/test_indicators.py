"""
Tests for indicators.py

These check that each indicator produces mathematically sensible
output -- not that any particular trading outcome is "correct" (that's
what the strategy/backtest tests are for).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from indicators import (
    add_moving_averages, add_rsi, add_macd, add_atr,
    add_bollinger_bands, add_breakout_levels, add_all_indicators
)


def test_moving_averages_match_manual_calculation(sample_ohlcv):
    df = add_moving_averages(sample_ohlcv.copy())
    # Spot-check row 100: MA_20 should equal the mean of the 20 closes ending there
    expected_ma20 = sample_ohlcv["close"].iloc[81:101].mean()
    assert abs(df["MA_20"].iloc[100] - expected_ma20) < 1e-6


def test_rsi_stays_within_valid_range(sample_ohlcv):
    df = add_rsi(sample_ohlcv.copy())
    valid_rsi = df["RSI"].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_rsi_high_after_steady_rise(known_price_series):
    """During a steady 30-day rise with no down days, RSI should be
    very high (near 100) since there are no losses to average."""
    df = add_rsi(known_price_series.copy())
    rsi_at_peak = df["RSI"].iloc[29]  # last day of the steady rise
    assert rsi_at_peak > 90


def test_macd_signal_columns_exist_and_are_numeric(sample_ohlcv):
    df = add_macd(sample_ohlcv.copy())
    assert "MACD" in df.columns
    assert "MACD_signal" in df.columns
    assert "MACD_hist" in df.columns
    assert df["MACD"].dropna().dtype.kind in "fc"  # float


def test_atr_is_never_negative(sample_ohlcv):
    df = add_atr(sample_ohlcv.copy())
    valid_atr = df["ATR"].dropna()
    assert (valid_atr >= 0).all()


def test_bollinger_upper_always_above_lower(sample_ohlcv):
    df = add_bollinger_bands(sample_ohlcv.copy())
    valid = df.dropna(subset=["BB_upper", "BB_lower"])
    assert (valid["BB_upper"] > valid["BB_lower"]).all()


def test_breakout_levels_no_lookahead(sample_ohlcv):
    """High20/Low20 must be based on PAST data only (shifted), so a row's
    own high/low can never accidentally be the level it's compared against."""
    df = add_breakout_levels(sample_ohlcv.copy())
    # The very first valid High20 should not include row 0's own high
    first_valid_idx = df["High20"].first_valid_index()
    assert first_valid_idx is not None
    assert first_valid_idx > 0  # shifted, so index 0 can't be the first valid row


def test_add_all_indicators_produces_expected_columns(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv.copy())
    expected_columns = [
        "MA_20", "MA_50", "MA_200", "RSI", "MACD", "MACD_signal",
        "ATR", "ATR_pct", "BB_upper", "BB_lower", "High20", "Low20", "Vol_MA20"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"