"""
Tests for the strategy modules -- verify each one produces a valid
'signal' column (only BUY/SELL/HOLD, no missing/garbage values) when
run on realistic data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from indicators import add_all_indicators
from regime_detector import add_regime

import strategy as rsi_strategy
import strategy_v4_filtered_ma as ma_crossover_strategy
import strategy_bollinger
import strategy_macd
import strategy_breakout
import strategy_buyhold


VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

STRATEGY_MODULES = [
    rsi_strategy, ma_crossover_strategy, strategy_bollinger,
    strategy_macd, strategy_breakout, strategy_buyhold,
]


def _prepped(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv.copy())
    df = add_regime(df)
    return df


def test_all_strategies_produce_only_valid_signal_values(sample_ohlcv):
    df = _prepped(sample_ohlcv)
    for module in STRATEGY_MODULES:
        result = module.generate_signals(df.copy())
        unique_signals = set(result["signal"].unique())
        assert unique_signals.issubset(VALID_SIGNALS), (
            f"{module.__name__} produced invalid signal values: "
            f"{unique_signals - VALID_SIGNALS}"
        )


def test_all_strategies_produce_no_missing_signals(sample_ohlcv):
    df = _prepped(sample_ohlcv)
    for module in STRATEGY_MODULES:
        result = module.generate_signals(df.copy())
        assert result["signal"].isna().sum() == 0, (
            f"{module.__name__} left some rows with no signal value"
        )


def test_buyhold_only_buys_once():
    """The buy-and-hold baseline should BUY exactly once (day 1) and
    HOLD forever after -- never sell."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
        "close": range(100, 110),
    })
    result = strategy_buyhold.generate_signals(df)
    assert (result["signal"] == "BUY").sum() == 1
    assert result.iloc[0]["signal"] == "BUY"
    assert (result["signal"] == "SELL").sum() == 0


def test_ma_crossover_only_signals_during_trending_regime(sample_ohlcv):
    """The regime-filtered MA crossover strategy should never emit a
    BUY or SELL signal on a day classified as CHOPPY -- that's the
    whole point of the filter."""
    df = _prepped(sample_ohlcv)
    result = ma_crossover_strategy.generate_signals(df.copy())
    active_signals = result[result["signal"] != "HOLD"]
    if len(active_signals) > 0:
        assert (active_signals["regime"] == "TRENDING").all()