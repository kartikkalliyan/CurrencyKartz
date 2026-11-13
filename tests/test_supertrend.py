"""
Tests for the Supertrend indicator and strategy.

Regression guard for a real bug: the bands were seeded with 0.0 and then updated
from an ATR that is NaN for the first `period` rows. The NaN propagated forever
(NaN comparisons are always False), so ~98% of Supertrend values were NaN and the
strategy produced zero trades on real BTC data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from indicators import add_supertrend
from strategy_supertrend import generate_signals


def _wave_df(n=400, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 0.03 * t + 15 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    })


def test_supertrend_is_defined_after_atr_warmup():
    """Only the ATR warm-up rows may be NaN -- nothing after them."""
    df = add_supertrend(_wave_df(), period=10)
    first_valid = df["ATR"].first_valid_index()
    after = df.loc[first_valid:, "Supertrend"]
    assert after.notna().all()


def test_warmup_rows_are_nan_not_zero():
    df = add_supertrend(_wave_df(), period=10)
    first_valid = df["ATR"].first_valid_index()
    assert df.loc[:first_valid - 1, "Supertrend"].isna().all()


def test_direction_only_one_or_minus_one():
    df = add_supertrend(_wave_df())
    assert set(df["Supertrend_dir"].unique()) <= {1, -1}


def test_both_directions_occur_on_an_oscillating_market():
    df = add_supertrend(_wave_df())
    assert (df["Supertrend_dir"] == 1).any()
    assert (df["Supertrend_dir"] == -1).any()


def test_strategy_emits_both_buy_and_sell_signals():
    """The headline regression: this used to produce zero signals."""
    sig = generate_signals(_wave_df())
    counts = sig["signal"].value_counts().to_dict()
    assert counts.get("BUY", 0) >= 2
    assert counts.get("SELL", 0) >= 2


def test_sharp_crash_flips_to_bearish():
    n = 200
    close = np.concatenate([np.linspace(100, 200, 150), np.linspace(100, 90, 50)])  # rally, then crash
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    })
    out = add_supertrend(df)
    assert out["Supertrend_dir"].iloc[149] == 1        # bullish at the top of the rally
    assert out["Supertrend_dir"].iloc[150] == -1       # flips the very bar price breaks the band
    assert out["Supertrend_dir"].iloc[-1] == -1


def test_steady_rally_stays_bullish_with_support_below_price():
    n = 200
    close = np.linspace(100, 300, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    })
    out = add_supertrend(df)
    tail = out.iloc[50:]
    assert (tail["Supertrend_dir"] == 1).all()
    assert (tail["Supertrend"] < tail["close"]).all()   # bullish line sits under price


def test_direction_is_causal():
    """Adding future rows must not change past directions (no look-ahead)."""
    df = _wave_df()
    full = add_supertrend(df.copy())
    prefix = add_supertrend(df.iloc[:250].copy())
    pd.testing.assert_series_equal(full["Supertrend_dir"].iloc[:250].reset_index(drop=True),
                                   prefix["Supertrend_dir"].reset_index(drop=True))


def test_all_nan_atr_does_not_crash():
    df = _wave_df(n=5)   # shorter than the ATR period -> ATR never becomes valid
    out = add_supertrend(df, period=10)
    assert out["Supertrend"].isna().all()
