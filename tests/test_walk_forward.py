"""
Tests for walk_forward.py

Regression guard for a real bug: indicators used to be recomputed inside each
short test window, so a 40-row window could never fill MA50 and the strategy
silently never traded out-of-sample (all windows showed 0 trades / 0% return).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import walk_forward
from walk_forward import run_walk_forward_analysis, _prep


def _wave_df(n=700, seed=0):
    """Trending price with a regular oscillation, so MA20/MA50 cross often."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 0.05 * t + 12 * np.sin(2 * np.pi * t / 70) + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close + rng.normal(0, 0.1, n),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })


def ma_cross(df):
    """Plain causal MA20/MA50 crossover (no regime filter) for testing."""
    df = df.copy()
    df["signal"] = "HOLD"
    prev20, prev50 = df["MA_20"].shift(1), df["MA_50"].shift(1)
    df.loc[(prev20 <= prev50) & (df["MA_20"] > df["MA_50"]), "signal"] = "BUY"
    df.loc[(prev20 >= prev50) & (df["MA_20"] < df["MA_50"]), "signal"] = "SELL"
    return df


def test_test_windows_actually_trade():
    """The bug: 40-row test windows produced zero trades. They must trade now."""
    res = run_walk_forward_analysis(_wave_df(), ma_cross, train_days=120, test_days=40)
    w = res["windows_df"]
    assert len(w) > 5
    assert res["oos_windows_with_trades"] > 0
    assert w["Out-Sample Trades"].sum() > 0


def test_signals_are_causal_when_computed_on_full_history():
    """Signals for the first N rows must not change when later rows are added.
    This is what makes computing indicators once on the full history safe."""
    df = _wave_df()
    n = 400
    full = ma_cross(_prep(df.copy())).iloc[:n]
    prefix = ma_cross(_prep(df.iloc[:n].copy()))
    pd.testing.assert_series_equal(full["signal"].reset_index(drop=True),
                                   prefix["signal"].reset_index(drop=True))


def test_windows_never_start_inside_warmup():
    df = _wave_df()
    res = run_walk_forward_analysis(df, ma_cross, train_days=120, test_days=40, warmup_rows=50)
    first_train_start = pd.Timestamp(res["windows_df"].iloc[0]["Train Start"])
    assert first_train_start >= df["timestamp"].iloc[50]


def test_windows_do_not_overlap_in_test_periods():
    res = run_walk_forward_analysis(_wave_df(), ma_cross, train_days=120, test_days=40)
    w = res["windows_df"]
    test_ends = pd.to_datetime(w["Test End"])
    gaps = test_ends.diff().dropna().dt.days
    assert (gaps == 40).all()   # each step advances exactly one test window


def test_not_robust_when_no_out_of_sample_trades():
    """Sitting in cash must never be reported as a 'robust' strategy."""
    def never_trades(df):
        df = df.copy()
        df["signal"] = "HOLD"
        return df

    res = run_walk_forward_analysis(_wave_df(), never_trades, train_days=120, test_days=40)
    assert res["oos_windows_with_trades"] == 0
    assert res["is_robust"] is False


def test_too_little_data_returns_empty_result():
    res = run_walk_forward_analysis(_wave_df(n=100), ma_cross, train_days=120, test_days=40)
    assert res["windows_df"].empty
    assert res["is_robust"] is False
    assert res["oos_windows_with_trades"] == 0


def test_risk_kwargs_are_passed_through_to_backtest():
    """Higher fees must lower returns, proving risk_kwargs reach the backtester."""
    df = _wave_df()
    cheap = run_walk_forward_analysis(df, ma_cross, {"fee_rate": 0.0}, 120, 40)
    costly = run_walk_forward_analysis(df, ma_cross, {"fee_rate": 0.02}, 120, 40)
    assert costly["out_of_sample_avg_return"] < cheap["out_of_sample_avg_return"]
