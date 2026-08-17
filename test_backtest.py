"""
Tests for backtest.py

These verify the actual MONEY MATH is correct -- the most important
tests in the whole project, since a bug here would silently produce
wrong return numbers that look plausible.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from backtest import backtest


def _make_signal_df(closes, signals):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
        "signal": signals,
    })


def test_buy_and_hold_matches_benchmark_exactly():
    """If we buy on day 1 and never sell, our return should EXACTLY
    match the buy-and-hold benchmark the function itself computes."""
    df = _make_signal_df([100, 110, 120, 130], ["BUY", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0)
    assert abs(r["strategy_return_pct"] - r["buy_hold_return_pct"]) < 1e-9


def test_fee_reduces_return():
    df = _make_signal_df([100, 110], ["BUY", "SELL"])
    no_fee = backtest(df, starting_cash=1000, fee_rate=0.0)
    with_fee = backtest(df, starting_cash=1000, fee_rate=0.01)
    assert with_fee["strategy_return_pct"] < no_fee["strategy_return_pct"]


def test_fixed_stop_loss_triggers_at_correct_price():
    """Buy at 100 with a 10% stop -- should exit once price hits 90,
    not before and not after."""
    df = _make_signal_df([100, 95, 91, 89, 85], ["BUY", "HOLD", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0, stop_loss_pct=0.10)
    stop_trades = [t for t in r["trade_log"] if t["action"] == "STOP_LOSS_SELL"]
    assert len(stop_trades) == 1
    assert abs(stop_trades[0]["price"] - 89) < 1e-9  # first close that is >=10% below 100


def test_trailing_stop_locks_gains_better_than_fixed_stop():
    """Price rallies from 100 to 150, then falls back. A trailing stop
    should exit much higher (protecting the rally) than a fixed stop
    anchored to the original 100 entry."""
    closes = [100, 120, 140, 150, 140, 130, 110, 95]
    signals = ["BUY"] + ["HOLD"] * 7
    df = _make_signal_df(closes, signals)

    fixed = backtest(df, starting_cash=1000, fee_rate=0.0, stop_loss_pct=0.10)
    trailing = backtest(df, starting_cash=1000, fee_rate=0.0, trailing_stop_pct=0.10)

    assert trailing["final_value_strategy"] > fixed["final_value_strategy"]


def test_position_sizing_limits_loss_to_risk_percentage():
    """With risk_per_trade_pct=0.02 and a 10% stop, a full stop-out
    should cost close to 2% of starting capital, not 10%."""
    df = _make_signal_df([100, 95, 91, 89], ["BUY", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0,
                 stop_loss_pct=0.10, risk_per_trade_pct=0.02)
    loss_pct = (1000 - r["final_value_strategy"]) / 1000 * 100
    assert 1.5 < loss_pct < 2.5  # roughly 2%, not the full 10%


def test_position_sizing_without_stop_raises_error():
    """Risk-based sizing needs a stop distance to size against -- should
    fail loudly rather than silently doing something undefined."""
    df = _make_signal_df([100, 110], ["BUY", "SELL"])
    try:
        backtest(df, starting_cash=1000, risk_per_trade_pct=0.02)
        assert False, "Expected a ValueError but none was raised"
    except ValueError:
        pass


def test_slippage_reduces_return():
    df = _make_signal_df([100, 110, 120, 130], ["BUY", "HOLD", "HOLD", "SELL"])
    no_slip = backtest(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.0)
    with_slip = backtest(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.01)
    assert with_slip["strategy_return_pct"] < no_slip["strategy_return_pct"]


def test_no_trades_if_no_signals_fire():
    df = _make_signal_df([100, 101, 102, 103], ["HOLD", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000)
    assert r["num_trades"] == 0
    assert r["final_value_strategy"] == 1000  # cash never moved


# --- Added for Step 3 of the quant research roadmap (spread modeling
# and equity curve, needed for Sharpe/Sortino/drawdown in metrics.py) ---

def test_spread_costs_money_on_flat_round_trip():
    """Buying and selling at the exact same price should still lose
    money purely from crossing the spread twice."""
    df = _make_signal_df([100, 100], ["BUY", "SELL"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0, spread_pct=0.01)
    assert r["final_value_strategy"] < 1000


def test_spread_matches_exact_multiplicative_math():
    """The round-trip cost of a spread is a multiplicative effect, not
    a simple subtraction -- verify against the precise expected value."""
    df = _make_signal_df([100, 100], ["BUY", "SELL"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0, spread_pct=0.01)
    expected = 1000 * (1 - 0.005) / (1 + 0.005)
    assert abs(r["final_value_strategy"] - expected) < 0.001


def test_spread_and_slippage_combine():
    """Spread and slippage are independent costs -- having both active
    should cost more than either alone."""
    df = _make_signal_df([100, 100], ["BUY", "SELL"])
    spread_only = backtest(df, starting_cash=1000, fee_rate=0.0, spread_pct=0.01)
    both = backtest(df, starting_cash=1000, fee_rate=0.0, slippage_pct=0.01, spread_pct=0.01)
    assert both["final_value_strategy"] < spread_only["final_value_strategy"]


def test_zero_spread_matches_omitted_parameter():
    """Backward compatibility: explicitly passing spread_pct=0.0 must
    produce identical results to not passing it at all."""
    df = _make_signal_df([100, 110, 120, 130], ["BUY", "HOLD", "HOLD", "SELL"])
    with_zero = backtest(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.005, spread_pct=0.0)
    without_param = backtest(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.005)
    assert abs(with_zero["final_value_strategy"] - without_param["final_value_strategy"]) < 1e-9


def test_equity_curve_exists_and_matches_final_value():
    """The equity curve (needed for Sharpe/Sortino/drawdown) must have
    one entry per row and end exactly at the reported final value."""
    df = _make_signal_df([100, 110, 120, 130], ["BUY", "HOLD", "HOLD", "SELL"])
    r = backtest(df, starting_cash=1000, fee_rate=0.001)
    assert len(r["equity_curve"]) == len(df)
    assert abs(r["equity_curve"].iloc[-1] - r["final_value_strategy"]) < 1e-6


def test_equity_curve_tracks_mark_to_market_mid_position():
    """The equity curve must reflect the CURRENT value of an open
    position on every day, not just at trade events."""
    df = _make_signal_df([100, 150, 200], ["BUY", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0)
    # Day 2 (price=150): holding 10 btc bought at 100 -> should be worth 1500
    assert abs(r["equity_curve"].iloc[1] - 1500) < 1e-6