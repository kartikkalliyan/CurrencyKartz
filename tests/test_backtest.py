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
import pytest
from functools import partial
from backtest import backtest

# The original tests check the money math on tiny close-only data using the
# legacy same-bar-close fills. The realistic default is covered further below.
legacy = partial(backtest, fill_mode="same_close")


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
    r = legacy(df, starting_cash=1000, fee_rate=0.0)
    assert abs(r["strategy_return_pct"] - r["buy_hold_return_pct"]) < 1e-9


def test_fee_reduces_return():
    df = _make_signal_df([100, 110], ["BUY", "SELL"])
    no_fee = legacy(df, starting_cash=1000, fee_rate=0.0)
    with_fee = legacy(df, starting_cash=1000, fee_rate=0.01)
    assert with_fee["strategy_return_pct"] < no_fee["strategy_return_pct"]


def test_fixed_stop_loss_triggers_at_correct_price():
    """Buy at 100 with a 10% stop -- should exit once price hits 90,
    not before and not after."""
    df = _make_signal_df([100, 95, 91, 89, 85], ["BUY", "HOLD", "HOLD", "HOLD", "HOLD"])
    r = legacy(df, starting_cash=1000, fee_rate=0.0, stop_loss_pct=0.10)
    stop_trades = [t for t in r["trade_log"] if t["action"] == "STOP_LOSS_SELL"]
    assert len(stop_trades) == 1
    assert stop_trades[0]["price"] == 89  # first close that is >=10% below 100


def test_trailing_stop_locks_gains_better_than_fixed_stop():
    """Price rallies from 100 to 150, then falls back. A trailing stop
    should exit much higher (protecting the rally) than a fixed stop
    anchored to the original 100 entry."""
    closes = [100, 120, 140, 150, 140, 130, 110, 95]
    signals = ["BUY"] + ["HOLD"] * 7
    df = _make_signal_df(closes, signals)

    fixed = legacy(df, starting_cash=1000, fee_rate=0.0, stop_loss_pct=0.10)
    trailing = legacy(df, starting_cash=1000, fee_rate=0.0, trailing_stop_pct=0.10)

    assert trailing["final_value_strategy"] > fixed["final_value_strategy"]


def test_position_sizing_limits_loss_to_risk_percentage():
    """With risk_per_trade_pct=0.02 and a 10% stop, a full stop-out
    should cost close to 2% of starting capital, not 10%."""
    df = _make_signal_df([100, 95, 91, 89], ["BUY", "HOLD", "HOLD", "HOLD"])
    r = legacy(df, starting_cash=1000, fee_rate=0.0,
                 stop_loss_pct=0.10, risk_per_trade_pct=0.02)
    loss_pct = (1000 - r["final_value_strategy"]) / 1000 * 100
    assert 1.5 < loss_pct < 2.5  # roughly 2%, not the full 10%


def test_position_sizing_without_stop_raises_error():
    """Risk-based sizing needs a stop distance to size against -- should
    fail loudly rather than silently doing something undefined."""
    df = _make_signal_df([100, 110], ["BUY", "SELL"])
    try:
        legacy(df, starting_cash=1000, risk_per_trade_pct=0.02)
        assert False, "Expected a ValueError but none was raised"
    except ValueError:
        pass


def test_slippage_reduces_return():
    df = _make_signal_df([100, 110, 120, 130], ["BUY", "HOLD", "HOLD", "SELL"])
    no_slip = legacy(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.0)
    with_slip = legacy(df, starting_cash=1000, fee_rate=0.001, slippage_pct=0.01)
    assert with_slip["strategy_return_pct"] < no_slip["strategy_return_pct"]


def test_no_trades_if_no_signals_fire():
    df = _make_signal_df([100, 101, 102, 103], ["HOLD", "HOLD", "HOLD", "HOLD"])
    r = legacy(df, starting_cash=1000)
    assert r["num_trades"] == 0
    assert r["final_value_strategy"] == 1000  # cash never moved

# ---------------------------------------------------------------------------
# Realistic execution: decide on the close of bar t, fill at the open of t+1
# ---------------------------------------------------------------------------

def _make_ohlc_df(opens, closes, signals):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "open": opens,
        "close": closes,
        "signal": signals,
    })


def test_next_open_is_the_default_mode():
    df = _make_ohlc_df([100, 100, 100], [100, 100, 100], ["HOLD"] * 3)
    assert backtest(df)["fill_mode"] == "next_open"


def test_next_open_requires_open_column():
    df = _make_signal_df([100, 110], ["BUY", "SELL"])
    with pytest.raises(ValueError, match="open"):
        backtest(df)


def test_invalid_fill_mode_rejected():
    df = _make_ohlc_df([100, 100], [100, 100], ["HOLD", "HOLD"])
    with pytest.raises(ValueError, match="fill_mode"):
        backtest(df, fill_mode="same_open")


def test_buy_signal_fills_at_next_bars_open_not_signal_close():
    """BUY decided on day 0's close (100) must fill at day 1's open (105)."""
    df = _make_ohlc_df([100, 105, 110], [100, 108, 112], ["BUY", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0)
    buy = r["trade_log"][0]
    assert buy["action"] == "BUY"
    assert buy["price"] == 105
    assert buy["date"] == df["timestamp"][1]
    assert buy["signal_date"] == df["timestamp"][0]


def test_sell_signal_fills_at_next_bars_open():
    df = _make_ohlc_df([100, 100, 120, 90], [100, 110, 130, 95],
                       ["BUY", "SELL", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0)
    sell = [t for t in r["trade_log"] if t["action"] == "SELL"][0]
    # BUY decided bar0 -> fills bar1 open (100); SELL decided bar1 close -> fills bar2 open (120)
    assert sell["price"] == 120
    assert sell["date"] == df["timestamp"][2]
    assert r["final_value_strategy"] == pytest.approx(1000 * 120 / 100)


def test_signal_on_last_bar_never_fills():
    """There is no next bar to trade on, so the order must be dropped."""
    df = _make_ohlc_df([100, 100, 100], [100, 100, 100], ["HOLD", "HOLD", "BUY"])
    r = backtest(df, starting_cash=1000)
    assert r["num_trades"] == 0
    assert r["final_value_strategy"] == 1000


def test_stop_loss_decided_on_close_fills_at_next_open():
    """Close breaches the 10% stop on bar 2, but we can only exit at bar 3's open
    (here a gap down to 80) -- real stops suffer gap risk."""
    df = _make_ohlc_df([100, 100, 95, 80, 80], [100, 96, 89, 82, 81],
                       ["BUY", "HOLD", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0, stop_loss_pct=0.10)
    stops = [t for t in r["trade_log"] if t["action"] == "STOP_LOSS_SELL"]
    assert len(stops) == 1
    assert stops[0]["price"] == 80
    assert stops[0]["date"] == df["timestamp"][3]


def test_trailing_stop_fills_at_next_open():
    df = _make_ohlc_df([100, 100, 140, 120, 118], [100, 150, 135, 119, 118],
                       ["BUY", "HOLD", "HOLD", "HOLD", "HOLD"])
    r = backtest(df, starting_cash=1000, fee_rate=0.0, trailing_stop_pct=0.10)
    stops = [t for t in r["trade_log"] if t["action"] == "TRAILING_STOP_SELL"]
    assert len(stops) == 1
    # peak 150 (bar1), bar2 close 135 is exactly 10% below -> exit at bar3 open
    assert stops[0]["price"] == 120


def test_next_open_does_not_use_future_prices_for_the_decision():
    """Changing prices AFTER a decision bar must not change that decision."""
    base = _make_ohlc_df([100, 100, 100, 100], [100, 100, 100, 100],
                         ["BUY", "HOLD", "SELL", "HOLD"])
    changed = base.copy()
    changed.loc[3, ["open", "close"]] = 500  # future bar after the SELL decision
    r1 = backtest(base, fee_rate=0.0)
    r2 = backtest(changed, fee_rate=0.0)
    assert [t["action"] for t in r1["trade_log"]] == [t["action"] for t in r2["trade_log"]]
    assert [t["date"] for t in r1["trade_log"]] == [t["date"] for t in r2["trade_log"]]


def test_next_open_is_no_better_than_same_close_on_a_rising_market():
    """On a steadily rising market, delaying the entry to the next open can only
    cost money -- guards against accidentally re-introducing same-bar fills."""
    opens = [100, 103, 106, 109, 112, 115]
    closes = [102, 105, 108, 111, 114, 117]
    signals = ["BUY", "HOLD", "HOLD", "HOLD", "HOLD", "HOLD"]
    df = _make_ohlc_df(opens, closes, signals)
    realistic = backtest(df, fee_rate=0.0, fill_mode="next_open")
    optimistic = backtest(df, fee_rate=0.0, fill_mode="same_close")
    assert realistic["strategy_return_pct"] < optimistic["strategy_return_pct"]
