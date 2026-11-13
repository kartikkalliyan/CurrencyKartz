"""
Tests for strategy_ensemble.py (Ensemble Voting).

Regression guard: the old version required several strategies to fire a one-day
BUY/SELL event on the SAME bar. Those events are rare, so on 6 years of BTC it
produced 0 BUYs and 1 SELL and never traded. Voting now happens on each
strategy's current stance (long / not long).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import strategy_ensemble as ens
from strategy_ensemble import _stance, generate_signals
from indicators import add_all_indicators
from regime_detector import add_regime


def _df(n=10):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
    })


def _stub(events):
    """Build a fake strategy that emits the given {row_index: 'BUY'|'SELL'} events."""
    def fn(df):
        df = df.copy()
        df["signal"] = "HOLD"
        for i, s in events.items():
            df.loc[i, "signal"] = s
        return df
    return fn


def test_stance_holds_between_signals_and_resets_on_sell():
    sig = pd.Series(["HOLD", "BUY", "HOLD", "HOLD", "SELL", "HOLD"])
    assert _stance(sig).tolist() == [0, 1, 1, 1, 0, 0]


def test_stance_is_flat_before_any_signal():
    assert _stance(pd.Series(["HOLD", "HOLD"])).tolist() == [0, 0]


def test_agreement_does_not_need_to_happen_on_the_same_bar(monkeypatch):
    """Strategy A goes long on bar 2, B on bar 4. With min_votes=2 the ensemble
    must buy on bar 4, even though the two BUY events were on different days."""
    monkeypatch.setattr(ens, "VOTING_STRATEGIES",
                        [_stub({2: "BUY"}), _stub({4: "BUY"}), _stub({})])
    out = generate_signals(_df(), min_votes=2)
    assert out["signal"].tolist().index("BUY") == 4
    assert out["signal"].tolist().count("BUY") == 1


def test_sells_when_votes_fall_below_threshold(monkeypatch):
    monkeypatch.setattr(ens, "VOTING_STRATEGIES",
                        [_stub({1: "BUY", 7: "SELL"}), _stub({2: "BUY"}), _stub({})])
    out = generate_signals(_df(), min_votes=2)
    sig = out["signal"].tolist()
    assert sig[2] == "BUY"      # both long from bar 2
    assert sig[7] == "SELL"     # first strategy exits -> only 1 vote left


def test_signals_alternate_buy_sell_and_start_with_buy(monkeypatch):
    monkeypatch.setattr(ens, "VOTING_STRATEGIES",
                        [_stub({1: "BUY", 4: "SELL", 6: "BUY"}),
                         _stub({2: "BUY", 5: "SELL", 7: "BUY"}),
                         _stub({})])
    out = generate_signals(_df(), min_votes=2)
    seq = [s for s in out["signal"] if s != "HOLD"]
    assert seq[0] == "BUY"
    assert all(a != b for a, b in zip(seq, seq[1:]))   # strictly alternating


def test_vote_columns_are_consistent(monkeypatch):
    monkeypatch.setattr(ens, "VOTING_STRATEGIES", [_stub({1: "BUY"}), _stub({}), _stub({})])
    out = generate_signals(_df(), min_votes=2)
    assert (out["buy_votes"] + out["sell_votes"] == 3).all()
    assert out["buy_votes"].max() == 1


def test_a_crashing_strategy_votes_not_long_instead_of_breaking_the_ensemble(monkeypatch):
    def broken(df):
        raise RuntimeError("boom")
    monkeypatch.setattr(ens, "VOTING_STRATEGIES", [_stub({1: "BUY"}), _stub({1: "BUY"}), broken])
    out = generate_signals(_df(), min_votes=2)
    assert "BUY" in out["signal"].tolist()


def test_real_strategies_trade_and_are_causal():
    """The headline regression on realistic data, plus a no-look-ahead check."""
    rng = np.random.default_rng(5)
    n = 900
    t = np.arange(n)
    close = 100 + 0.05 * t + 14 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.5, n)
    raw = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": close + rng.normal(0, 0.2, n), "high": close + 1.5, "low": close - 1.5,
        "close": close, "volume": rng.uniform(1000, 5000, n),
    })
    prep = lambda d: add_regime(add_all_indicators(d))

    full = generate_signals(prep(raw.copy()))
    assert (full["signal"] == "BUY").sum() >= 1
    assert (full["signal"] == "SELL").sum() >= 1

    prefix = generate_signals(prep(raw.iloc[:600].copy()))
    assert full["signal"].iloc[:600].tolist() == prefix["signal"].tolist()
