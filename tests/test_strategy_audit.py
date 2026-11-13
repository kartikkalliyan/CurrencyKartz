"""
Audit that runs on EVERY strategy in the dashboard registry (config.STRATEGY_FUNCS),
so a newly added strategy is checked automatically.

Checks:
  - returns a 'signal' column containing only BUY / SELL / HOLD
  - is causal: signals for the first N rows are identical whether or not later rows
    exist (a strategy that peeks at the future would change its past signals)
  - does not crash on a realistic dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import STRATEGY_FUNCS
from core_logic import prep_data


def _market(n=700, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 0.04 * t + 14 * np.sin(2 * np.pi * t / 75) + rng.normal(0, 0.6, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="D"),
        "open": close + rng.normal(0, 0.2, n),
        "high": close + np.abs(rng.normal(1.0, 0.3, n)),
        "low": close - np.abs(rng.normal(1.0, 0.3, n)),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })


@pytest.mark.parametrize("name", list(STRATEGY_FUNCS))
def test_strategy_signals_are_valid(name):
    out = STRATEGY_FUNCS[name](prep_data(_market()))
    assert "signal" in out.columns
    assert set(out["signal"].unique()) <= {"BUY", "SELL", "HOLD"}


@pytest.mark.parametrize("name", list(STRATEGY_FUNCS))
def test_strategy_has_no_look_ahead(name):
    raw = _market()
    cut = 500
    full = STRATEGY_FUNCS[name](prep_data(raw.copy())).reset_index(drop=True)
    prefix = STRATEGY_FUNCS[name](prep_data(raw.iloc[:cut].copy())).reset_index(drop=True)
    assert full["signal"].iloc[:cut].tolist() == prefix["signal"].tolist(), \
        f"{name}: signals on past rows changed when future rows were added"
