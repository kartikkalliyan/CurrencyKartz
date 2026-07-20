"""
Shared pytest fixtures -- reusable sample data so every test file doesn't
need to regenerate its own price data from scratch.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """
    A realistic-shaped OHLCV dataset (random walk with drift), large
    enough (300 rows) for all indicators (MA50, RSI14, etc.) to have
    valid values well before the end of the series.
    """
    np.random.seed(42)
    n = 300
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")

    returns = np.random.normal(loc=0.0005, scale=0.03, size=n)
    close = 30000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + np.random.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": close,
        "volume": np.random.uniform(1000, 5000, n)
    })
    return df


@pytest.fixture
def known_price_series():
    """
    A small, hand-crafted price series with a KNOWN, easy-to-verify
    shape: rises steadily, then falls steadily. Useful for testing
    indicators/strategies where we know exactly what should happen.
    """
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    up = np.linspace(100, 150, 30)
    down = np.linspace(150, 100, 30)
    close = np.concatenate([up, down])

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(60, 1000.0)
    })
    return df