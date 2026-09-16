"""
strategy_vwap.py
----------------
VWAP (Volume-Weighted Average Price) Crossover strategy module.

Generates BUY signal when close price crosses above VWAP line,
and SELL signal when close price crosses below VWAP line.
"""

import pandas as pd
from indicators import add_vwap


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "VWAP" not in df.columns:
        df = add_vwap(df)

    prev_close = df["close"].shift(1)
    prev_vwap = df["VWAP"].shift(1)
    curr_close = df["close"]
    curr_vwap = df["VWAP"]

    buy_mask = (prev_close <= prev_vwap) & (curr_close > curr_vwap)
    sell_mask = (prev_close >= prev_vwap) & (curr_close < curr_vwap)

    signals = pd.Series("HOLD", index=df.index)
    signals[buy_mask] = "BUY"
    signals[sell_mask] = "SELL"

    df["signal"] = signals
    return df
