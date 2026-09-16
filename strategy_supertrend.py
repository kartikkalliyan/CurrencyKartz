"""
strategy_supertrend.py
----------------------
ATR-based Supertrend strategy module.

Generates BUY signal when Supertrend direction switches from -1 (Bearish) to 1 (Bullish),
and SELL signal when direction switches from 1 (Bullish) to -1 (Bearish).
"""

import pandas as pd
from indicators import add_supertrend


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Supertrend_dir" not in df.columns:
        df = add_supertrend(df)

    prev_dir = df["Supertrend_dir"].shift(1)
    curr_dir = df["Supertrend_dir"]

    buy_mask = (prev_dir == -1) & (curr_dir == 1)
    sell_mask = (prev_dir == 1) & (curr_dir == -1)

    signals = pd.Series("HOLD", index=df.index)
    signals[buy_mask] = "BUY"
    signals[sell_mask] = "SELL"

    df["signal"] = signals
    return df
