"""
strategy_sentiment.py
---------------------
Sentiment & Regime-Filtered Trading Strategy.

Applies a dual filter:
  - Technical MA20 / MA50 Golden Cross
  - RSI Oversold/Overbought filter (RSI < 55 for BUY to prevent buying top of extreme hype)
"""

import pandas as pd
from indicators import add_moving_averages, add_rsi


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MA_20" not in df.columns or "MA_50" not in df.columns:
        df = add_moving_averages(df)
    if "RSI" not in df.columns:
        df = add_rsi(df)

    prev_ma20 = df["MA_20"].shift(1)
    prev_ma50 = df["MA_50"].shift(1)
    curr_ma20 = df["MA_20"]
    curr_ma50 = df["MA_50"]
    rsi = df["RSI"]

    # Golden cross + non-overbought RSI (< 55)
    golden_cross = (prev_ma20 <= prev_ma50) & (curr_ma20 > curr_ma50) & (rsi < 55)

    # Death cross or overbought RSI (> 70)
    death_cross = (prev_ma20 >= prev_ma50) & (curr_ma20 < curr_ma50)

    signals = pd.Series("HOLD", index=df.index)
    signals[golden_cross] = "BUY"
    signals[death_cross] = "SELL"

    df["signal"] = signals
    return df
