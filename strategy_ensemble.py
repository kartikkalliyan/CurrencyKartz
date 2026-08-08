"""
ENSEMBLE VOTING strategy.

Runs all 7 existing strategies and only generates a BUY/SELL signal
when at least `min_votes` of them agree on the same direction. The
idea: a signal backed by multiple independent methods is more likely
to reflect something real, rather than one strategy's specific noise.

This is a genuine hypothesis worth testing, not an assumed improvement --
we validate it with the same rigor as everything else (train/test,
crash test, multi-asset) before trusting it.
"""

import pandas as pd

from strategy_v4_filtered_ma import generate_signals as ma_crossover_signals
from candlestick_strategy import generate_candlestick_signals
from strategy import generate_signals as rsi_signals
from strategy_bollinger import generate_signals as bollinger_signals
from strategy_macd import generate_signals as macd_signals
from strategy_breakout import generate_signals as breakout_signals

# Deliberately excludes Buy & Hold, since it always says BUY/HOLD and
# would just dilute the vote rather than add real information.
VOTING_STRATEGIES = [
    ma_crossover_signals, generate_candlestick_signals, rsi_signals,
    bollinger_signals, macd_signals, breakout_signals,
]

MIN_VOTES = 3  # out of 6 -- majority agreement required


def generate_signals(df, min_votes=MIN_VOTES):
    df = df.copy()

    all_signals = []
    for strat_func in VOTING_STRATEGIES:
        try:
            sdf = strat_func(df.copy())
            all_signals.append(sdf["signal"])
        except Exception:
            # If a strategy fails on this data, just skip its vote
            all_signals.append(pd.Series(["HOLD"] * len(df), index=df.index))

    votes_df = pd.concat(all_signals, axis=1)
    buy_votes = (votes_df == "BUY").sum(axis=1)
    sell_votes = (votes_df == "SELL").sum(axis=1)

    df["signal"] = "HOLD"
    df.loc[buy_votes >= min_votes, "signal"] = "BUY"
    df.loc[sell_votes >= min_votes, "signal"] = "SELL"
    df["buy_votes"] = buy_votes
    df["sell_votes"] = sell_votes

    return df


if __name__ == "__main__":
    from indicators import add_all_indicators
    from regime_detector import add_regime

    df = pd.read_csv("btc_daily_with_indicators.csv", parse_dates=["timestamp"])
    if "regime" not in df.columns:
        df = add_regime(df)

    df = generate_signals(df)
    trades = df[df["signal"] != "HOLD"]
    print(f"Total signals: {len(trades)}")
    print(trades[["timestamp", "close", "buy_votes", "sell_votes", "signal"]].to_string(index=False))