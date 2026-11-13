"""
ENSEMBLE VOTING strategy.

Runs 6 of the existing strategies and goes long only when at least `min_votes` of
them are CURRENTLY long. The idea: a position backed by multiple independent methods
is more likely to reflect something real, rather than one strategy's specific noise.

Why vote on stance instead of on signals: strategies emit BUY/SELL only on the single
bar where their condition flips, so requiring several strategies to flip on the very
same day almost never happens (on 6 years of BTC the old version produced 0 BUYs and
1 SELL, i.e. it never traded). Each strategy's signals are therefore turned into a
stance first -- long after its BUY, flat after its SELL, unchanged on HOLD -- and the
vote is taken on those stances. The ensemble emits BUY when the long-vote count
reaches `min_votes` and SELL when it drops back below it.

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


def _stance(signal: pd.Series) -> pd.Series:
    """1 while a strategy is long (since its last BUY), 0 after its SELL, 0 before any signal."""
    state = signal.map({"BUY": 1.0, "SELL": 0.0})   # HOLD -> NaN
    return state.ffill().fillna(0.0)


def generate_signals(df, min_votes=MIN_VOTES):
    df = df.copy()

    stances = []
    for strat_func in VOTING_STRATEGIES:
        try:
            sdf = strat_func(df.copy())
            stances.append(_stance(sdf["signal"]))
        except Exception:
            # If a strategy fails on this data, it simply votes "not long"
            stances.append(pd.Series(0.0, index=df.index))

    long_votes = pd.concat(stances, axis=1).sum(axis=1).astype(int)
    want_long = long_votes >= min_votes
    was_long = want_long.shift(1, fill_value=False)

    df["signal"] = "HOLD"
    df.loc[want_long & ~was_long, "signal"] = "BUY"
    df.loc[~want_long & was_long, "signal"] = "SELL"
    df["buy_votes"] = long_votes
    df["sell_votes"] = len(VOTING_STRATEGIES) - long_votes

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