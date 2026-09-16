"""
PHASE 3: Technical indicators, written from scratch with pandas.

Writing these yourself (instead of importing a black-box library)
means you actually understand what "RSI = 30" means, instead of
just trusting a number.
"""

import pandas as pd


def add_moving_averages(df, windows=(20, 50, 200)):
    """Simple moving averages over the close price."""
    for w in windows:
        df[f"MA_{w}"] = df["close"].rolling(window=w).mean()
    return df


def add_rsi(df, period=14):
    """
    RSI (Relative Strength Index): measures speed/magnitude of recent
    price changes. Ranges 0-100.
      RSI > 70  -> often considered "overbought"
      RSI < 30  -> often considered "oversold"
    """
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence): difference between
    a fast and slow exponential moving average, plus a signal line.
    Common rule: MACD crossing above signal line = bullish momentum.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=period).mean()
    df["ATR_pct"] = df["ATR"] / df["close"]

    return df


def add_bollinger_bands(df, window=20, num_std=2):
    mid = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + num_std * std
    df["BB_lower"] = mid - num_std * std
    return df


def add_breakout_levels(df, window=20):
    df["High20"] = df["high"].rolling(window=window).max().shift(1)
    df["Low20"] = df["low"].rolling(window=window).min().shift(1)
    df["Vol_MA20"] = df["volume"].rolling(window=window).mean()
    return df


def add_vwap(df):
    """
    VWAP (Volume-Weighted Average Price):
    Calculates cumulative (Typical Price * Volume) / cumulative Volume.
    Typical Price = (High + Low + Close) / 3
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tpv = typical_price * df["volume"]
    cum_tpv = tpv.cumsum()
    cum_vol = df["volume"].cumsum()
    df["VWAP"] = cum_tpv / cum_vol
    return df


def add_supertrend(df, period=10, multiplier=3):
    """
    Supertrend Indicator:
    Calculates upper and lower trailing bands using ATR to determine trend direction.
    - Supertrend line follows trailing stop level.
    - Supertrend_dir = 1 (Bullish) or -1 (Bearish).
    """
    if "ATR" not in df.columns:
        df = add_atr(df, period=period)

    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + (multiplier * df["ATR"])
    basic_lower = hl2 - (multiplier * df["ATR"])

    n = len(df)
    upper_band = [0.0] * n
    lower_band = [0.0] * n
    trend = [1] * n
    supertrend = [0.0] * n

    for i in range(1, n):
        # Upper band logic
        if basic_upper.iloc[i] < upper_band[i-1] or df["close"].iloc[i-1] > upper_band[i-1]:
            upper_band[i] = basic_upper.iloc[i]
        else:
            upper_band[i] = upper_band[i-1]

        # Lower band logic
        if basic_lower.iloc[i] > lower_band[i-1] or df["close"].iloc[i-1] < lower_band[i-1]:
            lower_band[i] = basic_lower.iloc[i]
        else:
            lower_band[i] = lower_band[i-1]

        # Trend direction assignment
        if trend[i-1] == 1:
            if df["close"].iloc[i] < lower_band[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if df["close"].iloc[i] > upper_band[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend[i] = upper_band[i]

    df["Supertrend"] = supertrend
    df["Supertrend_dir"] = trend
    return df


def add_all_indicators(df):
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_breakout_levels(df)
    df = add_vwap(df)
    df = add_supertrend(df)
    return df
if __name__ == "__main__":
    df = pd.read_csv("btc_daily.csv", parse_dates=["timestamp"])
    df = add_all_indicators(df)
    df.to_csv("btc_daily_with_indicators.csv", index=False)
    print(df.tail())
