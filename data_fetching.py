"""
All Binance API interaction lives here -- fetching recent data, paginated
history for long lookbacks, and the "search until N trades found" logic.
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime


@st.cache_data(ttl=300)
def fetch_data(symbol, interval="1d", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "num_trades",
               "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"open_time": "timestamp"})

    df = df.iloc[:-1].reset_index(drop=True)  # drop the current, still-forming candle
    return df


@st.cache_data(ttl=300)
def fetch_data_paginated(symbol, days, interval="1d"):
    """
    Like fetch_data, but can pull MORE than 1000 candles by making
    multiple requests and stitching them together.
    """
    all_rows = []
    end_time_ms = int(datetime.now().timestamp() * 1000)
    earliest_needed = datetime.now().timestamp() - (days * 86400)

    while True:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "endTime": end_time_ms}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        all_rows = batch + all_rows
        oldest_time = batch[0][0]
        if (oldest_time / 1000) <= earliest_needed:
            break
        end_time_ms = oldest_time - 1

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "num_trades",
               "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"open_time": "timestamp"})
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df = df.iloc[:-1].reset_index(drop=True)
    return df


def find_data_for_trade_target(symbol, strategy_func, target_trades, prep_data_fn, max_days=3000):
    """
    Keeps pulling further back in history, doubling the window each
    time, until the chosen strategy has generated at least
    target_trades BUY/SELL signals (or we hit max_days / run out of
    available history).

    prep_data_fn is passed in (rather than imported) to avoid a
    circular import between this module and core_logic.py.
    """
    days = 300
    while True:
        df = fetch_data_paginated(symbol, days)
        prepped = prep_data_fn(df.copy())
        signal_df = strategy_func(prepped)
        num_trades = len(signal_df[signal_df["signal"] != "HOLD"])

        ran_out_of_history = len(df) < days * 0.9
        if num_trades >= target_trades or days >= max_days or ran_out_of_history:
            return signal_df, num_trades, days

        days = min(days * 2, max_days)
