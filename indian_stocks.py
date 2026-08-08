"""
Indian stock (NSE) data fetching via Yahoo Finance.

This is a SECOND data source alongside Binance -- Binance only has
crypto pairs, so Indian equities need a different source entirely.
yfinance is free, needs no API key, and NSE tickers just need a
".NS" suffix (e.g. "IDEA.NS" for Vodafone Idea).

The output schema (timestamp, open, high, low, close, volume) matches
fetch_data() from data_fetching.py exactly, so every existing
indicator, strategy, and backtest function works unchanged regardless
of which data source was used.
"""

import yfinance as yf
import pandas as pd
import streamlit as st

INDIAN_STOCK_OPTIONS = {
    "Vodafone Idea": "IDEA.NS",
    "PC Jeweller": "PCJEWELLER.NS",
    "Samvardhana Motherson International": "MOTHERSON.NS",
    "Allcargo Logistics": "ALLCARGO.NS",
    "Suzlon Energy": "SUZLON.NS",
    "Electronics Mart India": "EMIL.NS",
    "Reliance Power": "RPOWER.NS",
    "GTL Infrastructure": "GTLINFRA.NS",
    "State Bank of India": "SBIN.NS",
    "LIC of India": "LICI.NS",
    "Kalyan Jewellers": "KALYANKJIL.NS",
    "Yes Bank": "YESBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}


@st.cache_data(ttl=300)
def fetch_stock_data(symbol, limit=200):
    """
    Fetches daily OHLCV data for an NSE stock from Yahoo Finance,
    returned in the SAME schema as fetch_data() from Binance --
    timestamp, open, high, low, close, volume (all lowercase) --
    so every downstream function works identically either way.

    limit is treated as "number of days" here, same as the Binance
    version, even though yfinance itself works with a period string.
    """
    period_days = max(limit + 10, 30)  # small buffer for weekends/holidays
    raw = yf.download(symbol, period=f"{period_days}d", interval="1d",
                      progress=False, auto_adjust=True)

    if raw is None or len(raw) == 0:
        raise ValueError(f"No data returned for '{symbol}' -- check the ticker is correct "
                         f"(NSE tickers need a .NS suffix, e.g. 'SBIN.NS').")

    # Newer yfinance versions return MultiIndex columns even for a
    # single ticker -- flatten them if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df = df.rename(columns={
        "Date": "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.tail(limit).reset_index(drop=True)