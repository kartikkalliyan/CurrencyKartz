"""
Central configuration: asset list, strategy registry, and all the
selectable options used throughout the app.
"""

from strategy_v4_filtered_ma import generate_signals as ma_crossover_signals
from candlestick_strategy import generate_candlestick_signals
from strategy import generate_signals as rsi_signals
from strategy_bollinger import generate_signals as bollinger_signals
from strategy_macd import generate_signals as macd_signals
from strategy_breakout import generate_signals as breakout_signals
from strategy_buyhold import generate_signals as buyhold_signals
from strategy_ensemble import generate_signals as ensemble_signals
from strategy_supertrend import generate_signals as supertrend_signals
from strategy_vwap import generate_signals as vwap_signals
from strategy_sentiment import generate_signals as sentiment_signals
from strategy_custom_ensemble import generate_weighted_ensemble_signals as weighted_ensemble_signals
from indian_stocks import INDIAN_STOCK_OPTIONS

ASSET_OPTIONS = {
    "Bitcoin (BTC)":      "BTCUSDT",
    "Ethereum (ETH)":     "ETHUSDT",
    "Solana (SOL)":       "SOLUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "PAX Gold (PAXG)":    "PAXGUSDT",
    # Extended free pairs (all from Binance public API)
    "XRP (XRP)":          "XRPUSDT",
    "Dogecoin (DOGE)":    "DOGEUSDT",
    "Cardano (ADA)":      "ADAUSDT",
    "Avalanche (AVAX)":   "AVAXUSDT",
    "Polygon (MATIC)":    "MATICUSDT",
    "Chainlink (LINK)":   "LINKUSDT",
    "Polkadot (DOT)":     "DOTUSDT",
}

# Kept separate from ASSET_OPTIONS since these use a different data
# source (Yahoo Finance, not Binance) -- see indian_stocks.py
ASSET_CLASS_OPTIONS = {
    "Crypto (Binance)": "crypto",
    "Indian Stocks (NSE)": "stocks",
}

# Binance kline intervals for multi-timeframe analysis (all free)
INTERVAL_OPTIONS = {
    "1 minute":   "1m",
    "5 minutes":  "5m",
    "15 minutes": "15m",
    "1 hour":     "1h",
    "4 hours":    "4h",
    "1 day":      "1d",
    "1 week":     "1w",
}

STRATEGY_FUNCS = {
    "MA Crossover (regime-filtered)": ma_crossover_signals,
    "Supertrend Trend-Following": supertrend_signals,
    "VWAP Crossover": vwap_signals,
    "Sentiment-Filtered MA": sentiment_signals,
    "Weighted Custom Ensemble": weighted_ensemble_signals,
    "Candlestick + Trend Filter": generate_candlestick_signals,
    "RSI Mean-Reversion": rsi_signals,
    "Bollinger Bands": bollinger_signals,
    "MACD Crossover": macd_signals,
    "Volume-Confirmed Breakout": breakout_signals,
    "Buy & Hold (baseline)": buyhold_signals,
    "Ensemble Voting (3-of-6 agreement)": ensemble_signals,
}

RISK_STYLES = {
    "No stop-loss": {},
    "Fixed 10% stop-loss": {"stop_loss_pct": 0.10},
    "Trailing 10% stop": {"trailing_stop_pct": 0.10},
}

POSITION_SIZE_OPTIONS = {
    "All-in per trade": None,
    "Risk 1% per trade": 0.01,
    "Risk 2% per trade": 0.02,
    "Risk 5% per trade": 0.05,
}

SLIPPAGE_OPTIONS = {
    "None (unrealistic best case)": 0.0,
    "Low (0.1%, liquid major coins)": 0.001,
    "Medium (0.5%, typical altcoins)": 0.005,
    "High (1%, thin/illiquid markets)": 0.01,
}