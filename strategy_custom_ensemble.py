"""
strategy_custom_ensemble.py
----------------------------
Weighted Custom Strategy Ensemble Builder.
Combines multiple strategy models with custom percentage weights to generate a composite signal.
"""

import pandas as pd
from indicators import add_all_indicators
from regime_detector import add_regime
from strategy_v4_filtered_ma import generate_signals as ma_signals
from strategy_supertrend import generate_signals as supertrend_signals
from strategy_vwap import generate_signals as vwap_signals
from strategy import generate_signals as rsi_signals
from strategy_macd import generate_signals as macd_signals
from strategy_bollinger import generate_signals as bollinger_signals


def generate_weighted_ensemble_signals(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Generates composite signals based on weighted voting.

    weights: dict of strategy_name -> float weight (should sum to 1.0)
    Default: equal weighting across MA, Supertrend, VWAP, RSI, MACD, Bollinger.
    """
    df = df.copy()
    if "MA_20" not in df.columns:
        df = add_all_indicators(df)
        df = add_regime(df)

    if weights is None:
        weights = {
            "MA Crossover": 0.25,
            "Supertrend": 0.25,
            "VWAP": 0.20,
            "RSI": 0.10,
            "MACD": 0.10,
            "Bollinger": 0.10,
        }

    # Normalize weights
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # Calculate individual signals (+1 for BUY, -1 for SELL, 0 for HOLD)
    strat_map = {
        "MA Crossover": ma_signals,
        "Supertrend": supertrend_signals,
        "VWAP": vwap_signals,
        "RSI": rsi_signals,
        "MACD": macd_signals,
        "Bollinger": bollinger_signals,
    }

    score = pd.Series(0.0, index=df.index)

    for name, weight in weights.items():
        if weight > 0 and name in strat_map:
            sig_df = strat_map[name](df)
            val = sig_df["signal"].map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}).fillna(0.0)
            score += val * weight

    signals = pd.Series("HOLD", index=df.index)
    signals[score >= 0.25] = "BUY"
    signals[score <= -0.25] = "SELL"

    df["signal"] = signals
    df["ensemble_score"] = score
    return df
