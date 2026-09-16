"""
Core decision logic: turning raw price data into recommendations,
backtests, portfolio results, and strategy comparisons. No Streamlit
UI code lives here, only functions that take data in and return
results out.
"""

import pandas as pd
import streamlit as st

from indicators import add_all_indicators
from regime_detector import add_regime
from backtest import backtest
from config import STRATEGY_FUNCS, RISK_STYLES, ASSET_OPTIONS
from data_fetching import fetch_data
from indian_stocks import INDIAN_STOCK_OPTIONS, fetch_stock_data


def prep_data(df):
    """Always computes the full set of indicators + regime, regardless of
    which strategy is chosen."""
    df = add_all_indicators(df)
    df = add_regime(df)
    return df


def reason_for_signal(recommendation):
    if recommendation == "BUY":
        return "The selected strategy's entry conditions were met on the most recent closed candle."
    elif recommendation == "SELL":
        return "The selected strategy's exit conditions were met on the most recent closed candle."
    else:
        return "No new entry/exit condition triggered on the most recent closed candle."


def get_recommendation(df, strategy_func):
    signal_df = strategy_func(df.copy())
    latest = signal_df.iloc[-1]
    recommendation = latest["signal"]
    reason = reason_for_signal(recommendation)
    return recommendation, reason, latest, signal_df


def run_track_record(signal_df, risk_kwargs):
    results = backtest(signal_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)
    return results


def get_signal_history(signal_df, n=10):
    history = signal_df[signal_df["signal"] != "HOLD"].copy()
    history = history[["timestamp", "close", "regime", "signal"]].tail(n)
    history = history.rename(columns={
        "timestamp": "Date", "close": "Price", "regime": "Regime", "signal": "Signal"
    })
    history["Date"] = history["Date"].dt.date
    history["Price"] = history["Price"].round(2)
    return history.iloc[::-1]


def get_consensus(df):
    votes = {}
    for name, func in STRATEGY_FUNCS.items():
        try:
            sdf = func(df.copy())
            votes[name] = sdf.iloc[-1]["signal"]
        except Exception:
            votes[name] = "N/A"
    return votes


@st.cache_data(ttl=300)
def get_all_recommendations():
    """Overview for crypto assets (Binance)."""
    rows = []
    default_func = STRATEGY_FUNCS["MA Crossover (regime-filtered)"]
    for label, sym in ASSET_OPTIONS.items():
        try:
            df = fetch_data(sym)
            df = prep_data(df)
            rec, reason, latest, _ = get_recommendation(df, default_func)
            rows.append({
                "Asset": label, "Price": f"${latest['close']:.2f}",
                "Regime": latest["regime"], "Recommendation": rec,
            })
        except Exception as e:
            rows.append({"Asset": label, "Price": "N/A", "Regime": "N/A",
                         "Recommendation": f"Error: {e}"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_all_stock_recommendations():
    """Overview for Indian stocks (Yahoo Finance) -- kept as a separate
    cached function from the crypto version to avoid cache-hashing
    complications with passing dicts/functions as parameters."""
    rows = []
    default_func = STRATEGY_FUNCS["MA Crossover (regime-filtered)"]
    for label, sym in INDIAN_STOCK_OPTIONS.items():
        try:
            df = fetch_stock_data(sym)
            df = prep_data(df)
            rec, reason, latest, _ = get_recommendation(df, default_func)
            rows.append({
                "Asset": label, "Price": f"Rs. {latest['close']:.2f}",
                "Regime": latest["regime"], "Recommendation": rec,
            })
        except Exception as e:
            rows.append({"Asset": label, "Price": "N/A", "Regime": "N/A",
                         "Recommendation": f"Error: {e}"})
    return pd.DataFrame(rows)


def run_portfolio_backtest(asset_symbols, strategy_func, risk_kwargs, total_capital,
                            lookback_days=200, fetch_func=None):
    """
    Splits total_capital equally across the chosen assets, runs the same
    strategy+risk combo on each, and combines the results. fetch_func
    lets this work with either fetch_data (crypto) or fetch_stock_data
    (Indian stocks) -- defaults to crypto for backward compatibility.
    """
    if fetch_func is None:
        fetch_func = fetch_data

    per_asset_capital = total_capital / len(asset_symbols)
    rows = []
    total_final = 0.0
    total_buyhold = 0.0

    for sym in asset_symbols:
        try:
            df = fetch_func(sym, limit=lookback_days)
            df = prep_data(df)
            signal_df = strategy_func(df.copy())
            r = backtest(signal_df, starting_cash=per_asset_capital, fee_rate=0.001, **risk_kwargs)

            rows.append({
                "Asset": sym,
                "Allocated": f"{per_asset_capital:.2f}",
                "Final Value": f"{r['final_value_strategy']:.2f}",
                "Return %": round(r["strategy_return_pct"], 2),
                "Buy&Hold %": round(r["buy_hold_return_pct"], 2),
                "Trades": r["num_trades"],
            })
            total_final += r["final_value_strategy"]
            total_buyhold += r["final_value_buy_hold"]
        except Exception as e:
            rows.append({"Asset": sym, "Allocated": "N/A", "Final Value": f"Error: {e}",
                         "Return %": None, "Buy&Hold %": None, "Trades": None})

    portfolio_return_pct = (total_final / total_capital - 1) * 100
    buyhold_return_pct = (total_buyhold / total_capital - 1) * 100

    return {
        "rows": rows, "total_final": total_final,
        "portfolio_return_pct": portfolio_return_pct,
        "total_buyhold": total_buyhold, "buyhold_return_pct": buyhold_return_pct,
    }


def run_optimizer(symbol, lookback_days=200, fetch_func=None):
    """
    Tests every strategy x risk style combo on the same data and ranks
    them. fetch_func lets this work with either data source.
    """
    if fetch_func is None:
        fetch_func = fetch_data

    df = fetch_func(symbol, limit=lookback_days)
    df = prep_data(df)

    results = []
    for strat_name, strat_func in STRATEGY_FUNCS.items():
        try:
            signal_df = strat_func(df.copy())
        except Exception as e:
            results.append({"Strategy": strat_name, "Risk Style": "N/A", "Return %": None,
                            "Buy&Hold %": None, "Beat B&H?": f"Error: {e}", "Trades": None})
            continue

        for risk_name, risk_kwargs in RISK_STYLES.items():
            try:
                r = backtest(signal_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)
                results.append({
                    "Strategy": strat_name, "Risk Style": risk_name,
                    "Return %": round(r["strategy_return_pct"], 2),
                    "Buy&Hold %": round(r["buy_hold_return_pct"], 2),
                    "Beat B&H?": "YES" if r["strategy_return_pct"] > r["buy_hold_return_pct"] else "no",
                    "Trades": r["num_trades"],
                })
            except Exception as e:
                results.append({"Strategy": strat_name, "Risk Style": risk_name, "Return %": None,
                                "Buy&Hold %": None, "Beat B&H?": f"Error: {e}", "Trades": None})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Return %", ascending=False, na_position="last")
    return results_df


def run_comparative_backtest(df: pd.DataFrame, strategy_names: list, risk_kwargs: dict = None) -> dict:
    """
    Runs backtests for multiple strategies on the same prepped dataset
    and returns aligned equity curves for side-by-side plotting.
    """
    if risk_kwargs is None:
        risk_kwargs = {}

    comp_df = pd.DataFrame({"timestamp": df["timestamp"].values})
    metrics_summary = []

    if len(df) > 0 and "close" in df.columns:
        first_close = df.iloc[0]["close"]
        if first_close > 0:
            comp_df["Buy & Hold Baseline"] = 1000.0 * (1 - 0.001) * (df["close"].values / first_close)

    for name in strategy_names:
        if name in STRATEGY_FUNCS:
            s_func = STRATEGY_FUNCS[name]
            sig_df = s_func(df.copy())
            res = backtest(sig_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)

            comp_df[name] = res["equity_curve"].values

            metrics_summary.append({
                "Strategy": name,
                "Return %": round(res["strategy_return_pct"], 2),
                "Trades": res["num_trades"],
                "Final Equity ($)": round(res["final_value_strategy"], 2),
            })

    return {
        "equity_df": comp_df,
        "summary": pd.DataFrame(metrics_summary),
    }