"""
Core decision logic: turning raw price data into recommendations,
backtests, portfolio results, and strategy comparisons. This is the
"brain" of the app -- no Streamlit UI code lives here, only functions
that take data in and return results out (easier to test and reuse).
"""

import pandas as pd
import streamlit as st

from indicators import add_all_indicators
from regime_detector import add_regime
from backtest import backtest
from config import STRATEGY_FUNCS, RISK_STYLES, ASSET_OPTIONS
from data_fetching import fetch_data


def prep_data(df):
    """Always computes the full set of indicators + regime, regardless of
    which strategy is chosen -- so info like regime/RSI can be displayed
    even if the selected strategy doesn't itself use it."""
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
    """
    Applies the CHOSEN strategy's signal logic to the most recent
    closed candle and returns a recommendation + reasoning.
    """
    signal_df = strategy_func(df.copy())
    latest = signal_df.iloc[-1]
    recommendation = latest["signal"]
    reason = reason_for_signal(recommendation)
    return recommendation, reason, latest, signal_df


def run_track_record(signal_df, risk_kwargs):
    """
    Runs a full backtest over the fetched history using the already
    generated signals and the chosen risk style.
    """
    results = backtest(signal_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)
    return results


def get_signal_history(signal_df, n=10):
    """Returns the last n actual BUY/SELL signals (not HOLD) for display."""
    history = signal_df[signal_df["signal"] != "HOLD"].copy()
    history = history[["timestamp", "close", "regime", "signal"]].tail(n)
    history = history.rename(columns={
        "timestamp": "Date", "close": "Price", "regime": "Regime", "signal": "Signal"
    })
    history["Date"] = history["Date"].dt.date
    history["Price"] = history["Price"].round(2)
    return history.iloc[::-1]  # most recent first


def get_consensus(df):
    """
    Runs EVERY strategy on the same data and returns each one's current
    signal, so you can see how much agreement there is rather than
    trusting a single strategy's opinion in isolation.
    """
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
    """
    Fetches and computes a recommendation for every asset in
    ASSET_OPTIONS, using the default recommended combo (MA Crossover +
    Trailing stop).
    """
    rows = []
    default_func = STRATEGY_FUNCS["MA Crossover (regime-filtered)"]
    for label, sym in ASSET_OPTIONS.items():
        try:
            df = fetch_data(sym)
            df = prep_data(df)
            rec, reason, latest, _ = get_recommendation(df, default_func)
            rows.append({
                "Asset": label,
                "Price": f"${latest['close']:.2f}",
                "Regime": latest["regime"],
                "Recommendation": rec,
            })
        except Exception as e:
            rows.append({
                "Asset": label, "Price": "N/A", "Regime": "N/A",
                "Recommendation": f"Error: {e}"
            })
    return pd.DataFrame(rows)


def run_portfolio_backtest(asset_symbols, strategy_func, risk_kwargs, total_capital, lookback_days=200):
    """
    Splits total_capital equally across the chosen assets, runs the same
    strategy+risk combo on each, and combines the results into an
    overall portfolio return -- vs a blended buy-and-hold benchmark.
    """
    per_asset_capital = total_capital / len(asset_symbols)
    rows = []
    total_final = 0.0
    total_buyhold = 0.0

    for sym in asset_symbols:
        try:
            df = fetch_data(sym, limit=lookback_days)
            df = prep_data(df)
            signal_df = strategy_func(df.copy())
            r = backtest(signal_df, starting_cash=per_asset_capital, fee_rate=0.001, **risk_kwargs)

            rows.append({
                "Asset": sym,
                "Allocated": f"${per_asset_capital:.2f}",
                "Final Value": f"${r['final_value_strategy']:.2f}",
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
        "rows": rows,
        "total_final": total_final,
        "portfolio_return_pct": portfolio_return_pct,
        "total_buyhold": total_buyhold,
        "buyhold_return_pct": buyhold_return_pct,
    }


def run_optimizer(symbol, lookback_days=200):
    """
    Tests every strategy x risk style combo (21 total) on the same data
    and ranks them by return. NOTE: the top result is NOT automatically
    the "best" strategy going forward -- see README for why (overfitting).
    """
    df = fetch_data(symbol, limit=lookback_days)
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
                    "Strategy": strat_name,
                    "Risk Style": risk_name,
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
