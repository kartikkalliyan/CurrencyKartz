"""
Performance and risk metrics, computed from a backtest's equity curve.

Every formula here is a standard, well-established quant finance
definition -- nothing novel or experimental. Comments cite the
specific convention used where multiple conventions exist (e.g.
Sharpe ratio has a few common variants).
"""

import numpy as np
import pandas as pd


def compute_daily_returns(equity_curve):
    """Simple daily % returns from the equity curve."""
    return equity_curve.pct_change().dropna()


def cagr(equity_curve, periods_per_year=365):
    """
    Compound Annual Growth Rate -- the constant annual growth rate that
    would take the starting value to the ending value over the actual
    time elapsed. Uses periods_per_year=365 for daily crypto data
    (trades every calendar day) vs 252 for stock data (trading days only).
    """
    if len(equity_curve) < 2:
        return 0.0
    total_periods = len(equity_curve)
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    if start_value <= 0:
        return 0.0
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def annualized_volatility(equity_curve, periods_per_year=365):
    """Standard deviation of daily returns, scaled to an annual figure."""
    returns = compute_daily_returns(equity_curve)
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=365):
    """
    Sharpe ratio: excess return per unit of total volatility (both
    upside and downside movement count as "risk" here). risk_free_rate
    is annualized; defaults to 0 since retail traders rarely hold a
    risk-free alternative in the same currency/timeframe being compared.
    """
    returns = compute_daily_returns(equity_curve)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=365):
    """
    Sortino ratio: like Sharpe, but only penalizes DOWNSIDE volatility --
    a strategy with big up-swings and a smooth downside gets a better
    Sortino than Sharpe, which is often a fairer picture for trend-following
    strategies specifically (they're often lumpy on the upside by design).
    """
    returns = compute_daily_returns(equity_curve)
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    # Use a small epsilon rather than exact-zero: near-identical downside
    # values can produce a floating-point std of ~1e-17 instead of exactly
    # 0, which would otherwise blow the ratio up to an absurd magnitude
    # (found via testing -- see project test suite).
    MIN_MEANINGFUL_STD = 1e-8
    if len(downside_returns) == 0 or downside_returns.std() < MIN_MEANINGFUL_STD:
        return None  # not enough genuine downside variation to compute a meaningful ratio

    downside_std = downside_returns.std()
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve):
    """
    Maximum drawdown: the largest peak-to-trough decline, and how many
    periods it took to reach that trough from the prior peak. Returns
    (max_drawdown_pct, drawdown_duration_days).
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    trough_idx = drawdown.idxmin()
    peak_idx = equity_curve[:trough_idx].idxmax()
    duration = (trough_idx - peak_idx).days if hasattr(trough_idx - peak_idx, "days") else 0

    return abs(max_dd) * 100, duration


def calmar_ratio(equity_curve, periods_per_year=365):
    """Calmar ratio: CAGR divided by max drawdown -- return per unit of
    worst-case pain, rather than per unit of average volatility."""
    growth_rate = cagr(equity_curve, periods_per_year)
    dd_pct, _ = max_drawdown(equity_curve)
    if dd_pct == 0:
        return 0.0
    return growth_rate / (dd_pct / 100)


def win_rate_and_profit_factor(trade_log):
    """
    Computes win rate and profit factor from the trade log. Pairs each
    BUY with its following exit (SELL/STOP_LOSS_SELL/TRAILING_STOP_SELL)
    to determine per-trade profit/loss.
    """
    trades = []
    entry_price = None
    entry_dollars = None

    for t in trade_log:
        if t["action"] == "BUY":
            entry_price = t["price"]
            entry_dollars = t.get("position_dollars", None)
        elif t["action"] in ["SELL", "STOP_LOSS_SELL", "TRAILING_STOP_SELL"] and entry_price is not None:
            exit_price = t["price"]
            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append(pnl_pct)
            entry_price = None

    if len(trades) == 0:
        return {"win_rate_pct": None, "profit_factor": None, "num_completed_trades": 0,
               "avg_win_pct": None, "avg_loss_pct": None}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = len(wins) / len(trades) * 100
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    return {
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "num_completed_trades": len(trades),
        "avg_win_pct": (sum(wins) / len(wins) * 100) if wins else None,
        "avg_loss_pct": (sum(losses) / len(losses) * 100) if losses else None,
    }


def full_report(backtest_results, periods_per_year=365, risk_free_rate=0.0):
    """
    Convenience function: takes the results dict from backtest() and
    returns every metric in one place.
    """
    equity = backtest_results["equity_curve"]
    dd_pct, dd_duration = max_drawdown(equity)
    win_stats = win_rate_and_profit_factor(backtest_results["trade_log"])

    sortino = sortino_ratio(equity, risk_free_rate, periods_per_year)

    return {
        "CAGR %": round(cagr(equity, periods_per_year) * 100, 2),
        "Annualized Volatility %": round(annualized_volatility(equity, periods_per_year) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(equity, risk_free_rate, periods_per_year), 3),
        "Sortino Ratio": round(sortino, 3) if sortino is not None else None,
        "Calmar Ratio": round(calmar_ratio(equity, periods_per_year), 3),
        "Max Drawdown %": round(dd_pct, 2),
        "Max Drawdown Duration (days)": dd_duration,
        **win_stats,
    }