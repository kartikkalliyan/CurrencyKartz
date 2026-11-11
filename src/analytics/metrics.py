"""
Performance & risk analytics module.

Standardized metrics used everywhere results get reported: single
backtests, walk-forward windows, portfolio-level results, and the
dashboard/report layer. Centralizing these here means every consumer
scores performance the same way (roadmap Section 2 / Section 10:
"Avoid duplicated strategy/backtest logic").

Conventions
-----------
- All "returns" inputs are periodic simple returns (e.g. daily),
  as a pandas Series indexed by date, NOT cumulative/equity values.
- `periods_per_year` lets the same functions work for daily (252),
  hourly, weekly, etc. data. Default is 252 (daily trading days).
- Functions return `np.nan` on insufficient/degenerate data rather
  than raising, so a walk-forward loop over many windows doesn't
  crash on a short or flat window -- callers can decide how to
  handle NaNs (e.g. exclude from aggregate stats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs/Infs so a single bad bar doesn't poison every metric."""
    r = pd.Series(returns).astype(float)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def equity_curve(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Compound periodic returns into a cumulative equity curve."""
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return starting_value * (1.0 + r).cumprod()


# ---------------------------------------------------------------------------
# Return / growth metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    n = len(r)
    if n == 0:
        return np.nan
    growth = (1.0 + r).prod()
    years = n / periods_per_year
    if years <= 0 or growth <= 0:
        return np.nan
    return float(growth ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    risk_free_rate is an ANNUAL rate (e.g. 0.04 for 4%); it's converted
    to a per-period rate internally so this works across daily/hourly/etc.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino ratio -- like Sharpe, but only penalizes
    downside deviation (returns below the risk-free/target rate).
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan  # no downside observed -- ratio is undefined, not infinite
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline, as a negative fraction (e.g. -0.23)."""
    curve = equity_curve(returns)
    if curve.empty:
        return np.nan
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """
    Longest stretch (in periods) spent below a prior equity peak,
    i.e. time-to-recovery for the worst drawdown. Returns 0 if the
    equity curve never dips below a prior peak.
    """
    curve = equity_curve(returns)
    if curve.empty:
        return 0
    running_max = curve.cummax()
    underwater = curve < running_max

    longest = 0
    current = 0
    for is_underwater in underwater:
        if is_underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR divided by the absolute value of max drawdown."""
    mdd = max_drawdown(returns)
    growth = cagr(returns, periods_per_year=periods_per_year)
    if mdd is None or np.isnan(mdd) or mdd == 0 or np.isnan(growth):
        return np.nan
    return float(growth / abs(mdd))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level, as a negative fraction
    of the period return distribution's lower tail (e.g. -0.05 means a
    5% chance of losing 5%+ in a single period).
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float(np.percentile(r, (1.0 - confidence) * 100))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Expected Shortfall / CVaR: average return in the tail beyond VaR.
    More informative than VaR alone since it captures tail severity,
    not just the threshold.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    var = value_at_risk(r, confidence=confidence)
    tail = r[r <= var]
    if tail.empty:
        return np.nan
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Trade-level metrics (operate on per-trade P&L, not periodic returns)
# ---------------------------------------------------------------------------

def win_rate(trade_pnls: pd.Series) -> float:
    t = _clean(trade_pnls)
    if t.empty:
        return np.nan
    return float((t > 0).sum() / len(t))


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss. >1 means winners outweigh losers in $."""
    t = _clean(trade_pnls)
    gross_profit = t[t > 0].sum()
    gross_loss = -t[t < 0].sum()
    if gross_loss == 0:
        return np.nan  # undefined -- no losing trades to divide by
    return float(gross_profit / gross_loss)


def average_win_loss(trade_pnls: pd.Series) -> dict:
    """Returns {'avg_win': ..., 'avg_loss': ..., 'win_loss_ratio': ...}."""
    t = _clean(trade_pnls)
    wins = t[t > 0]
    losses = t[t < 0]
    avg_win = float(wins.mean()) if not wins.empty else np.nan
    avg_loss = float(losses.mean()) if not losses.empty else np.nan
    if not losses.empty and avg_loss != 0 and not np.isnan(avg_loss) and not np.isnan(avg_win):
        ratio = abs(avg_win / avg_loss)
    else:
        ratio = np.nan
    return {"avg_win": avg_win, "avg_loss": avg_loss, "win_loss_ratio": ratio}


# ---------------------------------------------------------------------------
# Activity / exposure metrics
# ---------------------------------------------------------------------------

def turnover(position_changes: pd.Series) -> float:
    """
    Sum of absolute position changes over the period. Caller passes
    a series of per-period position deltas (e.g. weight_t - weight_{t-1}).
    """
    p = _clean(position_changes)
    if p.empty:
        return np.nan
    return float(p.abs().sum())


def exposure(positions: pd.Series) -> float:
    """Average absolute position size over the period (0 = flat, 1 = fully invested)."""
    p = _clean(positions)
    if p.empty:
        return np.nan
    return float(p.abs().mean())


# ---------------------------------------------------------------------------
# Rolling metrics (for dashboards / regime analysis)
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling annualized Sharpe over a trailing window (default ~1 quarter of daily bars)."""
    r = _clean(returns)
    if len(r) < window:
        return pd.Series(dtype=float)
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period

    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std(ddof=1)
    return (roll_mean / roll_std * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_volatility(returns: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    r = _clean(returns)
    if len(r) < window:
        return pd.Series(dtype=float)
    return (r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)).rename("rolling_volatility")


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    if curve.empty:
        return pd.Series(dtype=float)
    running_max = curve.cummax()
    return (curve / running_max - 1.0).rename("rolling_drawdown")


# ---------------------------------------------------------------------------
# Convenience: one-shot summary
# ---------------------------------------------------------------------------

def summarize(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute the standard roadmap metric set (Section 2) in one call.
    This is what walk_forward.py should call per window, and what the
    dashboard/report layer should call for any equity-curve result.
    """
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_duration": max_drawdown_duration(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year=periods_per_year),
        "value_at_risk_95": value_at_risk(returns, confidence=0.95),
        "expected_shortfall_95": expected_shortfall(returns, confidence=0.95),
        "n_periods": int(len(_clean(returns))),
    }
