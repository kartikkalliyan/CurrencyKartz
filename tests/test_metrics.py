"""
Unit tests for src/analytics/metrics.py.

These lock down two things the roadmap cares about most:
  1. Formulas are correct against hand-computed / known values.
  2. Degenerate inputs (empty, single-value, all-zero, no-losses,
     no-downside) return NaN instead of raising or silently lying --
     important because walk-forward will run this over many windows
     automatically, including short/flat/edge-case ones.
"""

import numpy as np
import pandas as pd
import pytest

try:
    # Roadmap's suggested structure: src/analytics/metrics.py
    from src.analytics.metrics import (
        equity_curve, total_return, cagr, annualized_volatility, sharpe_ratio,
        sortino_ratio, max_drawdown, max_drawdown_duration, calmar_ratio,
        value_at_risk, expected_shortfall, win_rate, profit_factor,
        average_win_loss, turnover, exposure, rolling_sharpe,
        rolling_volatility, rolling_drawdown, summarize,
    )
except ImportError:
    # Fallback: metrics.py sitting next to this test file / on the path
    from metrics import (
    equity_curve,
    total_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_duration,
    calmar_ratio,
    value_at_risk,
    expected_shortfall,
    win_rate,
    profit_factor,
    average_win_loss,
    turnover,
    exposure,
    rolling_sharpe,
    rolling_volatility,
    rolling_drawdown,
    summarize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns():
    """Zero return every period -- everything should degrade gracefully."""
    return pd.Series([0.0] * 10)


@pytest.fixture
def simple_up_returns():
    """Steady 1% gain every period, no volatility -> std = 0 -> Sharpe is NaN."""
    return pd.Series([0.01] * 50)


@pytest.fixture
def known_drawdown_returns():
    """
    Equity path: 1.00 -> 1.10 -> 0.99 -> 1.188
    i.e. +10%, -10%, +20%.
    Peak is 1.10, trough after peak is 0.99 -> drawdown = 0.99/1.10 - 1 = -0.10
    """
    return pd.Series([0.10, -0.10, 0.20])


# ---------------------------------------------------------------------------
# equity_curve / total_return / cagr
# ---------------------------------------------------------------------------

def test_equity_curve_compounds_correctly():
    r = pd.Series([0.10, 0.10])
    curve = equity_curve(r)
    assert curve.iloc[0] == pytest.approx(1.10)
    assert curve.iloc[1] == pytest.approx(1.21)


def test_equity_curve_empty_returns_empty_series():
    assert equity_curve(pd.Series(dtype=float)).empty


def test_total_return_known_value():
    r = pd.Series([0.10, 0.10])
    assert total_return(r) == pytest.approx(0.21)


def test_total_return_empty_is_nan():
    assert np.isnan(total_return(pd.Series(dtype=float)))


def test_cagr_known_value():
    # 10% total return over exactly 1 year (252 daily periods) -> CAGR == total return
    r = pd.Series([0.10] + [0.0] * 251)
    result = cagr(r, periods_per_year=252)
    assert result == pytest.approx(0.10, abs=1e-6)


def test_cagr_empty_is_nan():
    assert np.isnan(cagr(pd.Series(dtype=float)))


def test_cagr_negative_growth_handled():
    # Total wipeout -> growth factor <= 0 -> NaN, not a complex number
    r = pd.Series([-1.0])
    assert np.isnan(cagr(r))


# ---------------------------------------------------------------------------
# volatility / sharpe / sortino
# ---------------------------------------------------------------------------

def test_annualized_volatility_zero_for_constant_returns(simple_up_returns):
    assert annualized_volatility(simple_up_returns) == pytest.approx(0.0, abs=1e-9)


def test_annualized_volatility_insufficient_data_is_nan():
    assert np.isnan(annualized_volatility(pd.Series([0.01])))


def test_sharpe_zero_std_is_nan(simple_up_returns):
    # Constant positive returns -> zero volatility -> Sharpe undefined, not infinite
    assert np.isnan(sharpe_ratio(simple_up_returns))


def test_sharpe_positive_for_positive_mean_excess_returns():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(loc=0.001, scale=0.01, size=500))
    result = sharpe_ratio(r, risk_free_rate=0.0)
    assert result > 0


def test_sharpe_insufficient_data_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01])))


def test_sortino_no_downside_is_nan():
    # All-positive returns -> no downside deviation -> ratio undefined
    r = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino_ratio(r))


def test_sortino_penalizes_downside_only():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(loc=0.001, scale=0.01, size=500))
    sharpe = sharpe_ratio(r)
    sortino = sortino_ratio(r)
    # Not a strict mathematical law in general, but for roughly symmetric
    # noise around a small positive mean, Sortino should exceed Sharpe
    # since it only penalizes the downside half of the distribution.
    assert sortino > sharpe


# ---------------------------------------------------------------------------
# drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_known_value(known_drawdown_returns):
    assert max_drawdown(known_drawdown_returns) == pytest.approx(-0.10, abs=1e-6)


def test_max_drawdown_no_decline_is_zero(simple_up_returns):
    assert max_drawdown(simple_up_returns) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_empty_is_nan():
    assert np.isnan(max_drawdown(pd.Series(dtype=float)))


def test_max_drawdown_duration_counts_periods_underwater():
    # Peak at t=0, underwater for 3 periods, recovers at t=4
    r = pd.Series([0.0, -0.05, -0.01, 0.0, 0.10])
    assert max_drawdown_duration(r) == 3


def test_max_drawdown_duration_zero_when_always_at_peak():
    r = pd.Series([0.01, 0.01, 0.01])
    assert max_drawdown_duration(r) == 0


def test_calmar_ratio_known_relationship(known_drawdown_returns):
    result = calmar_ratio(known_drawdown_returns, periods_per_year=252)
    expected = cagr(known_drawdown_returns, periods_per_year=252) / abs(
        max_drawdown(known_drawdown_returns)
    )
    assert result == pytest.approx(expected)


def test_calmar_ratio_zero_drawdown_is_nan(simple_up_returns):
    assert np.isnan(calmar_ratio(simple_up_returns))


# ---------------------------------------------------------------------------
# VaR / Expected Shortfall
# ---------------------------------------------------------------------------

def test_value_at_risk_known_percentile():
    r = pd.Series(range(1, 101)) / 100.0  # 0.01 .. 1.00
    result = value_at_risk(r, confidence=0.95)
    assert result == pytest.approx(np.percentile(r, 5), abs=1e-6)


def test_expected_shortfall_worse_than_or_equal_to_var():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(loc=0.0, scale=0.02, size=1000))
    var = value_at_risk(r, confidence=0.95)
    es = expected_shortfall(r, confidence=0.95)
    assert es <= var  # tail average is at least as bad as the threshold


def test_var_empty_is_nan():
    assert np.isnan(value_at_risk(pd.Series(dtype=float)))


# ---------------------------------------------------------------------------
# trade-level metrics
# ---------------------------------------------------------------------------

def test_win_rate_known_value():
    t = pd.Series([10, -5, 20, -1, 0])  # 0 is neither win nor loss
    assert win_rate(t) == pytest.approx(2 / 5)


def test_win_rate_empty_is_nan():
    assert np.isnan(win_rate(pd.Series(dtype=float)))


def test_profit_factor_known_value():
    t = pd.Series([10, 10, -5, -5])
    assert profit_factor(t) == pytest.approx(20 / 10)


def test_profit_factor_no_losses_is_nan():
    t = pd.Series([10, 5, 2])
    assert np.isnan(profit_factor(t))


def test_average_win_loss_known_values():
    t = pd.Series([10, 20, -5, -15])
    result = average_win_loss(t)
    assert result["avg_win"] == pytest.approx(15.0)
    assert result["avg_loss"] == pytest.approx(-10.0)
    assert result["win_loss_ratio"] == pytest.approx(1.5)


def test_average_win_loss_no_losses():
    t = pd.Series([10, 20])
    result = average_win_loss(t)
    assert np.isnan(result["avg_loss"])
    assert np.isnan(result["win_loss_ratio"])


# ---------------------------------------------------------------------------
# turnover / exposure
# ---------------------------------------------------------------------------

def test_turnover_sums_absolute_changes():
    deltas = pd.Series([0.5, -0.3, 0.2])
    assert turnover(deltas) == pytest.approx(1.0)


def test_exposure_average_absolute_position():
    positions = pd.Series([1.0, -1.0, 0.5, -0.5])
    assert exposure(positions) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# rolling metrics
# ---------------------------------------------------------------------------

def test_rolling_sharpe_shorter_than_window_is_empty():
    r = pd.Series([0.01] * 5)
    assert rolling_sharpe(r, window=10).empty


def test_rolling_volatility_has_correct_length():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    result = rolling_volatility(r, window=20)
    assert len(result) == 100  # rolling keeps NaNs for the warm-up period, not drop them


def test_rolling_drawdown_matches_full_series_calc():
    r = pd.Series([0.10, -0.10, 0.20])
    rd = rolling_drawdown(r)
    assert rd.iloc[1] == pytest.approx(max_drawdown(r))


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------

def test_summarize_returns_all_expected_keys():
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 300))
    result = summarize(r)
    expected_keys = {
        "total_return", "cagr", "annualized_volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "max_drawdown_duration",
        "calmar_ratio", "value_at_risk_95", "expected_shortfall_95",
        "n_periods",
    }
    assert set(result.keys()) == expected_keys
    assert result["n_periods"] == 300


def test_summarize_empty_series_all_nan_but_no_crash():
    result = summarize(pd.Series(dtype=float))
    assert result["n_periods"] == 0
    assert np.isnan(result["sharpe_ratio"])
    assert np.isnan(result["cagr"])
