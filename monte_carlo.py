"""
monte_carlo.py
--------------
Monte Carlo Strategy Stress-Tester & Value at Risk (VaR) Engine.
Shuffles historical trade returns with replacement to simulate 1,000 future equity paths.
"""

import numpy as np
import pandas as pd


def run_monte_carlo_simulation(backtest_results: dict, num_simulations: int = 1000, num_days: int = 100) -> dict:
    """
    Runs Monte Carlo simulation on backtest daily returns or trade log.

    Returns:
      - simulation_paths: DataFrame of shape (num_days, num_simulations)
      - percentiles: DataFrame with 5th (pessimistic), 50th (median), and 95th (optimistic) percentiles
      - var_95: 95% 1-day Value at Risk (%)
      - var_99: 99% 1-day Value at Risk (%)
      - ruin_probability_pct: % of paths that experienced >30% drawdown
    """
    df = backtest_results.get("df", None)
    initial_capital = backtest_results.get("initial_capital", 1000.0)

    if df is None or "equity" not in df.columns or len(df) < 5:
        # Fallback dummy simulation if empty backtest
        sim_data = np.full((num_days, num_simulations), initial_capital)
        return {
            "paths": pd.DataFrame(sim_data),
            "percentiles": pd.DataFrame({
                "5th (Pessimistic)": sim_data[:, 0],
                "50th (Median)": sim_data[:, 0],
                "95th (Optimistic)": sim_data[:, 0]
            }),
            "var_95": 0.0,
            "var_99": 0.0,
            "ruin_probability_pct": 0.0,
        }

    daily_returns = df["equity"].pct_change().dropna().values
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        daily_returns = np.array([0.0001, -0.0001, 0.0002])

    np.random.seed(42)
    # Sample daily returns randomly with replacement for num_simulations paths
    sampled_returns = np.random.choice(daily_returns, size=(num_days, num_simulations), replace=True)

    # Compute cumulative equity multiplier per simulation path
    equity_paths = initial_capital * np.cumprod(1.0 + sampled_returns, axis=0)
    # Prepend starting capital row
    start_row = np.full((1, num_simulations), initial_capital)
    equity_paths = np.vstack([start_row, equity_paths])

    paths_df = pd.DataFrame(equity_paths)

    # Calculate percentiles across simulations
    p5 = np.percentile(equity_paths, 5, axis=1)
    p50 = np.percentile(equity_paths, 50, axis=1)
    p95 = np.percentile(equity_paths, 95, axis=1)

    percentiles_df = pd.DataFrame({
        "5th (Pessimistic)": p5,
        "50th (Median)": p50,
        "95th (Optimistic)": p95,
    })

    # Value at Risk (VaR)
    var_95 = abs(np.percentile(daily_returns, 5)) * 100
    var_99 = abs(np.percentile(daily_returns, 1)) * 100

    # Calculate ruin probability (>30% drawdown along path)
    peak_per_path = np.maximum.accumulate(equity_paths, axis=0)
    dd_per_path = (equity_paths - peak_per_path) / peak_per_path
    ruin_count = np.sum(np.min(dd_per_path, axis=0) < -0.30)
    ruin_prob = (ruin_count / num_simulations) * 100

    return {
        "paths": paths_df,
        "percentiles": percentiles_df,
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "ruin_probability_pct": round(ruin_prob, 1),
    }
