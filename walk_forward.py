"""
walk_forward.py
---------------
Walk-Forward Optimization & Overfitting Guard Engine.
Splits data into rolling Train (In-Sample) and Test (Out-of-Sample) windows to verify
strategy robustness and prevent curve-fitting.
"""

import pandas as pd
from indicators import add_all_indicators
from regime_detector import add_regime
from backtest import backtest


def _prep(df):
    return add_regime(add_all_indicators(df))


def run_walk_forward_analysis(df: pd.DataFrame, strategy_func, risk_kwargs: dict = None, train_days: int = 120, test_days: int = 40) -> dict:
    """
    Runs rolling Walk-Forward analysis across the dataset.

    Returns:
      - windows_df: DataFrame summarizing each rolling window's Train vs Test return
      - in_sample_avg_return: float
      - out_of_sample_avg_return: float
      - efficiency_ratio: float (Out-Of-Sample Return / In-Sample Return)
      - is_robust: bool
    """
    if risk_kwargs is None:
        risk_kwargs = {}

    total_rows = len(df)
    window_size = train_days + test_days

    if total_rows < window_size:
        return {
            "windows_df": pd.DataFrame(),
            "in_sample_avg_return": 0.0,
            "out_of_sample_avg_return": 0.0,
            "efficiency_ratio": 0.0,
            "is_robust": False,
        }

    results = []
    step_days = test_days   # slide window by test_days each step

    for start_idx in range(0, total_rows - window_size + 1, step_days):
        train_df = _prep(df.iloc[start_idx : start_idx + train_days].copy())
        test_df = _prep(df.iloc[start_idx + train_days : start_idx + window_size].copy())

        # In-Sample Backtest
        train_signals = strategy_func(train_df)
        train_bt = backtest(train_signals, **risk_kwargs)

        # Out-of-Sample Backtest
        test_signals = strategy_func(test_df)
        test_bt = backtest(test_signals, **risk_kwargs)

        results.append({
            "Window": f"Step {len(results)+1}",
            "Train Start": train_df.iloc[0]["timestamp"].strftime("%Y-%m-%d"),
            "Train End": train_df.iloc[-1]["timestamp"].strftime("%Y-%m-%d"),
            "Test End": test_df.iloc[-1]["timestamp"].strftime("%Y-%m-%d"),
            "In-Sample Return %": round(train_bt["strategy_return_pct"], 2),
            "Out-Sample Return %": round(test_bt["strategy_return_pct"], 2),
            "Out-Sample Trades": test_bt["num_trades"],
        })

    windows_df = pd.DataFrame(results)
    if len(windows_df) == 0:
        return {
            "windows_df": pd.DataFrame(),
            "in_sample_avg_return": 0.0,
            "out_of_sample_avg_return": 0.0,
            "efficiency_ratio": 0.0,
            "is_robust": False,
        }

    in_avg = windows_df["In-Sample Return %"].mean()
    out_avg = windows_df["Out-Sample Return %"].mean()
    efficiency = (out_avg / in_avg) if in_avg != 0 else 0.0
    is_robust = bool(out_avg > 0 and efficiency >= 0.5)

    return {
        "windows_df": windows_df,
        "in_sample_avg_return": round(in_avg, 2),
        "out_of_sample_avg_return": round(out_avg, 2),
        "efficiency_ratio": round(efficiency, 2),
        "is_robust": is_robust,
    }
