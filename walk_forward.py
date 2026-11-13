"""
walk_forward.py
---------------
Walk-Forward Validation Engine.
Splits data into rolling Train (In-Sample) and Test (Out-of-Sample) windows to see
whether a strategy's results hold up on data it was not evaluated on before.

How indicators are handled (this matters):
    Indicators (MA50, regime, ...) are computed ONCE on the full history, then the
    resulting signals are sliced into windows. Computing them separately inside each
    short window leaves moving averages empty (a 40-day test window can never fill a
    50-day MA), so the strategy silently never trades. The indicators are all
    backward-looking, so computing them on the full history does not leak future data;
    tests/test_walk_forward.py checks this.

Note: the Train window here is evaluated with the SAME fixed parameters as the Test
window; nothing is optimised on it. "In-Sample" is therefore a baseline for
comparison, not the result of a tuning step.
"""

import pandas as pd
from indicators import add_all_indicators
from regime_detector import add_regime
from backtest import backtest

DEFAULT_WARMUP_ROWS = 50   # skip the first rows where MA50 / regime are still empty


def _prep(df):
    return add_regime(add_all_indicators(df))


def _empty_result():
    return {
        "windows_df": pd.DataFrame(),
        "in_sample_avg_return": 0.0,
        "out_of_sample_avg_return": 0.0,
        "efficiency_ratio": 0.0,
        "is_robust": False,
        "oos_windows_with_trades": 0,
    }


def run_walk_forward_analysis(df: pd.DataFrame, strategy_func, risk_kwargs: dict = None,
                              train_days: int = 120, test_days: int = 40,
                              warmup_rows: int = DEFAULT_WARMUP_ROWS) -> dict:
    """
    Runs rolling Walk-Forward analysis across the dataset.

    warmup_rows : leading rows skipped before the first window so indicators are
                  populated. Windows never start inside the warm-up period.

    Returns:
      - windows_df: DataFrame summarizing each rolling window's Train vs Test return
      - in_sample_avg_return: float
      - out_of_sample_avg_return: float
      - efficiency_ratio: float (Out-Of-Sample Return / In-Sample Return)
      - is_robust: bool (only True if out-of-sample actually traded)
      - oos_windows_with_trades: int (how many test windows contained any trade)
    """
    if risk_kwargs is None:
        risk_kwargs = {}

    window_size = train_days + test_days
    usable_rows = len(df) - warmup_rows

    if usable_rows < window_size:
        return _empty_result()

    # Indicators and signals are computed once on the full history (see module docstring).
    signals = strategy_func(_prep(df.copy())).reset_index(drop=True)

    results = []
    step_days = test_days   # slide window by test_days each step

    for start_idx in range(warmup_rows, len(signals) - window_size + 1, step_days):
        train_df = signals.iloc[start_idx : start_idx + train_days].reset_index(drop=True)
        test_df = signals.iloc[start_idx + train_days : start_idx + window_size].reset_index(drop=True)

        train_bt = backtest(train_df, **risk_kwargs)
        test_bt = backtest(test_df, **risk_kwargs)

        results.append({
            "Window": f"Step {len(results)+1}",
            "Train Start": train_df.iloc[0]["timestamp"].strftime("%Y-%m-%d"),
            "Train End": train_df.iloc[-1]["timestamp"].strftime("%Y-%m-%d"),
            "Test End": test_df.iloc[-1]["timestamp"].strftime("%Y-%m-%d"),
            "In-Sample Return %": round(train_bt["strategy_return_pct"], 2),
            "Out-Sample Return %": round(test_bt["strategy_return_pct"], 2),
            "Out-Sample Buy&Hold %": round(test_bt["buy_hold_return_pct"], 2),
            "Out-Sample Trades": test_bt["num_trades"],
        })

    windows_df = pd.DataFrame(results)
    if len(windows_df) == 0:
        return _empty_result()

    in_avg = windows_df["In-Sample Return %"].mean()
    out_avg = windows_df["Out-Sample Return %"].mean()
    efficiency = (out_avg / in_avg) if in_avg != 0 else 0.0
    oos_with_trades = int((windows_df["Out-Sample Trades"] > 0).sum())
    # "Robust" must be earned by actually trading out-of-sample, not by sitting in cash.
    is_robust = bool(oos_with_trades > 0 and out_avg > 0 and efficiency >= 0.5)

    return {
        "windows_df": windows_df,
        "in_sample_avg_return": round(in_avg, 2),
        "out_of_sample_avg_return": round(out_avg, 2),
        "efficiency_ratio": round(efficiency, 2),
        "is_robust": is_robust,
        "oos_windows_with_trades": oos_with_trades,
    }
