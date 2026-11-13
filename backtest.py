import pandas as pd

FILL_MODES = ("next_open", "same_close")


def backtest(df, starting_cash=1000.0, fee_rate=0.001, stop_loss_pct=None,
             trailing_stop_pct=None, trailing_stop_col=None, atr_multiplier=2.0,
             risk_per_trade_pct=None, slippage_pct=0.0, spread_pct=0.0,
             fill_mode="next_open"):
    """
    starting_cash      : hypothetical starting capital
    fee_rate           : trading fee per transaction (0.001 = 0.1%)
    stop_loss_pct      : fixed stop-loss from entry price. None = disabled.
    trailing_stop_pct  : FIXED trailing stop. None = disabled.
    trailing_stop_col  : column name (e.g. "ATR_pct") for a DYNAMIC trailing stop.
    atr_multiplier     : multiple of ATR_pct used when trailing_stop_col is set.
    risk_per_trade_pct : risk this fraction of capital per trade, sized by stop distance.
    slippage_pct       : additional price movement between decision and fill
                         (e.g. from order size vs available liquidity, or
                         latency). Distinct from spread -- this represents
                         the market MOVING against you, not the structural
                         cost of crossing the bid-ask gap.
    spread_pct         : the bid-ask spread itself, as a % of price (e.g.
                         0.002 for a 0.2% spread). Buys execute at the ask
                         (price + half the spread), sells execute at the
                         bid (price - half the spread) -- this cost exists
                         even with zero slippage and zero fees, since it's
                         a structural feature of the order book, not price
                         movement. Real for both crypto and equities;
                         typically larger for less liquid assets.
    fill_mode          : WHEN decisions are filled.
                         "next_open"  (default, realistic): every decision
                             (signal, stop-loss, trailing stop) is made using
                             the CLOSE of bar t and filled at the OPEN of bar
                             t+1. Requires an "open" column. A decision on the
                             very last bar never fills (there is no next bar).
                         "same_close" (legacy, optimistic): the decision and
                             the fill both happen at the close of bar t. This
                             assumes you could act on a close price at that
                             same close, so results are biased upward. Kept
                             only for comparison / backward compatibility.

    Returns a results dict including "equity_curve" -- a day-by-day
    mark-to-market portfolio value, needed for Sharpe/Sortino/volatility
    /drawdown (see metrics.py).
    """
    if fill_mode not in FILL_MODES:
        raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}")

    if fill_mode == "next_open" and "open" not in df.columns:
        raise ValueError("fill_mode='next_open' needs an 'open' column in the data. "
                         "Add it, or pass fill_mode='same_close' to use the legacy "
                         "(optimistic) same-bar-close fills.")

    if risk_per_trade_pct is not None and stop_loss_pct is None and \
       trailing_stop_pct is None and trailing_stop_col is None:
        raise ValueError("risk_per_trade_pct requires a stop_loss_pct, trailing_stop_pct, "
                          "or trailing_stop_col to be set -- position sizing needs a defined "
                          "stop distance to calculate how much to risk per trade.")

    state = {
        "cash": starting_cash,
        "btc_held": 0.0,
        "in_position": False,
        "entry_price": None,
        "peak_price": None,
    }
    half_spread = spread_pct / 2
    trade_log = []
    equity_curve = []

    def execute(order, raw_price, fill_date, signal_date):
        """Fill an order at raw_price (before spread/slippage/fees)."""
        action = order["action"]
        if action == "BUY":
            position_dollars = order["position_dollars"]
            if position_dollars is None:
                position_dollars = state["cash"]
            else:
                position_dollars = min(state["cash"], position_dollars)
            exec_price = raw_price * (1 + half_spread) * (1 + slippage_pct)
            btc_bought = (position_dollars * (1 - fee_rate)) / exec_price
            state["btc_held"] = btc_bought
            state["cash"] -= position_dollars
            state["in_position"] = True
            state["entry_price"] = raw_price
            state["peak_price"] = raw_price
            entry = {"date": fill_date, "action": "BUY", "price": exec_price,
                     "position_dollars": position_dollars,
                     "btc_held": state["btc_held"], "cash": state["cash"]}
        else:  # any SELL flavour
            exec_price = raw_price * (1 - half_spread) * (1 - slippage_pct)
            state["cash"] += (state["btc_held"] * exec_price) * (1 - fee_rate)
            state["btc_held"] = 0.0
            state["in_position"] = False
            state["entry_price"] = None
            state["peak_price"] = None
            entry = {"date": fill_date, "action": action, "price": exec_price,
                     "btc_held": state["btc_held"], "cash": state["cash"]}
        if fill_mode == "next_open":
            entry["signal_date"] = signal_date
        trade_log.append(entry)

    pending = None  # order decided on bar t, waiting to fill at the open of bar t+1

    for _, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        ts = row["timestamp"]

        # --- Fill yesterday's decision at today's OPEN (next_open mode only) ---
        if pending is not None:
            execute(pending, row["open"], ts, pending["decided_on"])
            pending = None

        if state["in_position"]:
            state["peak_price"] = max(state["peak_price"], price)

        decided_exit = False

        def place(order):
            """Fill now (legacy) or queue for the next bar's open (realistic)."""
            nonlocal pending
            order["decided_on"] = ts
            if fill_mode == "same_close":
                execute(order, price, ts, ts)
            else:
                pending = order

        # --- Stop-loss checks happen FIRST ---
        if state["in_position"] and stop_loss_pct is not None:
            drop_from_entry = (state["entry_price"] - price) / state["entry_price"]
            if drop_from_entry >= stop_loss_pct:
                place({"action": "STOP_LOSS_SELL"})
                decided_exit = True

        if not decided_exit and state["in_position"] and \
                (trailing_stop_pct is not None or trailing_stop_col is not None):
            if trailing_stop_col is not None:
                current_trailing_pct = row[trailing_stop_col] * atr_multiplier
            else:
                current_trailing_pct = trailing_stop_pct

            drop_from_peak = (state["peak_price"] - price) / state["peak_price"]
            if drop_from_peak >= current_trailing_pct:
                place({"action": "TRAILING_STOP_SELL"})
                decided_exit = True

        if not decided_exit:
            if signal == "BUY" and not state["in_position"] and pending is None:
                position_dollars = None  # None = all-in, sized at fill time
                if risk_per_trade_pct is not None:
                    if trailing_stop_col is not None:
                        stop_distance = row[trailing_stop_col] * atr_multiplier
                    elif trailing_stop_pct is not None:
                        stop_distance = trailing_stop_pct
                    else:
                        stop_distance = stop_loss_pct
                    position_dollars = (state["cash"] * risk_per_trade_pct) / stop_distance
                place({"action": "BUY", "position_dollars": position_dollars})

            elif signal == "SELL" and state["in_position"]:
                place({"action": "SELL"})

        mtm_value = state["cash"] + (state["btc_held"] * price)
        equity_curve.append({"date": ts, "equity": mtm_value})

    final_price = df.iloc[-1]["close"]
    final_value = state["cash"] + (state["btc_held"] * final_price)

    # Buy & hold benchmark follows the same fill rule as the strategy so the
    # comparison is fair: next_open buys at the first bar's open, legacy at its close.
    first_price = df.iloc[0]["open"] if fill_mode == "next_open" else df.iloc[0]["close"]
    buy_hold_btc = (starting_cash * (1 - fee_rate)) / first_price
    buy_hold_value = buy_hold_btc * final_price

    equity_df = pd.DataFrame(equity_curve).set_index("date")["equity"]

    return {
        "starting_cash": starting_cash,
        "final_value_strategy": final_value,
        "final_value_buy_hold": buy_hold_value,
        "strategy_return_pct": (final_value / starting_cash - 1) * 100,
        "buy_hold_return_pct": (buy_hold_value / starting_cash - 1) * 100,
        "num_trades": len(trade_log),
        "trade_log": trade_log,
        "equity_curve": equity_df,
        "fill_mode": fill_mode,
    }


def print_report(results):
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Fill mode:               {results.get('fill_mode', 'same_close')}")
    print(f"Starting capital:        ${results['starting_cash']:.2f}")
    print(f"Strategy final value:    ${results['final_value_strategy']:.2f}")
    print(f"Buy & hold final value:  ${results['final_value_buy_hold']:.2f}")
    print(f"Strategy return:         {results['strategy_return_pct']:.2f}%")
    print(f"Buy & hold return:       {results['buy_hold_return_pct']:.2f}%")
    print(f"Number of trades:        {results['num_trades']}")
    print("=" * 50)

    if results["strategy_return_pct"] > results["buy_hold_return_pct"]:
        print("Strategy BEAT buy-and-hold.")
    else:
        print("Strategy UNDERPERFORMED buy-and-hold.")


if __name__ == "__main__":
    df = pd.read_csv("btc_with_signals.csv", parse_dates=["timestamp"])
    results = backtest(df, starting_cash=1000.0, fee_rate=0.001)
    print_report(results)
