import pandas as pd


def backtest(df, starting_cash=1000.0, fee_rate=0.001, stop_loss_pct=None,
             trailing_stop_pct=None, trailing_stop_col=None, atr_multiplier=2.0,
             risk_per_trade_pct=None, slippage_pct=0.0, spread_pct=0.0):
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

    Returns a results dict including "equity_curve" -- a day-by-day
    mark-to-market portfolio value, needed for Sharpe/Sortino/volatility
    /drawdown (see metrics.py).
    """
    if risk_per_trade_pct is not None and stop_loss_pct is None and \
       trailing_stop_pct is None and trailing_stop_col is None:
        raise ValueError("risk_per_trade_pct requires a stop_loss_pct, trailing_stop_pct, "
                          "or trailing_stop_col to be set -- position sizing needs a defined "
                          "stop distance to calculate how much to risk per trade.")

    cash = starting_cash
    btc_held = 0.0
    in_position = False
    entry_price = None
    peak_price = None
    half_spread = spread_pct / 2

    trade_log = []
    equity_curve = []

    for _, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        exited_this_row = False

        if in_position:
            peak_price = max(peak_price, price)

        # --- Stop-loss checks happen FIRST ---
        if in_position and stop_loss_pct is not None:
            drop_from_entry = (entry_price - price) / entry_price
            if drop_from_entry >= stop_loss_pct:
                exec_price = price * (1 - half_spread) * (1 - slippage_pct)
                cash += (btc_held * exec_price) * (1 - fee_rate)
                btc_held = 0.0
                in_position = False
                entry_price = None
                peak_price = None
                exited_this_row = True
                trade_log.append({
                    "date": row["timestamp"], "action": "STOP_LOSS_SELL",
                    "price": exec_price, "btc_held": btc_held, "cash": cash
                })

        if not exited_this_row and in_position and (trailing_stop_pct is not None or trailing_stop_col is not None):
            if trailing_stop_col is not None:
                current_trailing_pct = row[trailing_stop_col] * atr_multiplier
            else:
                current_trailing_pct = trailing_stop_pct

            drop_from_peak = (peak_price - price) / peak_price
            if drop_from_peak >= current_trailing_pct:
                exec_price = price * (1 - half_spread) * (1 - slippage_pct)
                cash += (btc_held * exec_price) * (1 - fee_rate)
                btc_held = 0.0
                in_position = False
                entry_price = None
                peak_price = None
                exited_this_row = True
                trade_log.append({
                    "date": row["timestamp"], "action": "TRAILING_STOP_SELL",
                    "price": exec_price, "btc_held": btc_held, "cash": cash
                })

        if not exited_this_row:
            if signal == "BUY" and not in_position:
                if risk_per_trade_pct is not None:
                    if trailing_stop_col is not None:
                        stop_distance = row[trailing_stop_col] * atr_multiplier
                    elif trailing_stop_pct is not None:
                        stop_distance = trailing_stop_pct
                    else:
                        stop_distance = stop_loss_pct

                    risk_amount = cash * risk_per_trade_pct
                    position_dollars = min(cash, risk_amount / stop_distance)
                else:
                    position_dollars = cash

                exec_price = price * (1 + half_spread) * (1 + slippage_pct)
                btc_bought = (position_dollars * (1 - fee_rate)) / exec_price
                btc_held = btc_bought
                cash -= position_dollars
                in_position = True
                entry_price = price
                peak_price = price
                trade_log.append({
                    "date": row["timestamp"], "action": "BUY",
                    "price": exec_price, "position_dollars": position_dollars,
                    "btc_held": btc_held, "cash": cash
                })

            elif signal == "SELL" and in_position:
                exec_price = price * (1 - half_spread) * (1 - slippage_pct)
                cash += (btc_held * exec_price) * (1 - fee_rate)
                btc_held = 0.0
                in_position = False
                entry_price = None
                peak_price = None
                trade_log.append({
                    "date": row["timestamp"], "action": "SELL",
                    "price": exec_price, "btc_held": btc_held, "cash": cash
                })

        mtm_value = cash + (btc_held * price)
        equity_curve.append({"date": row["timestamp"], "equity": mtm_value})

    final_price = df.iloc[-1]["close"]
    final_value = cash + (btc_held * final_price)

    first_price = df.iloc[0]["close"]
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
    }


def print_report(results):
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
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