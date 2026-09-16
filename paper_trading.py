"""
paper_trading.py
----------------
Paper Trading Simulation Engine for CurrencyKartz.
Manages a virtual cash balance, open positions, trade history, and PnL metrics.
Persists state cleanly using CSV files.
"""

import os
import pandas as pd
from datetime import datetime

ACCOUNT_FILE = "paper_account.csv"
TRADES_FILE = "paper_trades.csv"
DEFAULT_INITIAL_CASH = 10000.0


def init_paper_trading(initial_cash: float = DEFAULT_INITIAL_CASH, force_reset: bool = False):
    """Initializes the paper account if it does not exist."""
    if force_reset or not os.path.exists(ACCOUNT_FILE):
        df_acc = pd.DataFrame([{
            "initial_cash": initial_cash,
            "cash_balance": initial_cash,
            "realized_pnl": 0.0,
            "created_at": datetime.now()
        }])
        df_acc.to_csv(ACCOUNT_FILE, index=False)

    if force_reset or not os.path.exists(TRADES_FILE):
        df_trades = pd.DataFrame(columns=[
            "trade_id", "timestamp", "symbol", "action", "price", "qty",
            "position_value", "realized_pnl", "notes"
        ])
        df_trades.to_csv(TRADES_FILE, index=False)


def get_account_summary() -> dict:
    """Returns the current paper account state: cash, positions, realized PnL, total equity."""
    init_paper_trading()
    df_acc = pd.read_csv(ACCOUNT_FILE)
    cash = float(df_acc.iloc[0]["cash_balance"])
    initial_cash = float(df_acc.iloc[0]["initial_cash"])
    realized_pnl = float(df_acc.iloc[0]["realized_pnl"])

    positions = get_open_positions()
    total_position_val = sum(p["current_value"] for p in positions)
    unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)

    total_equity = cash + total_position_val
    total_pnl = total_equity - initial_cash
    total_pnl_pct = (total_pnl / initial_cash) * 100 if initial_cash > 0 else 0.0

    return {
        "initial_cash": initial_cash,
        "cash_balance": cash,
        "position_value": total_position_val,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "total_equity": total_equity,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "open_positions_count": len(positions),
    }


def get_paper_trades() -> pd.DataFrame:
    """Returns all recorded paper trades."""
    init_paper_trading()
    return pd.read_csv(TRADES_FILE)


def get_open_positions() -> list[dict]:
    """Calculates active open positions by netting completed BUY and SELL orders."""
    trades = get_paper_trades()
    if len(trades) == 0:
        return []

    positions = {}
    for _, row in trades.iterrows():
        sym = row["symbol"]
        action = row["action"]
        qty = float(row["qty"])
        price = float(row["price"])

        if sym not in positions:
            positions[sym] = {"symbol": sym, "qty": 0.0, "total_cost": 0.0}

        if action == "BUY":
            positions[sym]["qty"] += qty
            positions[sym]["total_cost"] += (qty * price)
        elif action == "SELL":
            if positions[sym]["qty"] > 0:
                avg_cost = positions[sym]["total_cost"] / positions[sym]["qty"]
                positions[sym]["qty"] -= qty
                positions[sym]["total_cost"] -= (qty * avg_cost)
                if positions[sym]["qty"] <= 0:
                    positions[sym]["qty"] = 0.0
                    positions[sym]["total_cost"] = 0.0

    # Build active list
    active_list = []
    for sym, data in positions.items():
        if data["qty"] > 0.000001:
            avg_entry = data["total_cost"] / data["qty"]
            # Estimate current price (can be updated with live spot price)
            current_price = avg_entry
            curr_val = data["qty"] * current_price
            unrealized = curr_val - data["total_cost"]
            unrealized_pct = (unrealized / data["total_cost"]) * 100 if data["total_cost"] > 0 else 0.0

            active_list.append({
                "symbol": sym,
                "qty": data["qty"],
                "avg_entry_price": avg_entry,
                "total_cost": data["total_cost"],
                "current_price": current_price,
                "current_value": curr_val,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": unrealized_pct,
            })

    return active_list


def execute_paper_order(symbol: str, action: str, price: float, amount_usd: float = None, qty: float = None, notes: str = "") -> dict:
    """
    Executes a paper market BUY or SELL order.

    - BUY: deducts USD from cash balance and increases position.
    - SELL: closes position, realizes PnL, and credits cash balance.
    """
    init_paper_trading()
    df_acc = pd.read_csv(ACCOUNT_FILE)
    cash = float(df_acc.iloc[0]["cash_balance"])
    realized = float(df_acc.iloc[0]["realized_pnl"])

    if price <= 0:
        return {"success": False, "error": "Price must be greater than 0."}

    action = action.upper()
    if action == "BUY":
        if amount_usd is None and qty is not None:
            amount_usd = qty * price
        elif amount_usd is None:
            return {"success": False, "error": "Must specify either amount_usd or qty."}

        if amount_usd > cash:
            return {"success": False, "error": f"Insufficient cash balance. Available: ${cash:,.2f}"}

        calc_qty = amount_usd / price
        new_cash = cash - amount_usd
        pnl = 0.0

    elif action == "SELL":
        open_pos = {p["symbol"]: p for p in get_open_positions()}
        if symbol not in open_pos or open_pos[symbol]["qty"] <= 0:
            return {"success": False, "error": f"No open position in {symbol} to sell."}

        pos_qty = open_pos[symbol]["qty"]
        avg_entry = open_pos[symbol]["avg_entry_price"]

        calc_qty = qty if (qty is not None and qty <= pos_qty) else pos_qty
        amount_usd = calc_qty * price
        cost_basis = calc_qty * avg_entry
        pnl = amount_usd - cost_basis

        new_cash = cash + amount_usd
        realized += pnl
    else:
        return {"success": False, "error": f"Invalid action: {action}"}

    # Update account state
    df_acc.at[0, "cash_balance"] = new_cash
    df_acc.at[0, "realized_pnl"] = realized
    df_acc.to_csv(ACCOUNT_FILE, index=False)

    # Log trade
    trades_df = get_paper_trades()
    trade_entry = {
        "trade_id": len(trades_df) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol.upper(),
        "action": action,
        "price": price,
        "qty": calc_qty,
        "position_value": amount_usd,
        "realized_pnl": pnl,
        "notes": notes,
    }
    trades_df = pd.concat([trades_df, pd.DataFrame([trade_entry])], ignore_index=True)
    trades_df.to_csv(TRADES_FILE, index=False)

    return {"success": True, "trade": trade_entry, "new_cash": new_cash}


def reset_paper_account(initial_cash: float = DEFAULT_INITIAL_CASH):
    """Resets paper account to initial state."""
    init_paper_trading(initial_cash=initial_cash, force_reset=True)
