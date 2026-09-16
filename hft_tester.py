"""
hft_tester.py
-------------
Zero-risk High-Frequency Trading (HFT) & Micro-Structure Virtual Tester.
Simulates Level-2 Order Book Imbalance (OBI) micro-scalping on real-time market data
with virtual capital and realistic exchange fee deductions.
"""

from datetime import datetime
import pandas as pd
import numpy as np


class HFTTesterBot:
    """
    Virtual High-Frequency Microstructure Test Bot.

    Simulates an automated market taker/scalper responding to real-time Level-2
    Order Book Imbalance (OBI) without risking real money.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        order_size_usd: float = 100.0,
        imbalance_threshold: float = 0.35,
        take_profit_pct: float = 0.0015,  # 0.15% (15 bps)
        stop_loss_pct: float = 0.0010,    # 0.10% (10 bps)
        fee_rate: float = 0.00075,        # 0.075% Binance taker fee tier
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.order_size_usd = float(order_size_usd)
        self.imbalance_threshold = float(imbalance_threshold)
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.fee_rate = float(fee_rate)

        # Position tracking
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.cost_basis = 0.0

        # Stats
        self.trade_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    def reset(self):
        """Resets virtual capital and trade statistics."""
        self.cash = self.initial_capital
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.cost_basis = 0.0
        self.trade_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    @staticmethod
    def calculate_obi(bids: list, asks: list, depth: int = 5) -> float:
        """
        Calculates Order Book Imbalance (OBI) across top depth levels:
          OBI = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
        Range: [-1.0 (pure sell pressure), +1.0 (pure buy pressure)].
        """
        if not bids or not asks:
            return 0.0

        # Extract quantities
        bid_vol = sum(float(b.get("qty", b.get("quantity", 0.0)) if isinstance(b, dict) else b[1]) for b in bids[:depth])
        ask_vol = sum(float(a.get("qty", a.get("quantity", 0.0)) if isinstance(a, dict) else a[1]) for a in asks[:depth])

        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return 0.0

        return (bid_vol - ask_vol) / total_vol

    def process_tick(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: float = 1.0,
        ask_qty: float = 1.0,
        timestamp: str = None,
    ) -> dict:
        """
        Processes a single live or simulated market tick.
        Evaluates micro-entry, take-profit, stop-loss, and signals.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]

        bid_price = float(bid_price)
        ask_price = float(ask_price)
        mid_price = (bid_price + ask_price) / 2.0
        spread = ask_price - bid_price
        spread_bps = (spread / mid_price) * 10000.0 if mid_price > 0 else 0.0

        total_vol = float(bid_qty) + float(ask_qty)
        obi = (float(bid_qty) - float(ask_qty)) / total_vol if total_vol > 0 else 0.0

        action_taken = "HOLD"
        trade_pnl = 0.0

        # 1. Evaluate Exit if currently holding a position
        if self.position_qty > 0:
            unrealized_return = (bid_price / self.entry_price - 1.0) if self.entry_price > 0 else 0.0

            is_tp = unrealized_return >= self.take_profit_pct
            is_sl = unrealized_return <= -self.stop_loss_pct
            is_reversal = obi <= -self.imbalance_threshold

            if is_tp or is_sl or is_reversal:
                exit_price = bid_price
                gross_proceeds = self.position_qty * exit_price
                fee = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fee
                trade_pnl = net_proceeds - self.cost_basis

                self.cash += net_proceeds
                self.total_fees += fee
                self.realized_pnl += trade_pnl
                self.total_trades += 1
                if trade_pnl > 0:
                    self.winning_trades += 1

                reason = "TAKE_PROFIT" if is_tp else ("STOP_LOSS" if is_sl else "SIGNAL_REVERSAL")
                action_taken = f"SELL ({reason})"

                self.trade_log.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "reason": reason,
                    "price": exit_price,
                    "qty": round(self.position_qty, 6),
                    "fee": round(fee, 4),
                    "pnl": round(trade_pnl, 4),
                    "pnl_pct": round(unrealized_return * 100.0, 2),
                    "obi_at_exit": round(obi, 3),
                })

                # Clear position
                self.position_qty = 0.0
                self.entry_price = 0.0
                self.entry_time = None
                self.cost_basis = 0.0

        # 2. Evaluate Entry if flat and cash available
        elif self.position_qty == 0 and self.cash >= self.order_size_usd:
            if obi >= self.imbalance_threshold:
                entry_price = ask_price
                gross_cost = self.order_size_usd
                fee = gross_cost * self.fee_rate
                net_invested = gross_cost - fee
                qty = net_invested / entry_price

                self.cash -= gross_cost
                self.total_fees += fee
                self.cost_basis = gross_cost
                self.position_qty = qty
                self.entry_price = entry_price
                self.entry_time = timestamp
                action_taken = "BUY (OBI_SPIKE)"

                self.trade_log.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "reason": "OBI_SPIKE",
                    "price": entry_price,
                    "qty": round(qty, 6),
                    "fee": round(fee, 4),
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "obi_at_entry": round(obi, 3),
                })

        # Calculate live equity
        unrealized_pnl = 0.0
        if self.position_qty > 0:
            current_market_val = self.position_qty * bid_price
            exit_fee_est = current_market_val * self.fee_rate
            unrealized_pnl = (current_market_val - exit_fee_est) - self.cost_basis

        current_equity = self.cash + (self.position_qty * bid_price)
        total_pnl = current_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100.0
        win_rate = (self.winning_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0

        return {
            "timestamp": timestamp,
            "mid_price": round(mid_price, 4),
            "spread_bps": round(spread_bps, 2),
            "obi": round(obi, 3),
            "action": action_taken,
            "current_equity": round(current_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_trades": self.total_trades,
            "win_rate": round(win_rate, 1),
            "total_fees": round(self.total_fees, 4),
            "in_position": self.position_qty > 0,
        }

    def run_replay_simulation(self, order_book: dict, recent_trades: list) -> pd.DataFrame:
        """
        Executes a rapid high-speed replay simulation using real-time Binance
        Level-2 order book depth and recent trade ticks.
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks or not recent_trades:
            return pd.DataFrame()

        if isinstance(bids[0], dict):
            base_bid = float(bids[0].get("price", 0.0))
        else:
            base_bid = float(bids[0][0])

        if isinstance(asks[0], dict):
            base_ask = float(asks[0].get("price", 0.0))
        else:
            base_ask = float(asks[0][0])
        obi = self.calculate_obi(bids, asks, depth=5)

        results = []
        for trade in recent_trades:
            t_price = float(trade.get("price", base_bid))
            t_qty = float(trade.get("qty", trade.get("quantity", 1.0)))
            is_buyer_maker = trade.get("is_buyer_maker", trade.get("side", "BUY") == "SELL")
            t_time = trade.get("time", "")

            # Synthetic micro-tick bid/ask around trade price
            spread_offset = (base_ask - base_bid) / 2.0 if base_ask > base_bid else t_price * 0.0001
            bid_p = t_price - spread_offset
            ask_p = t_price + spread_offset

            # Assign tick quantity to bid/ask depending on maker side
            b_qty = t_qty * 1.5 if is_buyer_maker else t_qty * 0.5
            a_qty = t_qty * 0.5 if is_buyer_maker else t_qty * 1.5

            tick_res = self.process_tick(bid_p, ask_p, bid_qty=b_qty, ask_qty=a_qty, timestamp=str(t_time))
            results.append(tick_res)

        return pd.DataFrame(results)
