"""
execution_slicer.py
-------------------
Institutional Trade Execution Slicing Simulator (TWAP & VWAP).
Simulates how institutional desks slice large parent orders into time-weighted (TWAP)
or volume-weighted (VWAP) micro child orders to minimize market impact and implementation shortfall.
"""

import numpy as np
import pandas as pd


class InstitutionalExecutionSlicer:
    """
    Simulates TWAP and VWAP algorithmic order execution strategies.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        parent_order_usd: float = 10000.0,
        num_slices: int = 10,
        side: str = "BUY",
        base_fee_rate: float = 0.00075,
        impact_coefficient: float = 0.00005  # Price impact factor per $1k volume
    ):
        """
        ohlcv_df: Historical or intraday kline data containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = ohlcv_df.copy().reset_index(drop=True)
        self.parent_order_usd = float(parent_order_usd)
        self.num_slices = max(2, min(int(num_slices), len(self.df)))
        self.side = side.upper()
        self.base_fee_rate = float(base_fee_rate)
        self.impact_coef = float(impact_coefficient)

        # Truncate data window to matching slice count
        self.slice_window = self.df.tail(self.num_slices).copy().reset_index(drop=True)
        self.arrival_price = float(self.slice_window.iloc[0]["open"])

    def simulate_twap(self) -> dict:
        """
        TWAP (Time-Weighted Average Price):
        Splits parent order equally into N slice buckets across time.
        """
        child_usd = self.parent_order_usd / self.num_slices
        fills = []
        cum_qty = 0.0
        cum_cost = 0.0

        for i, row in self.slice_window.iterrows():
            market_price = float(row["close"])
            market_vol = float(row.get("volume", 1000.0))

            # Market Impact Model: Larger child orders relative to bar volume increase slippage
            impact_bps = (child_usd / (market_vol * market_price + 1e-6)) * self.impact_coef * 10000.0
            impact_factor = (1.0 + (impact_bps / 10000.0)) if self.side == "BUY" else (1.0 - (impact_bps / 10000.0))

            exec_price = market_price * impact_factor
            fee = child_usd * self.base_fee_rate
            net_usd = child_usd - fee
            qty = net_usd / exec_price

            cum_qty += qty
            cum_cost += child_usd

            fills.append({
                "slice": i + 1,
                "timestamp": row.get("timestamp", f"Slice {i+1}"),
                "market_price": round(market_price, 4),
                "exec_price": round(exec_price, 4),
                "child_usd": round(child_usd, 2),
                "fill_qty": round(qty, 6),
                "impact_bps": round(impact_bps, 2),
                "fee": round(fee, 4)
            })

        fills_df = pd.DataFrame(fills)
        avg_exec_price = cum_cost / cum_qty if cum_qty > 0 else self.arrival_price
        
        # Implementation Shortfall in basis points (bps)
        if self.side == "BUY":
            shortfall_bps = ((avg_exec_price - self.arrival_price) / self.arrival_price) * 10000.0
        else:
            shortfall_bps = ((self.arrival_price - avg_exec_price) / self.arrival_price) * 10000.0

        return {
            "algo": "TWAP",
            "arrival_price": round(self.arrival_price, 4),
            "avg_exec_price": round(avg_exec_price, 4),
            "total_executed_usd": round(cum_cost, 2),
            "total_qty": round(cum_qty, 6),
            "shortfall_bps": round(shortfall_bps, 2),
            "total_fees": round(fills_df["fee"].sum(), 4),
            "fills_df": fills_df
        }

    def simulate_vwap(self) -> dict:
        """
        VWAP (Volume-Weighted Average Price):
        Slices parent order proportionally to historical volume in each slice bucket.
        """
        volumes = self.slice_window["volume"].values
        tot_vol = np.sum(volumes)
        if tot_vol <= 0:
            weights = np.full(self.num_slices, 1.0 / self.num_slices)
        else:
            weights = volumes / tot_vol

        fills = []
        cum_qty = 0.0
        cum_cost = 0.0

        for i, row in self.slice_window.iterrows():
            weight = weights[i]
            child_usd = self.parent_order_usd * weight
            market_price = float(row["close"])
            market_vol = float(row.get("volume", 1000.0))

            # Market Impact Model
            impact_bps = (child_usd / (market_vol * market_price + 1e-6)) * self.impact_coef * 10000.0
            impact_factor = (1.0 + (impact_bps / 10000.0)) if self.side == "BUY" else (1.0 - (impact_bps / 10000.0))

            exec_price = market_price * impact_factor
            fee = child_usd * self.base_fee_rate
            net_usd = child_usd - fee
            qty = net_usd / exec_price if exec_price > 0 else 0.0

            cum_qty += qty
            cum_cost += child_usd

            fills.append({
                "slice": i + 1,
                "timestamp": row.get("timestamp", f"Slice {i+1}"),
                "weight_pct": round(weight * 100.0, 2),
                "market_price": round(market_price, 4),
                "exec_price": round(exec_price, 4),
                "child_usd": round(child_usd, 2),
                "fill_qty": round(qty, 6),
                "impact_bps": round(impact_bps, 2),
                "fee": round(fee, 4)
            })

        fills_df = pd.DataFrame(fills)
        avg_exec_price = cum_cost / cum_qty if cum_qty > 0 else self.arrival_price

        if self.side == "BUY":
            shortfall_bps = ((avg_exec_price - self.arrival_price) / self.arrival_price) * 10000.0
        else:
            shortfall_bps = ((self.arrival_price - avg_exec_price) / self.arrival_price) * 10000.0

        return {
            "algo": "VWAP",
            "arrival_price": round(self.arrival_price, 4),
            "avg_exec_price": round(avg_exec_price, 4),
            "total_executed_usd": round(cum_cost, 2),
            "total_qty": round(cum_qty, 6),
            "shortfall_bps": round(shortfall_bps, 2),
            "total_fees": round(fills_df["fee"].sum(), 4),
            "fills_df": fills_df
        }
