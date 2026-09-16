"""
tests/test_hft.py
-----------------
Unit tests for HFTTesterBot (hft_tester.py).
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_tester import HFTTesterBot


class TestHFTTesterBot:
    def test_init_and_reset(self):
        bot = HFTTesterBot(initial_capital=2000.0, order_size_usd=200.0)
        assert bot.cash == 2000.0
        assert bot.position_qty == 0.0
        assert bot.total_trades == 0

        # Simulate a fake state
        bot.cash = 1500.0
        bot.total_trades = 5
        bot.reset()
        assert bot.cash == 2000.0
        assert bot.total_trades == 0

    def test_calculate_obi(self):
        # Heavy bid volume
        bids = [{"price": 100.0, "qty": 10.0}]
        asks = [{"price": 100.1, "qty": 2.0}]
        obi = HFTTesterBot.calculate_obi(bids, asks)
        # (10 - 2) / (10 + 2) = 8 / 12 = 0.666...
        assert round(obi, 3) == 0.667

        # Heavy ask volume
        bids = [{"price": 100.0, "qty": 1.0}]
        asks = [{"price": 100.1, "qty": 9.0}]
        obi = HFTTesterBot.calculate_obi(bids, asks)
        # (1 - 9) / 10 = -0.8
        assert round(obi, 2) == -0.80

        # Empty inputs
        assert HFTTesterBot.calculate_obi([], []) == 0.0

    def test_process_tick_triggers_entry(self):
        bot = HFTTesterBot(initial_capital=1000.0, order_size_usd=100.0, imbalance_threshold=0.3)
        # Large bid_qty to trigger OBI > 0.3
        res = bot.process_tick(bid_price=50000.0, ask_price=50001.0, bid_qty=5.0, ask_qty=1.0)
        assert "BUY" in res["action"]
        assert bot.position_qty > 0
        assert bot.cash < 1000.0
        assert res["in_position"] is True

    def test_process_tick_take_profit(self):
        bot = HFTTesterBot(initial_capital=1000.0, order_size_usd=100.0, imbalance_threshold=0.3, take_profit_pct=0.002)
        # 1. Trigger Buy at 50,000
        bot.process_tick(bid_price=50000.0, ask_price=50000.0, bid_qty=5.0, ask_qty=1.0)
        assert bot.position_qty > 0

        # 2. Price surges by 0.5% (above 0.2% TP)
        res = bot.process_tick(bid_price=50250.0, ask_price=50251.0, bid_qty=1.0, ask_qty=1.0)
        assert "SELL (TAKE_PROFIT)" in res["action"]
        assert bot.position_qty == 0
        assert bot.total_trades == 1
        assert bot.winning_trades == 1
        assert len(bot.trade_log) == 2  # 1 BUY + 1 SELL

    def test_process_tick_stop_loss(self):
        bot = HFTTesterBot(initial_capital=1000.0, order_size_usd=100.0, imbalance_threshold=0.3, stop_loss_pct=0.001)
        # 1. Trigger Buy at 50,000
        bot.process_tick(bid_price=50000.0, ask_price=50000.0, bid_qty=5.0, ask_qty=1.0)
        assert bot.position_qty > 0

        # 2. Price drops by 0.3% (below 0.1% SL)
        res = bot.process_tick(bid_price=49850.0, ask_price=49851.0, bid_qty=1.0, ask_qty=1.0)
        assert "SELL (STOP_LOSS)" in res["action"]
        assert bot.position_qty == 0
        assert bot.total_trades == 1
        assert bot.winning_trades == 0

    def test_run_replay_simulation(self):
        bot = HFTTesterBot(initial_capital=1000.0)
        order_book = {
            "bids": [{"price": 100.0, "qty": 10.0}],
            "asks": [{"price": 100.1, "qty": 10.0}],
        }
        recent_trades = [
            {"price": 100.05, "qty": 1.5, "is_buyer_maker": True, "time": "12:00:00"},
            {"price": 100.10, "qty": 2.0, "is_buyer_maker": True, "time": "12:00:01"},
            {"price": 100.25, "qty": 3.0, "is_buyer_maker": False, "time": "12:00:02"},
        ]
        df_sim = bot.run_replay_simulation(order_book, recent_trades)
        assert isinstance(df_sim, pd.DataFrame)
        assert len(df_sim) == 3
        assert "current_equity" in df_sim.columns
        assert "obi" in df_sim.columns
