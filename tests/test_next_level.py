"""
tests/test_next_level.py
------------------------
Unit tests for Next-Level Quant modules:
  - genetic_optimizer.py (GeneticStrategyOptimizer)
  - portfolio_optimizer.py (MarkowitzOptimizer)
  - ml_model.py (MarketMLPredictor)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_optimizer import GeneticStrategyOptimizer
from portfolio_optimizer import MarkowitzOptimizer
from ml_model import MarketMLPredictor
from display_helpers import make_ga_fitness_chart, make_efficient_frontier_chart, make_feature_importance_chart


@pytest.fixture
def sample_price_history():
    """Generates synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    base = 100.0
    returns = np.random.normal(0.001, 0.02, size=120)
    prices = base * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 - 0.005),
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(1000, 5000, size=120)
    })
    return df


@pytest.fixture
def multi_asset_prices():
    """Generates multi-asset price matrix for MPT."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "BTC": 50000.0 * np.cumprod(1 + np.random.normal(0.001, 0.03, size=100)),
        "ETH": 3000.0 * np.cumprod(1 + np.random.normal(0.0008, 0.035, size=100)),
        "SOL": 120.0 * np.cumprod(1 + np.random.normal(0.0015, 0.045, size=100)),
    })
    return df


# ---------------------------------------------------------------------------
# 1. Genetic Algorithm Tests
# ---------------------------------------------------------------------------
class TestGeneticOptimizer:
    def test_ga_initialization(self, sample_price_history):
        ga = GeneticStrategyOptimizer(sample_price_history, population_size=12, generations=3)
        assert ga.population_size == 12
        assert ga.generations == 3

    def test_ga_chromosome_generation(self, sample_price_history):
        ga = GeneticStrategyOptimizer(sample_price_history)
        chrom = ga._random_chromosome()
        assert "fast_ma" in chrom
        assert "slow_ma" in chrom
        assert "stop_loss_pct" in chrom
        assert "trailing_stop_pct" in chrom
        assert 5 <= chrom["fast_ma"] <= 35
        assert 40 <= chrom["slow_ma"] <= 140

    def test_ga_optimization_runs_and_improves(self, sample_price_history):
        ga = GeneticStrategyOptimizer(sample_price_history, population_size=8, generations=3, random_seed=42)
        res = ga.optimize()
        assert "best_params" in res
        assert "best_fitness" in res
        assert "history_df" in res
        assert len(res["history_df"]) == 3
        assert res["best_params"]["fast_ma"] < res["best_params"]["slow_ma"]

    def test_ga_chart_renders(self):
        history_df = pd.DataFrame([
            {"generation": 1, "best_fitness": 0.5, "avg_fitness": 0.2},
            {"generation": 2, "best_fitness": 0.8, "avg_fitness": 0.4},
        ])
        fig = make_ga_fitness_chart(history_df)
        assert fig is not None
        assert len(fig.data) == 2


# ---------------------------------------------------------------------------
# 2. Modern Portfolio Theory (MPT) Tests
# ---------------------------------------------------------------------------
class TestMarkowitzOptimizer:
    def test_mpt_performance_calculation(self, multi_asset_prices):
        mpt = MarkowitzOptimizer(multi_asset_prices)
        assert mpt.num_assets == 3
        equal_weights = np.array([1/3, 1/3, 1/3])
        ret, vol, sharpe = mpt.portfolio_performance(equal_weights)
        assert vol > 0.0

    def test_mpt_max_sharpe_weights_sum_to_100(self, multi_asset_prices):
        mpt = MarkowitzOptimizer(multi_asset_prices)
        res = mpt.optimize_max_sharpe()
        assert "weights" in res
        assert "sharpe_ratio" in res
        total_weight = sum(res["weights"].values())
        assert pytest.approx(total_weight, abs=0.5) == 100.0

    def test_mpt_min_volatility_weights_sum_to_100(self, multi_asset_prices):
        mpt = MarkowitzOptimizer(multi_asset_prices)
        res = mpt.optimize_min_volatility()
        assert "volatility_pct" in res
        total_weight = sum(res["weights"].values())
        assert pytest.approx(total_weight, abs=0.5) == 100.0

    def test_mpt_monte_carlo_frontier_generates_dataframe(self, multi_asset_prices):
        mpt = MarkowitzOptimizer(multi_asset_prices)
        mc_df = mpt.generate_monte_carlo_frontier(num_portfolios=50)
        assert len(mc_df) == 50
        assert "Return %" in mc_df.columns
        assert "Volatility %" in mc_df.columns

    def test_efficient_frontier_chart_renders(self, multi_asset_prices):
        mpt = MarkowitzOptimizer(multi_asset_prices)
        mc_df = mpt.generate_monte_carlo_frontier(num_portfolios=30)
        max_s = mpt.optimize_max_sharpe()
        min_v = mpt.optimize_min_volatility()
        fig = make_efficient_frontier_chart(mc_df, max_s, min_v)
        assert fig is not None
        assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# 3. Machine Learning Predictor Tests
# ---------------------------------------------------------------------------
class TestMarketMLPredictor:
    def test_ml_prepare_features(self, sample_price_history):
        ml = MarketMLPredictor(sample_price_history)
        clean_df = ml._prepare_features()
        assert "target" in clean_df.columns
        assert "ret_1d" in clean_df.columns
        assert len(ml.feature_names) >= 4

    def test_ml_train_and_predict_returns_valid_probabilities(self, sample_price_history):
        ml = MarketMLPredictor(sample_price_history)
        res = ml.train_and_predict()
        assert "next_direction" in res
        assert "test_accuracy_pct" in res
        assert "bullish_prob_pct" in res
        assert "bearish_prob_pct" in res
        assert pytest.approx(res["bullish_prob_pct"] + res["bearish_prob_pct"], abs=1.0) == 100.0
        assert isinstance(res["feature_importance_df"], pd.DataFrame)

    def test_feature_importance_chart_renders(self):
        fi_df = pd.DataFrame({
            "Feature": ["RSI", "MACD_hist", "ret_1d"],
            "Importance %": [45.0, 35.0, 20.0]
        })
        fig = make_feature_importance_chart(fi_df)
        assert fig is not None
        assert len(fig.data) == 1
