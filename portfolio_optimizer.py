"""
portfolio_optimizer.py
----------------------
Modern Portfolio Theory (MPT) & Markowitz Efficient Frontier Engine.
Computes optimal risk-adjusted asset allocations using quadratic optimization
(SciPy SLSQP) and simulates 2,500 Monte Carlo portfolio weights.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzOptimizer:
    """
    Solves Markowitz Mean-Variance Optimization for Tangency (Max Sharpe)
    and Minimum Volatility Portfolios.
    """

    def __init__(self, asset_price_df: pd.DataFrame, risk_free_rate: float = 0.02, trading_days: int = 365):
        """
        asset_price_df: DataFrame with 'timestamp' or DatetimeIndex and asset close prices as columns.
        """
        self.prices = asset_price_df.copy()
        if "timestamp" in self.prices.columns:
            self.prices = self.prices.drop(columns=["timestamp"])

        # Drop any non-numeric columns
        self.prices = self.prices.select_dtypes(include=[np.number])
        self.assets = list(self.prices.columns)
        self.num_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

        # Daily percentage returns
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean() * self.trading_days
        self.cov_matrix = self.returns.cov() * self.trading_days

    def portfolio_performance(self, weights: np.ndarray) -> tuple:
        """
        Returns (annualized_return, annualized_volatility, sharpe_ratio).
        """
        weights = np.array(weights)
        port_return = np.sum(self.mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 1e-6 else 0.0
        return port_return, port_vol, sharpe

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        return -self.portfolio_performance(weights)[2]

    def _portfolio_vol(self, weights: np.ndarray) -> float:
        return self.portfolio_performance(weights)[1]

    def optimize_max_sharpe(self) -> dict:
        """Finds portfolio weights that maximize the Sharpe Ratio."""
        init_guess = self.num_assets * [1.0 / self.num_assets]
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt = minimize(
            self._neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        opt_weights = opt.x
        ret, vol, sharpe = self.portfolio_performance(opt_weights)
        weights_dict = {asset: round(float(w) * 100.0, 2) for asset, w in zip(self.assets, opt_weights)}

        return {
            "weights": weights_dict,
            "return_pct": round(float(ret) * 100.0, 2),
            "volatility_pct": round(float(vol) * 100.0, 2),
            "sharpe_ratio": round(float(sharpe), 2),
        }

    def optimize_min_volatility(self) -> dict:
        """Finds portfolio weights that minimize overall volatility."""
        init_guess = self.num_assets * [1.0 / self.num_assets]
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt = minimize(
            self._portfolio_vol,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        opt_weights = opt.x
        ret, vol, sharpe = self.portfolio_performance(opt_weights)
        weights_dict = {asset: round(float(w) * 100.0, 2) for asset, w in zip(self.assets, opt_weights)}

        return {
            "weights": weights_dict,
            "return_pct": round(float(ret) * 100.0, 2),
            "volatility_pct": round(float(vol) * 100.0, 2),
            "sharpe_ratio": round(float(sharpe), 2),
        }

    def generate_monte_carlo_frontier(self, num_portfolios: int = 2000) -> pd.DataFrame:
        """
        Simulates random portfolio allocations to plot the Markowitz cloud.
        """
        np.random.seed(42)
        results = []

        for _ in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            ret, vol, sharpe = self.portfolio_performance(weights)
            results.append({
                "Return %": round(float(ret) * 100.0, 2),
                "Volatility %": round(float(vol) * 100.0, 2),
                "Sharpe Ratio": round(float(sharpe), 2),
            })

        return pd.DataFrame(results)
