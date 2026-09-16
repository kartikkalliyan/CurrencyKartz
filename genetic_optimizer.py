"""
genetic_optimizer.py
--------------------
Evolutionary Genetic Algorithm (GA) Strategy Auto-Tuner.
Evolves strategy parameters (fast/slow spans, stop-loss, trailing stops)
over multiple generations using natural selection, crossover, and mutation
to maximize risk-adjusted returns (Sharpe Ratio).
"""

import random
import numpy as np
import pandas as pd
from backtest import backtest


class GeneticStrategyOptimizer:
    """
    Genetic Algorithm for automated parameter discovery and hyperparameter tuning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        population_size: int = 24,
        generations: int = 10,
        mutation_rate: float = 0.20,
        tournament_size: int = 3,
        random_seed: int = 42,
    ):
        self.df = df.copy()
        self.population_size = max(8, population_size)
        self.generations = max(2, generations)
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.random_seed = random_seed

        # Parameter Search Space Bounds
        self.param_bounds = {
            "fast_ma": (5, 35),
            "slow_ma": (40, 140),
            "stop_loss_pct": (0.02, 0.15),
            "trailing_stop_pct": (0.02, 0.15),
        }

    def _random_chromosome(self) -> dict:
        return {
            "fast_ma": random.randint(self.param_bounds["fast_ma"][0], self.param_bounds["fast_ma"][1]),
            "slow_ma": random.randint(self.param_bounds["slow_ma"][0], self.param_bounds["slow_ma"][1]),
            "stop_loss_pct": round(random.uniform(self.param_bounds["stop_loss_pct"][0], self.param_bounds["stop_loss_pct"][1]), 4),
            "trailing_stop_pct": round(random.uniform(self.param_bounds["trailing_stop_pct"][0], self.param_bounds["trailing_stop_pct"][1]), 4),
        }

    def _evaluate_fitness(self, chrom: dict) -> tuple:
        """
        Evaluates an individual chromosome by running a fast backtest.
        Fitness = Annualized Sharpe Ratio penalized for excessive drawdown (>25%).
        """
        fast = int(chrom["fast_ma"])
        slow = int(chrom["slow_ma"])
        if fast >= slow:
            return -5.0, {}

        df_eval = self.df.copy()
        df_eval["fast_ma"] = df_eval["close"].rolling(fast).mean()
        df_eval["slow_ma"] = df_eval["close"].rolling(slow).mean()

        # Generate MA crossover signals
        prev_fast = df_eval["fast_ma"].shift(1)
        prev_slow = df_eval["slow_ma"].shift(1)
        buy_cond = (df_eval["fast_ma"] > df_eval["slow_ma"]) & (prev_fast <= prev_slow)
        sell_cond = (df_eval["fast_ma"] < df_eval["slow_ma"]) & (prev_fast >= prev_slow)

        signals = np.full(len(df_eval), "HOLD", dtype=object)
        signals[buy_cond.fillna(False)] = "BUY"
        signals[sell_cond.fillna(False)] = "SELL"
        df_eval["signal"] = signals

        try:
            res = backtest(
                df_eval,
                starting_cash=1000.0,
                fee_rate=0.001,
                stop_loss_pct=chrom["stop_loss_pct"],
                trailing_stop_pct=chrom["trailing_stop_pct"],
            )
            ret_pct = res["strategy_return_pct"]
            num_trades = res["num_trades"]

            # Compute returns and Sharpe
            eq_curve = res["equity_curve"].values
            if len(eq_curve) > 1 and num_trades > 0:
                daily_rets = np.diff(eq_curve) / eq_curve[:-1]
                std = np.std(daily_rets)
                sharpe = (np.mean(daily_rets) / std * np.sqrt(365)) if std > 1e-6 else 0.0

                # Max drawdown penalty
                peak = np.maximum.accumulate(eq_curve)
                dd = (peak - eq_curve) / peak
                max_dd = np.max(dd) if len(dd) > 0 else 0.0

                penalty = max(0.0, max_dd - 0.25) * 3.0
                fitness = sharpe - penalty
            else:
                fitness = -2.0
                ret_pct = 0.0
                sharpe = 0.0
                max_dd = 0.0

            metrics = {
                "return_pct": round(ret_pct, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(max_dd * 100.0, 2),
                "trades": num_trades,
            }
            return fitness, metrics
        except Exception:
            return -5.0, {}

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """Uniform crossover between two parents."""
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def _mutate(self, chrom: dict) -> dict:
        """Gaussian / uniform mutation bounded by parameter limits."""
        mutated = chrom.copy()
        if random.random() < self.mutation_rate:
            mutated["fast_ma"] = random.randint(self.param_bounds["fast_ma"][0], self.param_bounds["fast_ma"][1])
        if random.random() < self.mutation_rate:
            mutated["slow_ma"] = random.randint(self.param_bounds["slow_ma"][0], self.param_bounds["slow_ma"][1])
        if random.random() < self.mutation_rate:
            mutated["stop_loss_pct"] = round(random.uniform(self.param_bounds["stop_loss_pct"][0], self.param_bounds["stop_loss_pct"][1]), 4)
        if random.random() < self.mutation_rate:
            mutated["trailing_stop_pct"] = round(random.uniform(self.param_bounds["trailing_stop_pct"][0], self.param_bounds["trailing_stop_pct"][1]), 4)

        if mutated["fast_ma"] >= mutated["slow_ma"]:
            mutated["slow_ma"] = mutated["fast_ma"] + 15
        return mutated

    def optimize(self) -> dict:
        """
        Executes evolutionary search across specified generations.
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 1. Initialize Population
        population = [self._random_chromosome() for _ in range(self.population_size)]
        history = []
        best_overall_chrom = None
        best_overall_fitness = -999.0
        best_overall_metrics = {}

        for gen in range(1, self.generations + 1):
            scores = []
            metrics_list = []
            for indiv in population:
                fit, m = self._evaluate_fitness(indiv)
                scores.append(fit)
                metrics_list.append(m)

            gen_best_idx = int(np.argmax(scores))
            gen_best_fit = scores[gen_best_idx]
            gen_avg_fit = float(np.mean(scores))

            if gen_best_fit > best_overall_fitness:
                best_overall_fitness = gen_best_fit
                best_overall_chrom = population[gen_best_idx].copy()
                best_overall_metrics = metrics_list[gen_best_idx]

            history.append({
                "generation": gen,
                "best_fitness": round(gen_best_fit, 3),
                "avg_fitness": round(gen_avg_fit, 3),
            })

            # Selection & Reproduction
            # Elitism: keep top 2 individuals
            sorted_indices = np.argsort(scores)[::-1]
            next_pop = [population[sorted_indices[0]], population[sorted_indices[1]]]

            while len(next_pop) < self.population_size:
                # Tournament selection
                candidates1 = random.sample(range(self.population_size), min(self.tournament_size, self.population_size))
                p1_idx = max(candidates1, key=lambda idx: scores[idx])
                candidates2 = random.sample(range(self.population_size), min(self.tournament_size, self.population_size))
                p2_idx = max(candidates2, key=lambda idx: scores[idx])

                child = self._crossover(population[p1_idx], population[p2_idx])
                child = self._mutate(child)
                next_pop.append(child)

            population = next_pop

        return {
            "best_params": best_overall_chrom,
            "best_fitness": round(best_overall_fitness, 3),
            "best_metrics": best_overall_metrics,
            "history_df": pd.DataFrame(history),
        }
