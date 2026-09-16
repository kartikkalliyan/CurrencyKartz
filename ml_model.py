"""
ml_model.py
-----------
Machine Learning Market Direction & Regime Predictor.
Uses a Random Forest Classifier trained on technical indicator features
with a strictly temporal train/test split to forecast next-candle direction
and rank feature importance without lookahead bias.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from indicators import add_all_indicators


class MarketMLPredictor:
    """
    Random Forest Machine Learning model for next-period market direction forecasting.
    """

    def __init__(self, df: pd.DataFrame, train_split: float = 0.80):
        self.raw_df = df.copy()
        self.train_split = train_split
        self.model = RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_split=6,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = []

    def _prepare_features(self) -> pd.DataFrame:
        """Constructs stationary indicator features and binary forward return target."""
        df = add_all_indicators(self.raw_df.copy())

        # Returns & Momentum
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_5d"] = df["close"].pct_change(5)
        df["volatility_10d"] = df["ret_1d"].rolling(10).std()

        # Moving Average spreads
        if "MA_20" in df.columns:
            df["ma20_spread"] = (df["close"] - df["MA_20"]) / df["MA_20"]
        if "MA_50" in df.columns:
            df["ma50_spread"] = (df["close"] - df["MA_50"]) / df["MA_50"]

        # Normalized ATR
        if "ATR" in df.columns:
            df["norm_atr"] = df["ATR"] / df["close"]

        # Volume acceleration
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(15).mean()
            df["vol_ratio"] = np.where(vol_ma > 0, df["volume"] / vol_ma, 1.0)

        # Target: 1 if next candle close is higher than current close, 0 otherwise
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        candidate_cols = [
            "ret_1d", "ret_5d", "volatility_10d", "RSI", "MACD_hist",
            "ma20_spread", "ma50_spread", "norm_atr", "vol_ratio", "target"
        ]
        available_cols = [c for c in candidate_cols if c in df.columns]
        clean_df = df[available_cols].dropna().copy()

        self.feature_names = [c for c in available_cols if c != "target"]
        return clean_df

    def train_and_predict(self) -> dict:
        """
        Trains model using temporal train/test split, evaluates test accuracy,
        and forecasts the probability of the upcoming candle.
        """
        data = self._prepare_features()
        if len(data) < 30:
            return {
                "test_accuracy_pct": 50.0,
                "next_direction": "NEUTRAL",
                "bullish_prob_pct": 50.0,
                "bearish_prob_pct": 50.0,
                "feature_importance_df": pd.DataFrame(),
            }

        X = data[self.feature_names].values
        y = data["target"].values

        split_idx = int(len(data) * self.train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model.fit(X_train, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.50

        # Feature Importances
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance %": np.round(importances * 100.0, 2)
        }).sort_values("Importance %", ascending=False)

        # Latest candle prediction
        latest_features = X[-1].reshape(1, -1)
        prob = self.model.predict_proba(latest_features)[0]
        # prob[0] is Bearish (0), prob[1] is Bullish (1)
        bear_prob = round(float(prob[0]) * 100.0, 1)
        bull_prob = round(float(prob[1]) * 100.0, 1)

        if bull_prob >= 56.0:
            direction = "BULLISH 🟢"
        elif bear_prob >= 56.0:
            direction = "BEARISH 🔴"
        else:
            direction = "NEUTRAL ⚪"

        return {
            "test_accuracy_pct": round(float(acc) * 100.0, 2),
            "next_direction": direction,
            "bullish_prob_pct": bull_prob,
            "bearish_prob_pct": bear_prob,
            "feature_importance_df": fi_df,
        }
