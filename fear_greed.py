"""
fear_greed.py
-------------
Free public API client for the Crypto Fear & Greed Index (https://api.alternative.me/fng/).
Requires zero API keys and zero paid subscriptions.
"""

import requests
import pandas as pd
from datetime import datetime


def fetch_fear_and_greed_index(limit: int = 30) -> dict:
    """
    Fetches the latest Crypto Fear & Greed Index values.

    Returns:
      - current_value: int (0 to 100)
      - current_classification: str ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")
      - history_df: DataFrame with date, value, classification
    """
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json().get("data", [])

        if not data:
            return _fallback_fear_greed()

        latest = data[0]
        val = int(latest["value"])
        classification = latest["value_classification"]

        history = []
        for item in data:
            dt = datetime.fromtimestamp(int(item["timestamp"]))
            history.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Value": int(item["value"]),
                "Classification": item["value_classification"],
            })

        return {
            "current_value": val,
            "current_classification": classification,
            "history_df": pd.DataFrame(history),
        }
    except Exception:
        return _fallback_fear_greed()


def _fallback_fear_greed():
    """Fallback values if network is unavailable."""
    return {
        "current_value": 50,
        "current_classification": "Neutral",
        "history_df": pd.DataFrame([{"Date": datetime.now().strftime("%Y-%m-%d"), "Value": 50, "Classification": "Neutral"}]),
    }
