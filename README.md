# CurrencyKartz

An open-source quantitative trading research platform for cryptocurrency markets.

CurrencyKartz provides a modular framework for developing, testing, comparing, and optimizing algorithmic trading strategies using historical market data.
---

## Features

✅ Historical Backtesting

✅ Multiple Trading Strategies

- Moving Average Crossover
- Filtered MA Strategy
- MACD
- Bollinger Bands
- Breakout Strategy
- Buy & Hold Benchmark
- Adaptive Strategy
- Candlestick Pattern Strategy

✅ Technical Indicators

- RSI
- EMA
- SMA
- MACD
- ATR
- Bollinger Bands

✅ Risk Management

- Trailing Stop Loss
- ATR-based exits
- Drawdown analysis

✅ Market Regime Detection

- Trend detection
- Adaptive strategy switching

✅ Multi Asset Testing

Run strategies across multiple cryptocurrencies instead of a single BTC dataset.

✅ Live Signal Generation

Generate trading signals from the latest market data.

---

## Project Structure

```text
app.py                        # Streamlit interface

backtest.py                   # Core backtesting engine

live_bot.py                   # Live signal generator

fetch_data.py                 # Historical data collection

indicators.py                 # Technical indicators

regime_detector.py            # Market regime detection

strategy_*.py                 # Trading strategies

multi_asset_*.py              # Multi-asset testing

run_train_test*.py            # Train/Test workflows

btc_*.csv                     # Historical datasets

trade_journal.csv             # Trade history

backtest_trade_log.csv        # Detailed backtest logs
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- yfinance
- Streamlit

---

## Installation

```bash
git clone https://github.com/kartikkalliyan/-crypto-strategy-backtester.git

cd -crypto-strategy-backtester

pip install -r requirements.txt
```

---

## Run Backtest

```bash
python backtest.py
```

---

## Launch Application

```bash
streamlit run app.py
```

---

## Strategy Comparison

| Strategy | Status |
|-----------|--------|
| Buy & Hold | ✅ |
| MA Crossover | ✅ |
| Filtered MA | ✅ |
| MACD | ✅ |
| Bollinger Bands | ✅ |
| Breakout | ✅ |
| Adaptive Strategy | ✅ |
| Candlestick Strategy | ✅ |

---

## Repository Highlights

- Modular strategy architecture
- Reusable indicator library
- Historical data pipeline
- Live trading signal generation
- Multi-asset evaluation
- Train/Test validation
- Crash testing framework
- Trade logging
- CSV export support

---

## Future Improvements

- [ ] Binance API integration
- [ ] Portfolio optimization
- [ ] Walk-forward optimization
- [ ] Genetic Algorithm parameter tuning
- [ ] Reinforcement Learning strategies
- [ ] Docker deployment
- [ ] REST API
- [ ] Interactive dashboard
- [ ] Performance analytics dashboard

---

## Why this project?

The goal of this project is to build a complete quantitative trading research framework rather than a simple trading bot.

It focuses on experimentation, strategy development, and realistic backtesting while maintaining a modular architecture that can be extended with new strategies and indicators.

---

## Author

**Kartik Kalliyan**

If you found this project useful, consider giving it a ⭐.
