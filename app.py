"""
LIVE TRADING ADVISOR - Streamlit UI

Lets you pick an asset (or type a custom one) from a dropdown, choose
a strategy and risk style, fetches live price data, and shows a clear
BUY/SELL/HOLD recommendation with reasoning and a historical track
record for that exact combination.

This is a DECISION-SUPPORT tool. It never places trades or touches
real money -- you decide whether to act on any recommendation shown.

Run with:
    pip install streamlit streamlit-autorefresh plotly
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import requests
import os
import json
import smtplib
from email.mime.text import MIMEText
from fpdf import FPDF
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from indicators import add_all_indicators
from regime_detector import add_regime
from strategy_v4_filtered_ma import generate_signals as ma_crossover_signals
from candlestick_strategy import generate_candlestick_signals
from strategy import generate_signals as rsi_signals
from strategy_bollinger import generate_signals as bollinger_signals
from strategy_macd import generate_signals as macd_signals
from strategy_breakout import generate_signals as breakout_signals
from strategy_buyhold import generate_signals as buyhold_signals
from backtest import backtest

st.set_page_config(page_title="Live Trading Advisor", layout="wide")

ASSET_OPTIONS = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "PAX Gold (PAXG)": "PAXGUSDT",
}

STRATEGY_FUNCS = {
    "MA Crossover (regime-filtered)": ma_crossover_signals,
    "Candlestick + Trend Filter": generate_candlestick_signals,
    "RSI Mean-Reversion": rsi_signals,
    "Bollinger Bands": bollinger_signals,
    "MACD Crossover": macd_signals,
    "Volume-Confirmed Breakout": breakout_signals,
    "Buy & Hold (baseline)": buyhold_signals,
}

RISK_STYLES = {
    "No stop-loss": {},
    "Fixed 10% stop-loss": {"stop_loss_pct": 0.10},
    "Trailing 10% stop": {"trailing_stop_pct": 0.10},
}

POSITION_SIZE_OPTIONS = {
    "All-in per trade": None,
    "Risk 1% per trade": 0.01,
    "Risk 2% per trade": 0.02,
    "Risk 5% per trade": 0.05,
}


@st.cache_data(ttl=300)
def fetch_data(symbol, interval="1d", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "num_trades",
               "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"open_time": "timestamp"})
    df = df.iloc[:-1].reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def fetch_data_paginated(symbol, days, interval="1d"):
    all_rows = []
    end_time_ms = int(datetime.now().timestamp() * 1000)
    earliest_needed = datetime.now().timestamp() - (days * 86400)

    while True:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "endTime": end_time_ms}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        all_rows = batch + all_rows
        oldest_time = batch[0][0]
        if (oldest_time / 1000) <= earliest_needed:
            break
        end_time_ms = oldest_time - 1

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "num_trades",
               "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"open_time": "timestamp"})
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def find_data_for_trade_target(symbol, strategy_func, target_trades, max_days=3000):
    days = 300
    while True:
        df = fetch_data_paginated(symbol, days)
        prepped = prep_data(df.copy())
        signal_df = strategy_func(prepped)
        num_trades = len(signal_df[signal_df["signal"] != "HOLD"])

        ran_out_of_history = len(df) < days * 0.9
        if num_trades >= target_trades or days >= max_days or ran_out_of_history:
            return signal_df, num_trades, days

        days = min(days * 2, max_days)


def prep_data(df):
    df = add_all_indicators(df)
    df = add_regime(df)
    return df


def reason_for_signal(recommendation):
    if recommendation == "BUY":
        return "The selected strategy's entry conditions were met on the most recent closed candle."
    elif recommendation == "SELL":
        return "The selected strategy's exit conditions were met on the most recent closed candle."
    else:
        return "No new entry/exit condition triggered on the most recent closed candle."


def get_recommendation(df, strategy_func):
    signal_df = strategy_func(df.copy())
    latest = signal_df.iloc[-1]
    recommendation = latest["signal"]
    reason = reason_for_signal(recommendation)
    return recommendation, reason, latest, signal_df


def get_consensus(df):
    """
    Runs EVERY strategy on the same data and returns each one's current
    signal, so you can see how much agreement there is rather than
    trusting a single strategy's opinion in isolation.
    """
    votes = {}
    for name, func in STRATEGY_FUNCS.items():
        try:
            sdf = func(df.copy())
            votes[name] = sdf.iloc[-1]["signal"]
        except Exception:
            votes[name] = "N/A"
    return votes


def run_track_record(signal_df, risk_kwargs):
    results = backtest(signal_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)
    return results


def generate_pdf_report(asset_label, symbol, strategy_choice, risk_choice, slippage_choice,
                        recommendation, reason, latest, track_results, window_label):
    """
    Builds a one-page PDF summary of the current recommendation and its
    backtest track record -- useful as a personal record of what the
    tool said and why, at a point in time.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Trading Advisor Report", ln=1)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Asset: {asset_label} ({symbol})", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Strategy: {strategy_choice}", ln=1)
    pdf.cell(0, 6, f"Risk style: {risk_choice}", ln=1)
    pdf.cell(0, 6, f"Slippage assumption: {slippage_choice}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Recommendation: {recommendation}", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, reason)
    pdf.ln(2)
    pdf.cell(0, 6, f"Price: ${latest['close']:.2f}   MA20: ${latest['MA_20']:.2f}   "
                   f"MA50: ${latest['MA_50']:.2f}", ln=1)
    pdf.cell(0, 6, f"Regime: {latest['regime']}   RSI: {latest['RSI']:.1f}", ln=1)
    pdf.cell(0, 6, f"As of candle: {latest['timestamp'].date()}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Track record ({window_label})", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Strategy return: {track_results['strategy_return_pct']:.2f}%", ln=1)
    pdf.cell(0, 6, f"Buy & hold return: {track_results['buy_hold_return_pct']:.2f}%", ln=1)
    pdf.cell(0, 6, f"Number of trades: {track_results['num_trades']}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "This is a decision-support tool, not financial advice. Past "
                        "performance does not guarantee future results.")

    return bytes(pdf.output())


JOURNAL_FILE = "trade_journal.csv"


def load_journal():
    if os.path.isfile(JOURNAL_FILE):
        return pd.read_csv(JOURNAL_FILE, parse_dates=["Date"])
    return pd.DataFrame(columns=["Date", "Asset", "Strategy", "Action", "Price", "Notes"])


def add_journal_entry(date, asset, strategy, action, price, notes):
    journal = load_journal()
    new_entry = pd.DataFrame([{
        "Date": date, "Asset": asset, "Strategy": strategy,
        "Action": action, "Price": price, "Notes": notes
    }])
    journal = pd.concat([journal, new_entry], ignore_index=True)
    journal.to_csv(JOURNAL_FILE, index=False)
    return journal


LAST_SIGNAL_FILE = "last_signals.json"


def load_last_signals():
    if os.path.isfile(LAST_SIGNAL_FILE):
        with open(LAST_SIGNAL_FILE, "r") as f:
            return json.load(f)
    return {}


def save_last_signals(data):
    with open(LAST_SIGNAL_FILE, "w") as f:
        json.dump(data, f)


def send_email_alert(subject, body):
    """
    Sends an email using credentials from .streamlit/secrets.toml.
    Returns (success: bool, message: str) so the UI can show what happened
    without crashing if email isn't configured.
    """
    try:
        email_from = st.secrets["email_from"]
        email_password = st.secrets["email_password"]
        email_to = st.secrets["email_to"]
    except Exception:
        return False, "Email not configured (missing .streamlit/secrets.toml)."

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_from, email_password)
            server.sendmail(email_from, email_to, msg.as_string())
        return True, "Email sent."
    except Exception as e:
        return False, f"Email failed to send: {e}"


def check_and_alert(symbol, strategy_choice, recommendation, latest_price):
    """
    Compares today's recommendation to the last one seen for this exact
    symbol+strategy combo. If it changed to a real BUY or SELL (not just
    still HOLD), sends an email alert and updates the stored state.
    """
    last_signals = load_last_signals()
    key = f"{symbol}_{strategy_choice}"
    previous = last_signals.get(key)

    alert_sent = None
    if recommendation in ["BUY", "SELL"] and recommendation != previous:
        subject = f"Trading Advisor Alert: {recommendation} signal on {symbol}"
        body = (f"{strategy_choice} just generated a {recommendation} signal "
                f"for {symbol} at price ${latest_price:.2f}.\n\n"
                f"This is decision-support only -- review before acting.")
        success, message = send_email_alert(subject, body)
        alert_sent = (success, message)

    last_signals[key] = recommendation
    save_last_signals(last_signals)
    return alert_sent


def get_signal_history(signal_df, n=10):
    history = signal_df[signal_df["signal"] != "HOLD"].copy()
    history = history[["timestamp", "close", "regime", "signal"]].tail(n)
    history = history.rename(columns={
        "timestamp": "Date", "close": "Price", "regime": "Regime", "signal": "Signal"
    })
    history["Date"] = history["Date"].dt.date
    history["Price"] = history["Price"].round(2)
    return history.iloc[::-1]


def run_optimizer(symbol, lookback_days=200):
    """
    Tests every strategy x risk style combo (21 total) on the same data
    and ranks them by return. NOTE: the top result here is NOT
    automatically the "best" strategy going forward -- it's whichever
    combo happened to fit this specific historical window best, which
    can just be overfitting (we proved this ourselves earlier in this
    project with the candlestick strategy). Use this as a starting
    point for further validation (train/test split, multiple periods),
    not as a final answer.
    """
    df = fetch_data(symbol, limit=lookback_days)
    df = prep_data(df)

    results = []
    for strat_name, strat_func in STRATEGY_FUNCS.items():
        try:
            signal_df = strat_func(df.copy())
        except Exception as e:
            results.append({"Strategy": strat_name, "Risk Style": "N/A", "Return %": None,
                            "Buy&Hold %": None, "Beat B&H?": f"Error: {e}", "Trades": None})
            continue

        for risk_name, risk_kwargs in RISK_STYLES.items():
            try:
                r = backtest(signal_df, starting_cash=1000.0, fee_rate=0.001, **risk_kwargs)
                results.append({
                    "Strategy": strat_name,
                    "Risk Style": risk_name,
                    "Return %": round(r["strategy_return_pct"], 2),
                    "Buy&Hold %": round(r["buy_hold_return_pct"], 2),
                    "Beat B&H?": "YES" if r["strategy_return_pct"] > r["buy_hold_return_pct"] else "no",
                    "Trades": r["num_trades"],
                })
            except Exception as e:
                results.append({"Strategy": strat_name, "Risk Style": risk_name, "Return %": None,
                                "Buy&Hold %": None, "Beat B&H?": f"Error: {e}", "Trades": None})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Return %", ascending=False, na_position="last")
    return results_df


def run_portfolio_backtest(asset_symbols, strategy_func, risk_kwargs, total_capital, lookback_days=200):
    """
    Splits total_capital equally across the chosen assets, runs the same
    strategy+risk combo on each, and combines the results into an
    overall portfolio return -- vs a blended buy-and-hold benchmark.
    """
    per_asset_capital = total_capital / len(asset_symbols)
    rows = []
    total_final = 0.0
    total_buyhold = 0.0

    for sym in asset_symbols:
        try:
            df = fetch_data(sym, limit=lookback_days)
            df = prep_data(df)
            signal_df = strategy_func(df.copy())
            r = backtest(signal_df, starting_cash=per_asset_capital, fee_rate=0.001, **risk_kwargs)

            rows.append({
                "Asset": sym,
                "Allocated": f"${per_asset_capital:.2f}",
                "Final Value": f"${r['final_value_strategy']:.2f}",
                "Return %": round(r["strategy_return_pct"], 2),
                "Buy&Hold %": round(r["buy_hold_return_pct"], 2),
                "Trades": r["num_trades"],
            })
            total_final += r["final_value_strategy"]
            total_buyhold += r["final_value_buy_hold"]
        except Exception as e:
            rows.append({"Asset": sym, "Allocated": "N/A", "Final Value": f"Error: {e}",
                         "Return %": None, "Buy&Hold %": None, "Trades": None})

    portfolio_return_pct = (total_final / total_capital - 1) * 100
    buyhold_return_pct = (total_buyhold / total_capital - 1) * 100

    return {
        "rows": rows,
        "total_final": total_final,
        "portfolio_return_pct": portfolio_return_pct,
        "total_buyhold": total_buyhold,
        "buyhold_return_pct": buyhold_return_pct,
    }


@st.cache_data(ttl=300)
def get_all_recommendations():
    rows = []
    default_func = STRATEGY_FUNCS["MA Crossover (regime-filtered)"]
    for label, sym in ASSET_OPTIONS.items():
        try:
            df = fetch_data(sym)
            df = prep_data(df)
            rec, reason, latest, _ = get_recommendation(df, default_func)
            rows.append({
                "Asset": label,
                "Price": f"${latest['close']:.2f}",
                "Regime": latest["regime"],
                "Recommendation": rec,
            })
        except Exception as e:
            rows.append({
                "Asset": label, "Price": "N/A", "Regime": "N/A",
                "Recommendation": f"Error: {e}"
            })
    return pd.DataFrame(rows)


def make_candlestick_chart(df, title=""):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["MA_20"], name="MA20",
        line=dict(color="#ffa726", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["MA_50"], name="MA50",
        line=dict(color="#42a5f5", width=1.5)
    ))
    fig.update_layout(
        title=title, template="plotly_dark", height=450,
        xaxis_rangeslider_visible=True,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def badge(text, kind):
    colors = {
        "BUY": ("#0f5132", "#75dd9a"), "SELL": ("#5c1a1a", "#f28b8b"),
        "HOLD": ("#3a3a3a", "#bbbbbb"),
        "TRENDING": ("#0f5132", "#75dd9a"), "CHOPPY": ("#4a3b0f", "#e0c060"),
    }
    bg, fg = colors.get(kind, ("#3a3a3a", "#bbbbbb"))
    return (f'<span style="background-color:{bg}; color:{fg}; padding:4px 12px; '
            f'border-radius:12px; font-weight:600; font-size:0.9em;">{text}</span>')


st.title("Live Trading Advisor")
st.caption("Decision-support only. This tool never places trades -- you decide.")

with st.expander("What this strategy is actually good at (read before trusting it)"):
    st.markdown("""
    The default combo (regime-filtered MA crossover with a 10% trailing stop)
    was tested across 5 assets and multiple market periods, including a
    proper train/test split and a dedicated crash-period test:

    - **Strength:** during the 2022 crash, it lost only ~10% while simply
      holding lost 65% -- a strong, real downside-protection edge.
    - **Strength:** on BTC and ETH specifically, it meaningfully beat the
      fixed-stop version, and on ETH it even beat buy-and-hold outright.
    - **Known limitation:** on more volatile assets (SOL, BNB), the 10%
      trailing stop tends to get triggered by normal volatility, exiting
      trades too early.
    - You now have 7 strategies, 3 risk styles, and 2 ways to size the
      backtest window (by days or by target trade count) to mix and
      match per asset -- use the Track Record section below to compare
      combos honestly before trusting any recommendation.
    """)

refresh_minutes = st.sidebar.selectbox(
    "Auto-refresh every:", [5, 15, 30, 60], index=1
)
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
           f"(auto-refreshing every {refresh_minutes} min)")

st.markdown("### All assets at a glance")
st.caption("Uses the default combo (MA Crossover + Trailing stop). Customize per asset below.")
with st.spinner("Checking all assets..."):
    overview_df = get_all_recommendations()


def highlight_recommendation(val):
    if val == "BUY":
        return "color: green; font-weight: bold"
    elif val == "SELL":
        return "color: red; font-weight: bold"
    elif val == "HOLD":
        return "color: gray"
    return ""


st.dataframe(
    overview_df.style.map(highlight_recommendation, subset=["Recommendation"]),
    use_container_width=True, hide_index=True
)

st.markdown("---")
st.markdown("### Portfolio backtest")
st.caption("Split capital across multiple assets at once, using one shared strategy/risk "
           "combo, and see the combined result -- instead of testing one asset in isolation.")

with st.expander("Run a portfolio backtest"):
    pf_assets = st.multiselect(
        "Choose assets to include:", list(ASSET_OPTIONS.keys()),
        default=["Bitcoin (BTC)", "Ethereum (ETH)"]
    )
    pcol1, pcol2, pcol3 = st.columns(3)
    pf_strategy_choice = pcol1.selectbox(
        "Strategy:", list(STRATEGY_FUNCS.keys()), key="pf_strategy"
    )
    pf_risk_choice = pcol2.selectbox(
        "Risk style:", list(RISK_STYLES.keys()), index=2, key="pf_risk"
    )
    pf_capital = pcol3.number_input(
        "Total capital ($):", min_value=100.0, value=1000.0, step=100.0, key="pf_capital"
    )

    if st.button("Run portfolio backtest", type="primary"):
        if len(pf_assets) == 0:
            st.warning("Select at least one asset.")
        else:
            pf_symbols = [ASSET_OPTIONS[a] for a in pf_assets]
            pf_func = STRATEGY_FUNCS[pf_strategy_choice]
            pf_risk_kwargs = dict(RISK_STYLES[pf_risk_choice])

            with st.spinner("Running portfolio backtest across all selected assets..."):
                pf_results = run_portfolio_backtest(pf_symbols, pf_func, pf_risk_kwargs, pf_capital)

            st.dataframe(pd.DataFrame(pf_results["rows"]), use_container_width=True, hide_index=True)

            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Portfolio final value", f"${pf_results['total_final']:.2f}",
                        f"{pf_results['portfolio_return_pct']:.2f}%")
            rcol2.metric("Blended buy & hold value", f"${pf_results['total_buyhold']:.2f}",
                        f"{pf_results['buyhold_return_pct']:.2f}%")

            if pf_results["portfolio_return_pct"] > pf_results["buyhold_return_pct"]:
                st.success("Portfolio beat the blended buy-and-hold benchmark.")
            else:
                st.warning("Portfolio underperformed the blended buy-and-hold benchmark.")

st.markdown("---")
st.markdown("### Auto-optimizer")
st.caption("Tests every strategy x risk style combo (21 total) on one asset and ranks them.")
st.warning("Important: the top-ranked combo here is whichever one happened to fit THIS "
           "specific historical window best -- that can easily be overfitting rather than "
           "a real edge (we proved this ourselves earlier with the candlestick strategy, "
           "which looked great on one window and fell apart on others). Treat this as a "
           "starting point to investigate further -- re-check any promising combo with a "
           "train/test split and on other assets before trusting it.")

with st.expander("Run auto-optimizer"):
    opt_asset = st.selectbox("Asset to optimize:", list(ASSET_OPTIONS.keys()), key="opt_asset")
    opt_days = st.slider("Days of history to test on:", min_value=60, max_value=1000,
                         value=200, step=10, key="opt_days")

    if st.button("Run auto-optimizer", type="primary"):
        opt_symbol = ASSET_OPTIONS[opt_asset]
        with st.spinner("Testing 21 strategy/risk combos... this may take a moment"):
            opt_results = run_optimizer(opt_symbol, opt_days)

        st.dataframe(opt_results, use_container_width=True, hide_index=True)

        best_row = opt_results.iloc[0]
        st.info(f"Top result on this window: **{best_row['Strategy']}** with "
                f"**{best_row['Risk Style']}** ({best_row['Return %']}% return). "
                f"Go validate this combo properly before trusting it with real decisions.")

st.markdown("---")
st.markdown("### Asset detail")

asset_label = st.selectbox("Choose an asset:", list(ASSET_OPTIONS.keys()) + ["Custom (type any symbol)"])

if asset_label == "Custom (type any symbol)":
    custom_input = st.text_input(
        "Enter a Binance trading pair symbol:",
        placeholder="e.g. DOGE, ADA, XRP, MATIC, LINK...",
        help="Just the coin name is fine -- USDT will be added automatically if you leave it off."
    )
    if not custom_input:
        st.info("Type a symbol above to continue (e.g. DOGE, ADA, XRP).")
        st.stop()

    symbol = custom_input.strip().upper().replace(" ", "")
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    asset_label = symbol

    try:
        _test_df = fetch_data(symbol, limit=5)
        if len(_test_df) == 0:
            st.error(f"'{symbol}' returned no data. Check the symbol is correct (e.g. DOGE, not DOGECOIN).")
            st.stop()
    except Exception as e:
        st.error(f"Couldn't find '{symbol}' on Binance. Check the spelling -- e.g. try 'DOGE' or 'ADA'. "
                  f"(Error: {e})")
        st.stop()
else:
    symbol = ASSET_OPTIONS[asset_label]

if symbol in ["SOLUSDT", "BNBUSDT"]:
    st.warning(f"Known limitation: the default trailing-stop MA crossover has tested "
               f"noticeably worse on {asset_label} historically due to its higher "
               f"volatility. Consider trying a different strategy/risk combo below, "
               f"or treat recommendations here with extra caution.")

scol1, scol2, scol3, scol4 = st.columns(4)
strategy_choice = scol1.selectbox(
    "Core strategy:", list(STRATEGY_FUNCS.keys()), key=f"strategy_{symbol}"
)
risk_choice = scol2.selectbox(
    "Risk style:", list(RISK_STYLES.keys()), index=2, key=f"risk_{symbol}"
)
sizing_choice = scol3.selectbox(
    "Position sizing:", list(POSITION_SIZE_OPTIONS.keys()), key=f"sizing_{symbol}"
)
SLIPPAGE_OPTIONS = {
    "None (unrealistic best case)": 0.0,
    "Low (0.1%, liquid major coins)": 0.001,
    "Medium (0.5%, typical altcoins)": 0.005,
    "High (1%, thin/illiquid markets)": 0.01,
}
slippage_choice = scol4.selectbox(
    "Slippage assumption:", list(SLIPPAGE_OPTIONS.keys()), index=1, key=f"slippage_{symbol}"
)
strategy_func = STRATEGY_FUNCS[strategy_choice]
risk_kwargs = dict(RISK_STYLES[risk_choice])
risk_per_trade_pct = POSITION_SIZE_OPTIONS[sizing_choice]
risk_kwargs["slippage_pct"] = SLIPPAGE_OPTIONS[slippage_choice]

if risk_per_trade_pct is not None:
    if risk_choice == "No stop-loss":
        st.warning("Risk-based position sizing needs a stop-loss to size against -- "
                   "using 'All-in per trade' instead since 'No stop-loss' is selected. "
                   "Pick a stop-loss risk style above to enable sizing.")
    else:
        risk_kwargs["risk_per_trade_pct"] = risk_per_trade_pct

mode = st.radio(
    "Backtest window:", ["By number of days", "By number of trades"],
    horizontal=True, key=f"mode_{symbol}"
)

if mode == "By number of days":
    lookback_days = st.slider(
        "Number of days to analyze/backtest:", min_value=60, max_value=1000, value=200, step=10,
        key=f"lookback_{symbol}"
    )
    with st.spinner(f"Fetching live data for {symbol}..."):
        df = fetch_data(symbol, limit=lookback_days)
        df = prep_data(df)
        recommendation, reason, latest, signal_df = get_recommendation(df, strategy_func)
    window_label = f"last {lookback_days} days"
else:
    target_trades = st.number_input(
        "Target number of trades to test:", min_value=5, max_value=500, value=20, step=5,
        key=f"target_trades_{symbol}",
        help="The app will keep pulling further back in history until this many "
             "BUY/SELL trades have occurred for the selected strategy."
    )
    with st.spinner(f"Searching history for {target_trades} trades... this may take a moment"):
        signal_df, achieved_trades, days_used = find_data_for_trade_target(
            symbol, strategy_func, target_trades
        )
        latest = signal_df.iloc[-1]
        recommendation = latest["signal"]
        reason = reason_for_signal(recommendation)
        df = signal_df

    if achieved_trades < target_trades:
        st.info(f"Only found {achieved_trades} trades in {days_used} days of available "
                f"history (couldn't reach {target_trades} -- this asset may not have that "
                f"much history, or this strategy trades rarely).")
    else:
        st.caption(f"Found {achieved_trades} trades within {days_used} days of history.")
    window_label = f"{achieved_trades} trades ({days_used} days of history)"

st.markdown(f"## {badge(recommendation, recommendation)}", unsafe_allow_html=True)
st.write(reason)
st.caption(f"Using: **{strategy_choice}** with **{risk_choice}**")

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${latest['close']:.2f}")
col2.metric("MA20", f"${latest['MA_20']:.2f}")
col3.metric("MA50", f"${latest['MA_50']:.2f}")

st.markdown(f"**Regime:** {badge(latest['regime'], latest['regime'])}  |  **RSI:** {latest['RSI']:.1f}",
            unsafe_allow_html=True)
st.caption(f"Based on most recent closed candle: {latest['timestamp'].date()}")

if recommendation == "HOLD" and latest["regime"] == "CHOPPY":
    st.caption("Context: markets are choppy about 80% of the time historically -- "
               "many strategies are designed to sit out these periods rather than "
               "guess, which may be intentional rather than a malfunction.")

st.markdown("#### Strategy consensus")
st.caption("How the other strategies vote on this same asset right now.")
votes = get_consensus(df)
buy_count = sum(1 for v in votes.values() if v == "BUY")
sell_count = sum(1 for v in votes.values() if v == "SELL")
hold_count = sum(1 for v in votes.values() if v == "HOLD")
total = len(votes)

vcol1, vcol2, vcol3 = st.columns(3)
vcol1.metric("BUY votes", f"{buy_count}/{total}")
vcol2.metric("SELL votes", f"{sell_count}/{total}")
vcol3.metric("HOLD votes", f"{hold_count}/{total}")

if buy_count >= total * 0.6:
    st.success(f"Strong agreement: {buy_count} of {total} strategies say BUY.")
elif sell_count >= total * 0.6:
    st.error(f"Strong agreement: {sell_count} of {total} strategies say SELL.")
else:
    st.info("Mixed signals across strategies -- no strong consensus right now.")

vote_table = pd.DataFrame(list(votes.items()), columns=["Strategy", "Current Signal"])
st.dataframe(vote_table, use_container_width=True, hide_index=True)

st.plotly_chart(make_candlestick_chart(df.tail(300), title=f"{asset_label} - Price with MA20/MA50"),
                use_container_width=True)
st.caption("Chart shows up to the last 300 candles for readability. "
           "The backtest below uses the full window you selected above.")

st.markdown(f"### Track record ({window_label})")
st.caption(f"How **{strategy_choice}** with **{risk_choice}** would have performed "
           f"historically on this asset -- so you can judge the recommendation with "
           f"real context, not just trust it blindly.")

track_results = run_track_record(signal_df, risk_kwargs)
tcol1, tcol2, tcol3 = st.columns(3)
tcol1.metric("Strategy return", f"{track_results['strategy_return_pct']:.2f}%")
tcol2.metric("Buy & hold return", f"{track_results['buy_hold_return_pct']:.2f}%")
tcol3.metric("Number of trades", track_results['num_trades'])

if track_results['strategy_return_pct'] > track_results['buy_hold_return_pct']:
    st.success("This combo beat buy-and-hold over this period.")
else:
    st.warning("This combo underperformed buy-and-hold over this period. "
               "Try a different strategy/risk combo above to compare.")

trade_log_df = pd.DataFrame(track_results["trade_log"])
if len(trade_log_df) > 0:
    st.download_button(
        "Download full trade log (CSV)",
        data=trade_log_df.to_csv(index=False),
        file_name=f"{symbol}_{strategy_choice.replace(' ', '_')}_trade_log.csv",
        mime="text/csv"
    )

pdf_bytes = generate_pdf_report(asset_label, symbol, strategy_choice, risk_choice,
                                slippage_choice, recommendation, reason, latest,
                                track_results, window_label)
st.download_button(
    "Download summary report (PDF)",
    data=pdf_bytes,
    file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf"
)

st.markdown("### Recent signal history")
st.caption(f"The actual BUY/SELL signals **{strategy_choice}** generated for this asset recently.")
history = get_signal_history(signal_df, n=10)
if len(history) > 0:
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button(
        "Download signal history (CSV)",
        data=history.to_csv(index=False),
        file_name=f"{symbol}_{strategy_choice.replace(' ', '_')}_signals.csv",
        mime="text/csv"
    )
else:
    st.write("No BUY/SELL signals in the recent history window.")

st.markdown("---")
st.markdown("### Trade journal")
st.caption("Log what you actually did in real life, so you can track real outcomes "
           "over time -- separate from the backtest above.")

with st.form(key=f"journal_form_{symbol}"):
    jcol1, jcol2, jcol3 = st.columns(3)
    entry_date = jcol1.date_input("Date", value=datetime.now())
    entry_action = jcol2.selectbox("Action taken", ["BUY", "SELL", "HOLD (no action)"])
    entry_price = jcol3.number_input("Price", value=float(latest["close"]), format="%.2f")
    entry_notes = st.text_input("Notes (optional)", placeholder="e.g. only bought half position")
    submitted = st.form_submit_button("Log this entry")

    if submitted:
        add_journal_entry(entry_date, asset_label, strategy_choice, entry_action, entry_price, entry_notes)
        st.success("Entry logged.")

journal = load_journal()
asset_journal = journal[journal["Asset"] == asset_label] if len(journal) > 0 else journal
if len(asset_journal) > 0:
    st.dataframe(asset_journal.sort_values("Date", ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button(
        "Download journal (CSV)",
        data=asset_journal.to_csv(index=False),
        file_name=f"{symbol}_journal.csv",
        mime="text/csv"
    )
else:
    st.write("No journal entries for this asset yet.")

st.markdown("---")
st.caption("Pick a strategy, risk style, and backtest window above for each asset -- "
           "your choices are remembered per asset as you switch between them. "
           "Past performance does not guarantee future results.")