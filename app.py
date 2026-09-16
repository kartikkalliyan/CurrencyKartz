"""
LIVE TRADING ADVISOR — Streamlit UI (Modern Dark Financial Terminal Redesign)

All actual logic lives in config.py, data_fetching.py, core_logic.py,
journal.py, reports.py, display_helpers.py, binance_market.py, binance_ws.py,
alerts.py, paper_trading.py, fear_greed.py, monte_carlo.py, walk_forward.py,
and indian_stocks.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from config import (ASSET_OPTIONS, STRATEGY_FUNCS, RISK_STYLES,
                    POSITION_SIZE_OPTIONS, SLIPPAGE_OPTIONS,
                    ASSET_CLASS_OPTIONS, INTERVAL_OPTIONS)
from indian_stocks import INDIAN_STOCK_OPTIONS, fetch_stock_data
from data_fetching import (fetch_data, find_data_for_trade_target,
                            fetch_ticker_snapshot, fetch_order_book_snapshot,
                            fetch_recent_trades_snapshot, fetch_all_usdt_symbols)
from core_logic import (
    prep_data, reason_for_signal, get_recommendation, run_track_record,
    get_signal_history, get_consensus, get_all_recommendations,
    get_all_stock_recommendations, run_portfolio_backtest, run_optimizer,
    run_comparative_backtest
)
from journal import load_journal, add_journal_entry
from reports import generate_pdf_report
from display_helpers import (make_candlestick_chart, make_equity_drawdown_chart,
                             make_monte_carlo_chart, make_correlation_heatmap,
                             make_comparative_equity_chart,
                             make_ga_fitness_chart, make_efficient_frontier_chart,
                             make_feature_importance_chart,
                             badge, inject_custom_css)
from metrics import full_report
from alerts import send_telegram_alert, send_discord_alert, format_signal_alert
from paper_trading import (
    get_account_summary, get_paper_trades, get_open_positions,
    execute_paper_order, reset_paper_account
)
from fear_greed import fetch_fear_and_greed_index
from monte_carlo import run_monte_carlo_simulation
from walk_forward import run_walk_forward_analysis
from hft_tester import HFTTesterBot
from genetic_optimizer import GeneticStrategyOptimizer
from portfolio_optimizer import MarkowitzOptimizer
from ml_model import MarketMLPredictor

# Page Configuration
st.set_page_config(page_title="CurrencyKartz — Quantitative Trading Terminal", layout="wide", page_icon="⚡")

# Inject Custom Modern CSS
inject_custom_css()


def resolve_custom_stock_symbol(user_input):
    """NSE tickers need a .NS suffix -- add it if the user left it off."""
    symbol = user_input.strip().upper().replace(" ", "")
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    return symbol


def highlight_recommendation(val):
    if val == "BUY":
        return "color: #00e676; font-weight: 700;"
    elif val == "SELL":
        return "color: #ff1744; font-weight: 700;"
    elif val == "HOLD":
        return "color: #94a3b8; font-weight: 500;"
    return ""


# ---------------------------------------------------------------------------
# Sidebar Settings & Alert Config
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Terminal Settings")
refresh_minutes = st.sidebar.selectbox("Auto-refresh frequency:", [5, 15, 30, 60], index=1)
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

st.sidebar.markdown("---")
with st.sidebar.expander("🔔 Telegram & Discord Push Alerts"):
    st.caption("Configure free alert channels for live signals.")
    tg_token = st.text_input("Telegram Bot Token:", type="password", key="sb_tg_token")
    tg_chat = st.text_input("Telegram Chat ID:", key="sb_tg_chat")
    dc_url = st.text_input("Discord Webhook URL:", type="password", key="sb_dc_url")

    if st.button("Send Test Alert", key="test_alert_btn"):
        test_msg = format_signal_alert("BTCUSDT", "BUY", 65432.10, "MA Crossover", "1h")
        if tg_token and tg_chat:
            tg_res = send_telegram_alert(tg_token, tg_chat, test_msg)
            if tg_res["success"]:
                st.success("Telegram test alert sent!")
            else:
                st.error(f"Telegram error: {tg_res.get('error')}")
        if dc_url:
            dc_res = send_discord_alert(dc_url, test_msg)
            if dc_res["success"]:
                st.success("Discord test alert sent!")
            else:
                st.error(f"Discord error: {dc_res.get('error')}")
        if not (tg_token and tg_chat) and not dc_url:
            st.info("Enter Telegram or Discord credentials above.")

st.sidebar.markdown("---")
st.sidebar.caption(f"🕒 **Last updated:** {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption(f"🔄 Auto-refreshing every {refresh_minutes} min")


# ---------------------------------------------------------------------------
# Modern Hero Banner & Live Fear & Greed Card
# ---------------------------------------------------------------------------
try:
    fg_data = fetch_fear_and_greed_index(limit=1)
    fg_val = fg_data["current_value"]
    fg_class = fg_data["current_classification"]
except Exception:
    fg_val, fg_class = 50, "Neutral"

fg_color = "#00e676" if fg_val >= 60 else ("#ff1744" if fg_val <= 40 else "#fbbf24")

st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 class="hero-title">CurrencyKartz Terminal ⚡</h1>
            <p class="hero-subtitle">Open-Source Quantitative Trading Research & Live Signal Advisor</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; padding: 8px 18px; border-radius: 14px; text-align: center;">
                <span style="font-size: 0.75rem; color: #94a3b8; display: block;">Crypto Fear & Greed</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: {fg_color};">{fg_val} — {fg_class}</span>
            </div>
            <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60a5fa; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                🟢 Binance Public Feed
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabbed Navigation Setup
# ---------------------------------------------------------------------------
tab_signals, tab_deepdive, tab_paper, tab_depth, tab_portfolio, tab_explorer = st.tabs([
    "📊 Live Advisor & Signals",
    "📈 Asset Deep-Dive & Backtest",
    "📝 Paper Trading Simulator",
    "⚡ Order Book & HFT Tester",
    "⚡ Portfolio & Auto-Optimizer",
    "🔍 Symbol Explorer"
])


# ============================================================================
# TAB 1: Live Advisor & Signals
# ============================================================================
with tab_signals:
    col_ov1, col_ov2 = st.columns([3, 1])
    with col_ov1:
        st.markdown("### All Assets at a Glance")
        st.caption("Default model overview across all tracked assets.")
    with col_ov2:
        overview_class_label = st.radio("Asset Class:", list(ASSET_CLASS_OPTIONS.keys()),
                                        horizontal=True, key="overview_class")
        overview_class = ASSET_CLASS_OPTIONS[overview_class_label]

    with st.spinner("Calculating live recommendations across all assets..."):
        if overview_class == "crypto":
            overview_df = get_all_recommendations()
        else:
            overview_df = get_all_stock_recommendations()

    st.dataframe(
        overview_df.style.map(highlight_recommendation, subset=["Recommendation"]),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # Market Snapshot section
    st.markdown("### 📊 Market Snapshot (Free 24h Ticker & Spread)")
    snap_c1, snap_c2 = st.columns([2, 1])
    with snap_c1:
        snap_symbol_label = st.selectbox("Select Asset:", list(ASSET_OPTIONS.keys()), key="snap_asset")
        snap_symbol = ASSET_OPTIONS[snap_symbol_label]
    with snap_c2:
        snap_custom = st.text_input("Or custom symbol:", placeholder="e.g. XRPUSDT", key="snap_custom")
        if snap_custom.strip():
            snap_symbol = snap_custom.strip().upper()
            if not snap_symbol.endswith("USDT"):
                snap_symbol += "USDT"

    try:
        with st.spinner(f"Fetching live snapshot for {snap_symbol}..."):
            snap = fetch_ticker_snapshot(snap_symbol)

        change_pct = snap["price_change_pct"]
        delta_color = "normal" if change_pct >= 0 else "inverse"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Price", f"${snap['last_price']:,.4f}", f"{change_pct:+.2f}% (24h)", delta_color=delta_color)
        m2.metric("24h High", f"${snap['high_24h']:,.4f}")
        m3.metric("24h Low", f"${snap['low_24h']:,.4f}")
        m4.metric("24h Volume (USDT)", f"${snap['quote_volume_24h']:,.0f}")

        b1, b2, b3 = st.columns(3)
        b1.metric("Best Bid", f"${snap['bid_price']:,.4f}", f"qty: {snap['bid_qty']:.4f}")
        b2.metric("Best Ask", f"${snap['ask_price']:,.4f}", f"qty: {snap['ask_qty']:.4f}")
        b3.metric("Bid/Ask Spread", f"${snap['spread']:,.6f}", f"{snap['spread_pct']:.4f}%")
    except Exception as e:
        st.warning(f"Unable to load snapshot for {snap_symbol}: {e}")

    st.markdown("---")

    # Multi-Timeframe Check
    st.markdown("### ⏱️ Multi-Timeframe Signal Consensus")
    mtf_col1, mtf_col2 = st.columns([1, 2])
    with mtf_col1:
        mtf_symbol_label = st.selectbox("Asset to analyze:", list(ASSET_OPTIONS.keys()), key="mtf_asset")
        mtf_symbol = ASSET_OPTIONS[mtf_symbol_label]
        mtf_strategy_label = st.selectbox("Target Strategy:", list(STRATEGY_FUNCS.keys()), key="mtf_strategy")
        mtf_strategy = STRATEGY_FUNCS[mtf_strategy_label]
        mtf_run = st.button("Analyze Timeframes", type="primary", key="mtf_run")

    if mtf_run:
        mtf_intervals = {"1 Hour": "1h", "4 Hours": "4h", "1 Day": "1d"}
        mtf_results = []
        with st.spinner("Checking timeframes..."):
            for label, iv in mtf_intervals.items():
                try:
                    df_tf = fetch_data(mtf_symbol, interval=iv, limit=200)
                    df_tf = prep_data(df_tf)
                    sig_df = mtf_strategy(df_tf)
                    latest_sig = sig_df.iloc[-1]["signal"]
                    latest_price = sig_df.iloc[-1]["close"]
                    mtf_results.append({
                        "Timeframe": label,
                        "Interval": iv,
                        "Signal": latest_sig,
                        "Latest Close": f"${latest_price:,.4f}",
                    })
                except Exception as e:
                    mtf_results.append({"Timeframe": label, "Interval": iv, "Signal": "ERROR", "Latest Close": str(e)})

        mtf_df = pd.DataFrame(mtf_results)
        with mtf_col2:
            st.dataframe(
                mtf_df.style.map(highlight_recommendation, subset=["Signal"]),
                use_container_width=True, hide_index=True
            )
            buys = sum(1 for r in mtf_results if r["Signal"] == "BUY")
            sells = sum(1 for r in mtf_results if r["Signal"] == "SELL")
            if buys == len(mtf_intervals):
                st.success("🎯 Strong Consensus: All timeframes confirm a BUY signal.")
            elif sells == len(mtf_intervals):
                st.error("⚠️ Strong Consensus: All timeframes confirm a SELL signal.")
            else:
                st.info("ℹ️ Mixed Timeframe Alignment: Exercise caution.")


# ============================================================================
# TAB 2: Asset Deep-Dive & Backtest
# ============================================================================
with tab_deepdive:
    st.markdown("### Single Asset Deep-Dive & Strategy Controls")

    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        asset_class_label = st.radio("Asset Class:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True, key="dd_class")
        asset_class = ASSET_CLASS_OPTIONS[asset_class_label]

        if asset_class == "crypto":
            asset_label = st.selectbox("Select Asset:", list(ASSET_OPTIONS.keys()) + ["Custom (type symbol)"], key="dd_asset_crypto")
        else:
            asset_label = st.selectbox("Select Stock:", list(INDIAN_STOCK_OPTIONS.keys()) + ["Custom (type NSE ticker)"], key="dd_asset_stock")

    with ctrl2:
        if asset_class == "crypto" and asset_label == "Custom (type symbol)":
            custom_input = st.text_input("Binance Symbol:", placeholder="e.g. DOGE, XRP, AVAX", key="dd_custom_crypto")
            if not custom_input:
                st.info("Enter a symbol to proceed.")
                st.stop()
            symbol = custom_input.strip().upper().replace(" ", "")
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            asset_label = symbol
        elif asset_class == "stocks" and asset_label == "Custom (type NSE ticker)":
            custom_stock_input = st.text_input("NSE Ticker:", placeholder="e.g. TCS, RELIANCE", key="dd_custom_stock")
            if not custom_stock_input:
                st.info("Enter an NSE ticker to proceed.")
                st.stop()
            symbol = resolve_custom_stock_symbol(custom_stock_input)
            asset_label = symbol
        elif asset_class == "crypto":
            symbol = ASSET_OPTIONS[asset_label]
        else:
            symbol = INDIAN_STOCK_OPTIONS[asset_label]

    # Model parameters card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    scol1, scol2, scol3, scol4 = st.columns(4)
    strategy_choice = scol1.selectbox("Strategy Model:", list(STRATEGY_FUNCS.keys()), key=f"strategy_{symbol}")
    risk_choice = scol2.selectbox("Risk Management:", list(RISK_STYLES.keys()), index=2, key=f"risk_{symbol}")
    sizing_choice = scol3.selectbox("Position Sizing:", list(POSITION_SIZE_OPTIONS.keys()), key=f"sizing_{symbol}")
    slippage_choice = scol4.selectbox("Slippage Model:", list(SLIPPAGE_OPTIONS.keys()), index=1, key=f"slippage_{symbol}")
    st.markdown("</div>", unsafe_allow_html=True)

    strategy_func = STRATEGY_FUNCS[strategy_choice]
    risk_kwargs = dict(RISK_STYLES[risk_choice])
    risk_per_trade_pct = POSITION_SIZE_OPTIONS[sizing_choice]
    risk_kwargs["slippage_pct"] = SLIPPAGE_OPTIONS[slippage_choice]

    if risk_per_trade_pct is not None:
        if risk_choice == "No stop-loss":
            st.warning("Position sizing requires a stop-loss. Defaulting to 'All-in per trade'.")
        else:
            risk_kwargs["risk_per_trade_pct"] = risk_per_trade_pct

    if asset_class == "crypto":
        mode = st.radio("Backtest Window Mode:", ["By number of days", "By number of trades"], horizontal=True, key=f"mode_{symbol}")
    else:
        mode = "By number of days"

    if mode == "By number of days":
        lookback_days = st.slider("Lookback Days:", min_value=60, max_value=1000, value=200, step=10, key=f"lookback_{symbol}")
        with st.spinner(f"Fetching candles for {symbol}..."):
            if asset_class == "crypto":
                df = fetch_data(symbol, limit=lookback_days)
            else:
                df = fetch_stock_data(symbol, limit=lookback_days)
            df = prep_data(df)
            recommendation, reason, latest, signal_df = get_recommendation(df, strategy_func)
        window_label = f"last {lookback_days} days"
    else:
        target_trades = st.number_input("Target Trades:", min_value=5, max_value=500, value=20, step=5, key=f"target_trades_{symbol}")
        with st.spinner(f"Paging history for {target_trades} trades..."):
            signal_df, achieved_trades, days_used = find_data_for_trade_target(symbol, strategy_func, target_trades, prep_data)
            latest = signal_df.iloc[-1]
            recommendation = latest["signal"]
            reason = reason_for_signal(recommendation)
            df = signal_df
        window_label = f"{achieved_trades} trades ({days_used} days)"

    currency_symbol = "Rs. " if asset_class == "stocks" else "$"

    # Recommendation Header Card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    rec_col1, rec_col2 = st.columns([1, 3])
    with rec_col1:
        st.markdown(f"### Signal: {badge(recommendation, recommendation)}", unsafe_allow_html=True)
        st.caption(f"Regime: {badge(latest['regime'], latest['regime'])}", unsafe_allow_html=True)
    with rec_col2:
        st.markdown(f"**Rationale:** {reason}")
        st.caption(f"Using **{strategy_choice}** with **{risk_choice}** on **{asset_label}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Metrics Grid
    km1, km2, km3, km4 = st.columns(4)
    km1.metric("Close Price", f"{currency_symbol}{latest['close']:,.2f}")
    km2.metric("MA 20", f"{currency_symbol}{latest['MA_20']:,.2f}")
    km3.metric("MA 50", f"{currency_symbol}{latest['MA_50']:,.2f}")
    km4.metric("RSI (14)", f"{latest['RSI']:.1f}")

    # TradingView-style Dual-Panel Candlestick Chart
    st.markdown("#### Interactive Technical Chart (Price & Volume)")
    st.plotly_chart(
        make_candlestick_chart(df.tail(300), title=f"{asset_label} — Price & Volume"),
        use_container_width=True
    )

    # Track Record Performance Backtest & Risk Analytics (Feature 1)
    st.markdown(f"### Historical Backtest & Quantitative Risk Analytics ({window_label})")
    track_results = run_track_record(signal_df, risk_kwargs)

    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Strategy Return", f"{track_results['strategy_return_pct']:.2f}%")
    tr2.metric("Buy & Hold Benchmark", f"{track_results['buy_hold_return_pct']:.2f}%")
    tr3.metric("Total Trades Executed", track_results['num_trades'])

    # Advanced Quant Metrics Card (Sharpe, Sortino, Calmar, Profit Factor)
    qmetrics = full_report(track_results)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 📐 Quantitative Risk-Adjusted Ratios")
    qr1, qr2, qr3, qr4, qr5 = st.columns(5)
    qr1.metric("Sharpe Ratio", f"{qmetrics['Sharpe Ratio']:.2f}" if qmetrics['Sharpe Ratio'] else "N/A")
    qr2.metric("Sortino Ratio", f"{qmetrics['Sortino Ratio']:.2f}" if qmetrics['Sortino Ratio'] else "N/A")
    qr3.metric("Calmar Ratio", f"{qmetrics['Calmar Ratio']:.2f}" if qmetrics['Calmar Ratio'] else "N/A")
    qr4.metric("Profit Factor", f"{qmetrics['profit_factor']:.2f}" if qmetrics['profit_factor'] else "N/A")
    qr5.metric("Win Rate", f"{qmetrics['win_rate_pct']:.1f}%" if qmetrics['win_rate_pct'] else "N/A")
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Equity Curve & Underwater Drawdown Chart
    st.markdown("#### Cumulative Equity Curve & Underwater Drawdown Chart")
    st.plotly_chart(
        make_equity_drawdown_chart(track_results, title=f"{asset_label} Cumulative Equity vs Benchmark"),
        use_container_width=True
    )

    # Upgrade 1: Monte Carlo Simulation & VaR Stress Tester
    with st.expander("🎲 Run Monte Carlo Stress Tester (1,000 Path Simulations)"):
        with st.spinner("Simulating 1,000 future equity paths..."):
            mc_res = run_monte_carlo_simulation(track_results, num_simulations=1000, num_days=90)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("95% 1-Day VaR (Risk)", f"{mc_res['var_95']:.2f}%")
        mc2.metric("99% 1-Day VaR (Extreme Risk)", f"{mc_res['var_99']:.2f}%")
        mc3.metric("Probability of >30% Drawdown", f"{mc_res['ruin_probability_pct']:.1f}%")

        if mc_res and "percentiles" in mc_res:
            st.plotly_chart(make_monte_carlo_chart(mc_res, title=f"{asset_label} Monte Carlo 90-Day Simulation Fan Chart"), use_container_width=True)

    # Upgrade 3: Walk-Forward Optimization & Overfitting Guard
    with st.expander("🔬 Run Walk-Forward Rolling Train/Test Validation"):
        if st.button("Run Walk-Forward Analysis", type="primary", key="wf_btn"):
            with st.spinner("Evaluating rolling train/test windows..."):
                wf_res = run_walk_forward_analysis(df, strategy_func, risk_kwargs, train_days=120, test_days=40)
            wf1, wf2, wf3 = st.columns(3)
            wf1.metric("In-Sample Return (Train)", f"{wf_res['in_sample_avg_return']:.2f}%")
            wf2.metric("Out-of-Sample Return (Test)", f"{wf_res['out_of_sample_avg_return']:.2f}%")
            wf3.metric("Efficiency Ratio", f"{wf_res['efficiency_ratio']:.2f}")

            if wf_res["is_robust"]:
                st.success("✅ Robust Strategy: Out-of-sample returns remain positive across market regimes.")
            else:
                st.warning("⚠️ Overfitting Risk: Out-of-sample performance drops significantly vs training window.")

            if len(wf_res["windows_df"]) > 0:
                st.dataframe(wf_res["windows_df"], use_container_width=True, hide_index=True)

    # Upgrade 4: Multi-Strategy Comparative Backtest Tool
    with st.expander("⚔️ Compare 3 Strategies Head-to-Head"):
        st.markdown("#### Select up to 3 strategies to compare on the same asset & time window")
        all_strategy_names = list(STRATEGY_FUNCS.keys())
        comp_strats = st.multiselect(
            "Choose strategies to compare:", all_strategy_names,
            default=all_strategy_names[:3] if len(all_strategy_names) >= 3 else all_strategy_names,
            max_selections=3, key="comp_strats"
        )
        if st.button("⚔️ Run Comparison", type="primary", key="comp_btn"):
            if len(comp_strats) < 2:
                st.warning("Please select at least 2 strategies to compare.")
            else:
                with st.spinner(f"Backtesting {', '.join(comp_strats)} on {asset_label}..."):
                    comp_res = run_comparative_backtest(df, comp_strats, risk_kwargs)
                if comp_res and "equity_df" in comp_res and len(comp_res["equity_df"]) > 0:
                    st.plotly_chart(
                        make_comparative_equity_chart(
                            comp_res["equity_df"],
                            title=f"{asset_label} — Strategy Equity Comparison"
                        ),
                        use_container_width=True
                    )
                    if "summary" in comp_res and len(comp_res["summary"]) > 0:
                        st.markdown("#### 📊 Performance Summary")
                        st.dataframe(comp_res["summary"], use_container_width=True, hide_index=True)
                else:
                    st.info("No backtest data returned. Try selecting different strategies.")

    # Next-Level Feature 1: Genetic Algorithm Strategy Auto-Tuner
    with st.expander("🧬 Run Genetic Algorithm (GA) Strategy Auto-Tuner"):
        st.markdown("#### Discover Optimal Strategy Parameters via Evolutionary Natural Selection")
        st.caption("Evolves fast/slow MA periods, stop-loss %, and trailing stops over generations to maximize the Sharpe Ratio.")

        ga_c1, ga_c2, ga_c3 = st.columns(3)
        with ga_c1:
            ga_pop = st.slider("Population Size (Chromosomes):", min_value=12, max_value=36, value=20, step=4, key="ga_pop")
        with ga_c2:
            ga_gens = st.slider("Evolution Generations:", min_value=5, max_value=20, value=10, step=1, key="ga_gens")
        with ga_c3:
            ga_mut = st.slider("Mutation Rate (%):", min_value=10, max_value=40, value=20, step=5, key="ga_mut") / 100.0

        if st.button("🧬 Start Genetic Evolution", type="primary", key="btn_ga"):
            with st.spinner(f"Evolving {ga_pop} strategies across {ga_gens} generations..."):
                ga_opt = GeneticStrategyOptimizer(
                    df, population_size=ga_pop, generations=ga_gens, mutation_rate=ga_mut
                )
                ga_res = ga_opt.optimize()

            best_p = ga_res["best_params"]
            best_m = ga_res["best_metrics"]

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Optimized Return", f"{best_m.get('return_pct', 0.0):+.2f}%")
            g2.metric("Discovered Sharpe Ratio", f"{best_m.get('sharpe_ratio', 0.0):.2f}")
            g3.metric("Max Drawdown", f"{best_m.get('max_drawdown_pct', 0.0):.2f}%")
            g4.metric("Trades Executed", f"{best_m.get('trades', 0)}")

            st.success(f"🏆 Best Evolved Parameters: Fast MA = **{best_p['fast_ma']}**, Slow MA = **{best_p['slow_ma']}**, Stop-Loss = **{best_p['stop_loss_pct']*100:.1f}%**, Trailing Stop = **{best_p['trailing_stop_pct']*100:.1f}%**")

            st.plotly_chart(
                make_ga_fitness_chart(ga_res["history_df"], title=f"{asset_label} — Genetic Algorithm Fitness Evolution"),
                use_container_width=True
            )

    # Next-Level Feature 2: Machine Learning Direction & Feature Importance
    with st.expander("🤖 Run Machine Learning Direction & Regime Predictor"):
        st.markdown("#### Random Forest Next-Candle Directional Forecast")
        st.caption("Trains a Random Forest classifier on technical features using a strict temporal split (no future lookahead bias).")

        if st.button("🔮 Train ML Model & Forecast Next Candle", type="primary", key="btn_ml"):
            with st.spinner(f"Training Random Forest on {asset_label} indicator features..."):
                ml_pred = MarketMLPredictor(df)
                ml_res = ml_pred.train_and_predict()

            m1, m2, m3 = st.columns(3)
            m1.metric("Forecast Direction", ml_res["next_direction"])
            m2.metric("Bullish Probability", f"{ml_res['bullish_prob_pct']:.1f}%")
            m3.metric("Out-of-Sample Test Accuracy", f"{ml_res['test_accuracy_pct']:.1f}%")

            if not ml_res["feature_importance_df"].empty:
                st.plotly_chart(
                    make_feature_importance_chart(
                        ml_res["feature_importance_df"],
                        title=f"{asset_label} — Technical Indicator Predictive Importance"
                    ),
                    use_container_width=True
                )

    # Export options
    exp1, exp2 = st.columns(2)
    trade_log_df = pd.DataFrame(track_results["trade_log"])
    if len(trade_log_df) > 0:
        exp1.download_button("📥 Download Trade Log (CSV)", data=trade_log_df.to_csv(index=False),
                             file_name=f"{symbol}_{strategy_choice}_trade_log.csv", mime="text/csv")

    pdf_bytes = generate_pdf_report(asset_label, symbol, strategy_choice, risk_choice, slippage_choice,
                                    recommendation, reason, latest, track_results, window_label,
                                    currency=currency_symbol)
    exp2.download_button("📄 Export Summary Report (PDF)", data=pdf_bytes,
                         file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf")

    # Trade Journal Section
    st.markdown("---")
    st.markdown("### 📓 Integrated Trade Journal")
    with st.form(key=f"journal_form_{symbol}"):
        jcol1, jcol2, jcol3 = st.columns(3)
        entry_date = jcol1.date_input("Date", value=datetime.now())
        entry_action = jcol2.selectbox("Action Taken", ["BUY", "SELL", "HOLD (no action)"])
        entry_price = jcol3.number_input("Execution Price", value=float(latest["close"]), format="%.2f")
        entry_notes = st.text_input("Notes / Strategy Rationale", placeholder="e.g. Entered half position on pullback")
        submitted = st.form_submit_button("Log Trade Entry", type="primary")
        if submitted:
            add_journal_entry(entry_date, asset_label, strategy_choice, entry_action, entry_price, entry_notes)
            st.success("Trade entry logged successfully.")

    journal = load_journal()
    asset_journal = journal[journal["Asset"] == asset_label] if len(journal) > 0 else journal
    if len(asset_journal) > 0:
        st.dataframe(asset_journal.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.caption("No trade journal entries for this asset yet.")


# ============================================================================
# TAB 3: Paper Trading Simulator (Feature 4 & NSE Support)
# ============================================================================
with tab_paper:
    st.markdown("### 📝 Paper Trading Simulation Engine")
    st.caption("Simulate live execution with a virtual $10,000 wallet — zero risk.")

    acc_summary = get_account_summary()

    # Account Balance Summary Cards
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Cash Balance", f"${acc_summary['cash_balance']:,.2f}")
    pc2.metric("Position Value", f"${acc_summary['position_value']:,.2f}")
    pc3.metric("Total Equity", f"${acc_summary['total_equity']:,.2f}", f"{acc_summary['total_pnl_pct']:+.2f}%")
    pc4.metric("Realized PnL", f"${acc_summary['realized_pnl']:,.2f}")

    st.markdown("---")

    # Order Entry Panel
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        st.markdown("#### ⚡ Order Entry")
        po_class_label = st.radio("Asset Class:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True, key="po_class")
        po_class = ASSET_CLASS_OPTIONS[po_class_label]
        po_currency = "$" if po_class == "crypto" else "Rs. "

        if po_class == "crypto":
            po_options = ASSET_OPTIONS
            po_asset_label = st.selectbox("Asset:", list(po_options.keys()) + ["Custom (type symbol)"], key="po_asset_crypto")
            if po_asset_label == "Custom (type symbol)":
                custom_po = st.text_input("Custom Symbol:", placeholder="e.g. DOGE, XRP", key="po_custom_c")
                po_symbol = (custom_po.strip().upper() + "USDT") if custom_po and not custom_po.endswith("USDT") else custom_po.strip().upper()
            else:
                po_symbol = po_options[po_asset_label]
        else:
            po_options = INDIAN_STOCK_OPTIONS
            po_asset_label = st.selectbox("Stock:", list(po_options.keys()) + ["Custom (type NSE ticker)"], key="po_asset_stock")
            if po_asset_label == "Custom (type NSE ticker)":
                custom_po_s = st.text_input("Custom Ticker:", placeholder="e.g. TCS, RELIANCE", key="po_custom_s")
                po_symbol = resolve_custom_stock_symbol(custom_po_s) if custom_po_s else ""
            else:
                po_symbol = po_options[po_asset_label]

        with st.form("paper_order_form"):
            po_action = st.selectbox("Action:", ["BUY", "SELL"])

            # Spot price lookup for Crypto vs Stocks
            po_spot = 100.0
            if po_symbol:
                try:
                    if po_class == "crypto":
                        po_spot = fetch_ticker_snapshot(po_symbol)["last_price"]
                    else:
                        st_df = fetch_stock_data(po_symbol, limit=5)
                        if len(st_df) > 0:
                            po_spot = float(st_df.iloc[-1]["close"])
                except Exception:
                    po_spot = 100.0

            po_price = st.number_input(f"Order Price ({po_currency}):", value=float(po_spot), format="%.2f")
            po_amount = st.number_input(f"Order Capital Value ({po_currency}):", value=1000.0 if po_class == "crypto" else 10000.0, step=500.0)
            po_notes = st.text_input("Order Notes:", placeholder="Paper trade test")

            po_submit = st.form_submit_button("Submit Paper Order", type="primary")

            if po_submit:
                if not po_symbol:
                    st.error("Please provide a valid asset symbol.")
                else:
                    res = execute_paper_order(po_symbol, po_action, po_price, amount_usd=po_amount, notes=f"[{po_class_label}] {po_notes}")
                    if res["success"]:
                        st.success(f"Order executed! {po_action} {po_symbol} at {po_currency}{po_price:,.2f}")
                        st.rerun()
                    else:
                        st.error(res["error"])

    with p_col2:
        st.markdown("#### 📊 Open Positions")
        open_pos = get_open_positions()
        if open_pos:
            df_open = pd.DataFrame(open_pos)[["symbol", "qty", "avg_entry_price", "current_value", "unrealized_pnl", "unrealized_pnl_pct"]]
            df_open.columns = ["Symbol", "Qty", "Avg Entry Price", "Current Value", "Unrealized PnL ($)", "PnL %"]
            st.dataframe(df_open, use_container_width=True, hide_index=True)
        else:
            st.info("No open paper positions.")

        st.markdown("#### 📜 Order History Log")
        trades_history = get_paper_trades()
        if len(trades_history) > 0:
            st.dataframe(trades_history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No paper orders executed yet.")

    st.markdown("---")
    if st.button("🔄 Reset Paper Wallet to $10,000"):
        reset_paper_account(10000.0)
        st.success("Paper account reset successfully.")
        st.rerun()


# ============================================================================
# TAB 4: Order Book & Time-Sales
# ============================================================================
with tab_depth:
    st.markdown("### Live Order Book & Time & Sales Tape")
    st.caption("Public market depth and trade stream — 100% free Binance API endpoints.")

    ob_col1, ob_col2 = st.columns([1, 2])
    with ob_col1:
        ob_symbol_label = st.selectbox("Target Symbol:", list(ASSET_OPTIONS.keys()), key="depth_tab_asset")
        ob_symbol = ASSET_OPTIONS[ob_symbol_label]
        ob_depth = st.select_slider("Depth Levels:", options=[5, 10, 20, 50], value=10, key="depth_tab_levels")

    try:
        with st.spinner("Fetching market depth..."):
            ob = fetch_order_book_snapshot(ob_symbol, limit=ob_depth)

        bids_df = pd.DataFrame(ob["bids"]).rename(columns={"price": "Price", "qty": "Qty"})
        asks_df = pd.DataFrame(ob["asks"]).rename(columns={"price": "Price", "qty": "Qty"})

        bids_df["Cumulative Qty"] = bids_df["Qty"].cumsum()
        asks_df["Cumulative Qty"] = asks_df["Qty"].cumsum()

        fig_ob = go.Figure()
        fig_ob.add_trace(go.Scatter(
            x=bids_df["Price"], y=bids_df["Cumulative Qty"], fill="tozeroy", name="Bids",
            line=dict(color="#00e676"), fillcolor="rgba(0,230,118,0.15)"
        ))
        fig_ob.add_trace(go.Scatter(
            x=asks_df["Price"], y=asks_df["Cumulative Qty"], fill="tozeroy", name="Asks",
            line=dict(color="#ff1744"), fillcolor="rgba(255,23,68,0.15)"
        ))
        fig_ob.update_layout(
            template="plotly_dark", title=f"{ob_symbol} Order Book Depth",
            xaxis_title="Price (USDT)", yaxis_title="Cumulative Depth",
            height=380, margin=dict(l=10, r=10, t=35, b=10)
        )
        with ob_col2:
            st.plotly_chart(fig_ob, use_container_width=True)

        dt1, dt2 = st.columns(2)
        dt1.caption("**Top Bids**")
        dt1.dataframe(bids_df[["Price", "Qty"]].head(5), use_container_width=True, hide_index=True)
        dt2.caption("**Top Asks**")
        dt2.dataframe(asks_df[["Price", "Qty"]].head(5), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Unable to load order book: {e}")

    st.markdown("---")
    st.markdown("### Recent Public Trades (Time & Sales)")
    try:
        with st.spinner("Fetching public trades..."):
            trades = fetch_recent_trades_snapshot(ob_symbol, limit=25)

        trades_df = pd.DataFrame(trades)[["time", "side", "price", "qty", "quote_qty"]]
        trades_df.columns = ["Time", "Side", "Price", "Qty", "Volume (USDT)"]
        trades_df["Price"] = trades_df["Price"].map(lambda x: f"${x:,.4f}")
        trades_df["Volume (USDT)"] = trades_df["Volume (USDT)"].map(lambda x: f"${x:,.2f}")

        def _color_side(val):
            return "color: #00e676; font-weight: 700;" if val == "BUY" else "color: #ff1744; font-weight: 700;"

        st.dataframe(trades_df.style.map(_color_side, subset=["Side"]), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Unable to load public trades: {e}")

    st.markdown("---")
    st.markdown("### ⚡ High-Frequency Microstructure Virtual Tester")
    st.caption("Zero-risk virtual testing engine: Simulates high-speed Level-2 Order Book Imbalance (OBI) scalping on real-time market data with realistic exchange fee modeling.")

    with st.expander("🛠️ HFT Virtual Tester Parameters & Strategy Setup", expanded=True):
        hft_c1, hft_c2, hft_c3 = st.columns(3)
        with hft_c1:
            hft_capital = st.number_input("Virtual Starting Capital ($):", value=1000.0, step=100.0, key="hft_cap")
            hft_size = st.number_input("Micro-Order Size ($):", value=100.0, step=25.0, key="hft_size")
        with hft_c2:
            hft_obi_thresh = st.slider("OBI Trigger Threshold (Buy Pressure):", min_value=0.10, max_value=0.80, value=0.35, step=0.05, key="hft_obi")
            hft_fee = st.number_input("Exchange Taker Fee (%):", value=0.075, step=0.01, format="%.3f", key="hft_fee") / 100.0
        with hft_c3:
            hft_tp = st.slider("Take Profit Target (%):", min_value=0.05, max_value=0.60, value=0.15, step=0.05, key="hft_tp") / 100.0
            hft_sl = st.slider("Stop Loss Target (%):", min_value=0.05, max_value=0.50, value=0.10, step=0.05, key="hft_sl") / 100.0

        run_hft_sim = st.button("🚀 Run HFT Simulation on Live Market Data", type="primary", key="btn_run_hft")

    if run_hft_sim:
        try:
            with st.spinner("Fetching live tick feeds & simulating micro-executions..."):
                sim_ob = fetch_order_book_snapshot(ob_symbol, limit=20)
                sim_trades = fetch_recent_trades_snapshot(ob_symbol, limit=50)

                bot = HFTTesterBot(
                    initial_capital=hft_capital,
                    order_size_usd=hft_size,
                    imbalance_threshold=hft_obi_thresh,
                    take_profit_pct=hft_tp,
                    stop_loss_pct=hft_sl,
                    fee_rate=hft_fee,
                )
                sim_df = bot.run_replay_simulation(sim_ob, sim_trades)

            if not sim_df.empty:
                last_tick = sim_df.iloc[-1]
                tot_pnl = last_tick["total_pnl"]
                tot_pnl_pct = last_tick["total_pnl_pct"]

                # Metric Cards
                hm1, hm2, hm3, hm4 = st.columns(4)
                hm1.metric("Virtual Portfolio Equity", f"${last_tick['current_equity']:,.2f}", f"{tot_pnl_pct:+.2f}%")
                hm2.metric("Simulated Net PnL", f"${tot_pnl:+.2f}")
                hm3.metric("Win Rate", f"{last_tick['win_rate']:.1f}%", f"{bot.winning_trades}/{bot.total_trades} wins")
                hm4.metric("Total Exchange Fees Paid", f"${last_tick['total_fees']:.4f}")

                # Live Order Book Imbalance (OBI) Gauge
                live_obi = last_tick["obi"]
                obi_color = "#00e676" if live_obi > 0.1 else ("#ff1744" if live_obi < -0.1 else "#fbbf24")
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 12px; padding: 14px 20px; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.95rem; color: #94a3b8;">Current Order Book Imbalance (OBI):</span>
                        <span style="font-weight: 700; font-size: 1.15rem; color: {obi_color};">{live_obi:+.3f} ({'Heavy Buyers' if live_obi > 0.2 else ('Heavy Sellers' if live_obi < -0.2 else 'Balanced Queues')})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Simulated Equity Trajectory Plot
                fig_hft = go.Figure()
                fig_hft.add_trace(go.Scatter(
                    y=sim_df["current_equity"],
                    mode="lines",
                    name="Virtual Equity ($)",
                    line=dict(color="#38bdf8", width=2.5)
                ))
                fig_hft.update_layout(
                    template="plotly_dark",
                    title=f"{ob_symbol} HFT Virtual Equity Trajectory (Live Tick Replay)",
                    xaxis_title="Tick Step",
                    yaxis_title="Portfolio Equity ($)",
                    height=360,
                    margin=dict(l=10, r=10, t=35, b=10)
                )
                st.plotly_chart(fig_hft, use_container_width=True)

                # Micro Trade Execution Log
                if len(bot.trade_log) > 0:
                    st.markdown("#### ⚡ Micro-Trade Execution Log")
                    trade_df = pd.DataFrame(bot.trade_log)
                    def _hft_action_style(val):
                        return "color: #00e676; font-weight: 700;" if val == "BUY" else "color: #ff1744; font-weight: 700;"
                    st.dataframe(trade_df.style.map(_hft_action_style, subset=["action"]), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No trades triggered on these ticks. The OBI did not cross the +{hft_obi_thresh:.2f} threshold. Try lowering the OBI Trigger Threshold.")
            else:
                st.warning("Insufficient tick data received from Binance public feeds. Please retry.")
        except Exception as e:
            st.error(f"HFT simulation error: {e}")


# ============================================================================
# TAB 5: Portfolio, Auto-Optimizer & Cross-Asset Correlation
# ============================================================================
with tab_portfolio:
    st.markdown("### Multi-Asset Portfolio Backtester")
    pf_class_label = st.radio("Asset Class:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True, key="pf_class_tab")
    pf_class = ASSET_CLASS_OPTIONS[pf_class_label]
    pf_options = ASSET_OPTIONS if pf_class == "crypto" else INDIAN_STOCK_OPTIONS
    pf_fetch_func = fetch_data if pf_class == "crypto" else fetch_stock_data
    pf_currency = "$" if pf_class == "crypto" else "Rs. "

    pf_default = list(pf_options.keys())[:3]
    pf_assets = st.multiselect("Included Assets:", list(pf_options.keys()), default=pf_default, key="pf_assets_tab")

    pcol1, pcol2, pcol3 = st.columns(3)
    pf_strategy_choice = pcol1.selectbox("Portfolio Strategy:", list(STRATEGY_FUNCS.keys()), key="pf_strategy_tab")
    pf_risk_choice = pcol2.selectbox("Risk Style:", list(RISK_STYLES.keys()), index=2, key="pf_risk_tab")
    pf_capital = pcol3.number_input("Starting Capital:", min_value=100.0, value=10000.0, step=500.0, key="pf_capital_tab")

    if st.button("Run Portfolio Backtest", type="primary", key="run_pf_btn"):
        if len(pf_assets) == 0:
            st.warning("Please select at least one asset.")
        else:
            pf_symbols = [pf_options[a] for a in pf_assets]
            pf_func = STRATEGY_FUNCS[pf_strategy_choice]
            pf_risk_kwargs = dict(RISK_STYLES[pf_risk_choice])

            with st.spinner("Simulating portfolio backtest..."):
                pf_results = run_portfolio_backtest(pf_symbols, pf_func, pf_risk_kwargs, pf_capital, fetch_func=pf_fetch_func)

            st.dataframe(pd.DataFrame(pf_results["rows"]), use_container_width=True, hide_index=True)

            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Final Portfolio Value", f"{pf_currency}{pf_results['total_final']:,.2f}", f"{pf_results['portfolio_return_pct']:.2f}%")
            rcol2.metric("Blended Buy & Hold Value", f"{pf_currency}{pf_results['total_buyhold']:,.2f}", f"{pf_results['buyhold_return_pct']:.2f}%")

    st.markdown("---")
    # Upgrade 4: Cross-Asset Correlation Heatmap
    st.markdown("### 📊 Cross-Asset Correlation Matrix")
    st.caption("Inspect live price return correlations across multiple assets to diversify your portfolio.")

    corr_assets = st.multiselect("Assets to correlate:", list(ASSET_OPTIONS.keys()), default=list(ASSET_OPTIONS.keys())[:5], key="corr_assets_select")
    if len(corr_assets) >= 2:
        price_dict = {}
        with st.spinner("Calculating correlation matrix..."):
            for a in corr_assets:
                sym = ASSET_OPTIONS[a]
                try:
                    df_c = fetch_data(sym, limit=100)
                    price_dict[a] = df_c["close"].pct_change()
                except Exception:
                    pass
        if price_dict:
            corr_df = pd.DataFrame(price_dict).corr()
            st.plotly_chart(make_correlation_heatmap(corr_df, title="Price Return Correlation Matrix"), use_container_width=True)
    else:
        st.info("Select at least 2 assets to render the correlation matrix.")

    st.markdown("---")
    # Next-Level Feature 3: Markowitz Modern Portfolio Theory & Efficient Frontier
    st.markdown("### 📈 Markowitz Modern Portfolio Theory & Efficient Frontier")
    st.caption("Calculate optimal capital allocations (Max Sharpe & Minimum Volatility) using quadratic optimization (SciPy SLSQP) across 2,000 simulated portfolios.")

    mpt_class_label = st.radio("Asset Universe for MPT:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True, key="mpt_class_radio")
    mpt_class = ASSET_CLASS_OPTIONS[mpt_class_label]
    mpt_opts = ASSET_OPTIONS if mpt_class == "crypto" else INDIAN_STOCK_OPTIONS
    mpt_fetch = fetch_data if mpt_class == "crypto" else fetch_stock_data

    mpt_selected = st.multiselect("Select assets for portfolio optimization (2 to 6):", list(mpt_opts.keys()), default=list(mpt_opts.keys())[:4], key="mpt_assets_select")

    if st.button("🎯 Calculate Markowitz Efficient Frontier", type="primary", key="btn_mpt"):
        if len(mpt_selected) < 2:
            st.warning("Please select at least 2 assets to optimize.")
        else:
            with st.spinner("Fetching historical prices and solving quadratic optimization..."):
                price_series = {}
                for a in mpt_selected:
                    sym = mpt_opts[a]
                    try:
                        df_a = mpt_fetch(sym, limit=180)
                        if len(df_a) > 20:
                            price_series[a] = df_a["close"].values
                    except Exception:
                        pass

                if price_series and len(price_series) >= 2:
                    min_len = min(len(v) for v in price_series.values())
                    aligned_prices = pd.DataFrame({k: v[-min_len:] for k, v in price_series.items()})

                    mpt = MarkowitzOptimizer(aligned_prices, trading_days=365 if mpt_class == "crypto" else 252)
                    max_s = mpt.optimize_max_sharpe()
                    min_v = mpt.optimize_min_volatility()
                    mc_df = mpt.generate_monte_carlo_frontier(num_portfolios=2000)

                    # Display metrics
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.markdown("#### ⭐ Tangency Portfolio (Max Sharpe)")
                        st.metric("Expected Annual Return", f"{max_s['return_pct']:.2f}%", f"Sharpe: {max_s['sharpe_ratio']:.2f}")
                        st.metric("Annualized Volatility (Risk)", f"{max_s['volatility_pct']:.2f}%")
                        st.write("**Optimal Capital Allocation Weights:**")
                        st.dataframe(pd.DataFrame(list(max_s["weights"].items()), columns=["Asset", "Weight %"]), use_container_width=True, hide_index=True)

                    with col_s2:
                        st.markdown("#### 🛡️ Minimum Volatility Portfolio")
                        st.metric("Expected Annual Return", f"{min_v['return_pct']:.2f}%", f"Sharpe: {min_v['sharpe_ratio']:.2f}")
                        st.metric("Annualized Volatility (Risk)", f"{min_v['volatility_pct']:.2f}%")
                        st.write("**Lowest Drawdown Allocation Weights:**")
                        st.dataframe(pd.DataFrame(list(min_v["weights"].items()), columns=["Asset", "Weight %"]), use_container_width=True, hide_index=True)

                    st.plotly_chart(
                        make_efficient_frontier_chart(mc_df, max_s, min_v, title=f"Markowitz Efficient Frontier — {mpt_class_label}"),
                        use_container_width=True
                    )
                else:
                    st.warning("Insufficient historical price data retrieved. Please select different assets.")

    st.markdown("---")
    st.markdown("### Automated Strategy Optimizer")
    opt_class_label = st.radio("Asset Class:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True, key="opt_class_tab")
    opt_class = ASSET_CLASS_OPTIONS[opt_class_label]
    opt_options = ASSET_OPTIONS if opt_class == "crypto" else INDIAN_STOCK_OPTIONS
    opt_fetch_func = fetch_data if opt_class == "crypto" else fetch_stock_data

    opt_asset = st.selectbox("Asset to Optimize:", list(opt_options.keys()), key="opt_asset_tab")
    opt_days = st.slider("Optimization Lookback (Days):", min_value=60, max_value=1000, value=200, step=10, key="opt_days_tab")

    if st.button("Run Combinatorial Optimizer", type="primary", key="run_opt_btn"):
        opt_symbol = opt_options[opt_asset]
        with st.spinner("Evaluating all strategy x risk combinations..."):
            opt_results = run_optimizer(opt_symbol, opt_days, fetch_func=opt_fetch_func)

        st.dataframe(opt_results, use_container_width=True, hide_index=True)
        best_row = opt_results.iloc[0]
        st.info(f"🏆 Top performing combo on this historical window: **{best_row['Strategy']}** with **{best_row['Risk Style']}** ({best_row['Return %']}% return).")


# ============================================================================
# TAB 6: Symbol Explorer
# ============================================================================
with tab_explorer:
    st.markdown("### Binance Symbol Directory Explorer")
    st.caption("Query all active USDT trading pairs from `/exchangeInfo`.")

    with st.spinner("Fetching active USDT trading pairs from Binance..."):
        all_syms = fetch_all_usdt_symbols()

    search_query = st.text_input("Filter symbols:", placeholder="e.g. BTC, SOL, MATIC, AVAX", key="explorer_search")
    filtered = [s for s in all_syms if search_query.upper() in s] if search_query else all_syms

    st.caption(f"Showing {len(filtered)} of {len(all_syms)} active USDT pairs.")

    if filtered:
        chosen_sym = st.selectbox("Select symbol to view details:", filtered, key="explorer_pick")
        if st.button("Inspect Market Snapshot", type="primary", key="explorer_load"):
            try:
                with st.spinner(f"Loading {chosen_sym}..."):
                    snap_ex = fetch_ticker_snapshot(chosen_sym)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Price", f"${snap_ex['last_price']:,.4f}", f"{snap_ex['price_change_pct']:+.2f}%")
                e2.metric("24h High", f"${snap_ex['high_24h']:,.4f}")
                e3.metric("24h Low", f"${snap_ex['low_24h']:,.4f}")
                e4.metric("24h Volume (USDT)", f"${snap_ex['quote_volume_24h']:,.0f}")
            except Exception as e:
                st.warning(f"Could not load snapshot for {chosen_sym}: {e}")