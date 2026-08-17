"""
LIVE TRADING ADVISOR - Streamlit UI (modularized, dual asset class)

All actual logic lives in config.py, data_fetching.py, core_logic.py,
journal.py, reports.py, display_helpers.py, and indian_stocks.py.
This file is just the UI flow.

Run with:
    pip install streamlit streamlit-autorefresh plotly fpdf2 yfinance
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from config import (ASSET_OPTIONS, STRATEGY_FUNCS, RISK_STYLES,
                    POSITION_SIZE_OPTIONS, SLIPPAGE_OPTIONS, ASSET_CLASS_OPTIONS)
from indian_stocks import INDIAN_STOCK_OPTIONS, fetch_stock_data
from data_fetching import fetch_data, find_data_for_trade_target
from core_logic import (
    prep_data, reason_for_signal, get_recommendation, run_track_record,
    get_signal_history, get_consensus, get_all_recommendations,
    get_all_stock_recommendations, run_portfolio_backtest, run_optimizer
)
from journal import load_journal, add_journal_entry
from reports import generate_pdf_report
from display_helpers import make_candlestick_chart, badge

st.set_page_config(page_title="Live Trading Advisor", layout="wide")


def resolve_custom_stock_symbol(user_input):
    """NSE tickers need a .NS suffix -- add it if the user left it off."""
    symbol = user_input.strip().upper().replace(" ", "")
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    return symbol


st.title("Live Trading Advisor")
st.caption("Decision-support only. This tool never places trades -- you decide.")

with st.expander("What this strategy is actually good at (read before trusting it)"):
    st.markdown("""
    The default combo (regime-filtered MA crossover with a 10% trailing stop)
    was tested across 5 crypto assets and multiple market periods, including a
    proper train/test split and a dedicated crash-period test:

    - **Strength:** during the 2022 crash, it lost only ~10% while simply
      holding lost 65% -- a strong, real downside-protection edge.
    - **Strength:** on BTC and ETH specifically, it meaningfully beat the
      fixed-stop version, and on ETH it even beat buy-and-hold outright.
    - **Known limitation:** on more volatile assets (SOL, BNB), the 10%
      trailing stop tends to get triggered by normal volatility, exiting
      trades too early.
    - **Important:** all of the above was validated on CRYPTO only. Indian
      stock behavior is different (different volatility, trading hours,
      circuit limits) -- treat stock recommendations as unvalidated until
      tested with the same rigor. See README.

    You now have 8 strategies, 3 risk styles, position sizing, slippage
    modeling, and two asset classes (crypto via Binance, Indian stocks via
    Yahoo Finance) -- use the Track Record section to compare combos
    honestly before trusting any recommendation.
    """)

refresh_minutes = st.sidebar.selectbox("Auto-refresh every:", [5, 15, 30, 60], index=1)
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
           f"(auto-refreshing every {refresh_minutes} min)")


st.markdown("### All assets at a glance")
overview_class_label = st.radio("Asset class:", list(ASSET_CLASS_OPTIONS.keys()),
                                horizontal=True, key="overview_class")
overview_class = ASSET_CLASS_OPTIONS[overview_class_label]
st.caption("Uses the default combo (MA Crossover + Trailing stop). Customize per asset below.")

with st.spinner("Checking all assets..."):
    if overview_class == "crypto":
        overview_df = get_all_recommendations()
    else:
        overview_df = get_all_stock_recommendations()


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
           "combo, and see the combined result.")

with st.expander("Run a portfolio backtest"):
    pf_class_label = st.radio("Asset class:", list(ASSET_CLASS_OPTIONS.keys()),
                              horizontal=True, key="pf_class")
    pf_class = ASSET_CLASS_OPTIONS[pf_class_label]
    pf_options = ASSET_OPTIONS if pf_class == "crypto" else INDIAN_STOCK_OPTIONS
    pf_fetch_func = fetch_data if pf_class == "crypto" else fetch_stock_data
    pf_currency = "$" if pf_class == "crypto" else "Rs. "

    pf_default = list(pf_options.keys())[:2]
    pf_assets = st.multiselect("Choose assets to include:", list(pf_options.keys()),
                               default=pf_default, key="pf_assets")
    pcol1, pcol2, pcol3 = st.columns(3)
    pf_strategy_choice = pcol1.selectbox("Strategy:", list(STRATEGY_FUNCS.keys()), key="pf_strategy")
    pf_risk_choice = pcol2.selectbox("Risk style:", list(RISK_STYLES.keys()), index=2, key="pf_risk")
    pf_capital = pcol3.number_input("Total capital:", min_value=100.0, value=1000.0,
                                    step=100.0, key="pf_capital")

    if st.button("Run portfolio backtest", type="primary"):
        if len(pf_assets) == 0:
            st.warning("Select at least one asset.")
        else:
            pf_symbols = [pf_options[a] for a in pf_assets]
            pf_func = STRATEGY_FUNCS[pf_strategy_choice]
            pf_risk_kwargs = dict(RISK_STYLES[pf_risk_choice])

            with st.spinner("Running portfolio backtest..."):
                pf_results = run_portfolio_backtest(pf_symbols, pf_func, pf_risk_kwargs,
                                                     pf_capital, fetch_func=pf_fetch_func)

            st.dataframe(pd.DataFrame(pf_results["rows"]), use_container_width=True, hide_index=True)

            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Portfolio final value", f"{pf_currency}{pf_results['total_final']:.2f}",
                        f"{pf_results['portfolio_return_pct']:.2f}%")
            rcol2.metric("Blended buy & hold value", f"{pf_currency}{pf_results['total_buyhold']:.2f}",
                        f"{pf_results['buyhold_return_pct']:.2f}%")

            if pf_results["portfolio_return_pct"] > pf_results["buyhold_return_pct"]:
                st.success("Portfolio beat the blended buy-and-hold benchmark.")
            else:
                st.warning("Portfolio underperformed the blended buy-and-hold benchmark.")


st.markdown("---")
st.markdown("### Auto-optimizer")
st.caption("Tests every strategy x risk style combo on one asset and ranks them.")
st.warning("Important: the top-ranked combo is whichever one happened to fit THIS specific "
           "historical window best -- that can easily be overfitting rather than a real edge. "
           "Re-check any promising combo with a train/test split before trusting it.")

with st.expander("Run auto-optimizer"):
    opt_class_label = st.radio("Asset class:", list(ASSET_CLASS_OPTIONS.keys()),
                               horizontal=True, key="opt_class")
    opt_class = ASSET_CLASS_OPTIONS[opt_class_label]
    opt_options = ASSET_OPTIONS if opt_class == "crypto" else INDIAN_STOCK_OPTIONS
    opt_fetch_func = fetch_data if opt_class == "crypto" else fetch_stock_data

    opt_asset = st.selectbox("Asset to optimize:", list(opt_options.keys()), key="opt_asset")
    opt_days = st.slider("Days of history to test on:", min_value=60, max_value=1000,
                         value=200, step=10, key="opt_days")

    if st.button("Run auto-optimizer", type="primary"):
        opt_symbol = opt_options[opt_asset]
        with st.spinner("Testing strategy/risk combos..."):
            opt_results = run_optimizer(opt_symbol, opt_days, fetch_func=opt_fetch_func)

        st.dataframe(opt_results, use_container_width=True, hide_index=True)

        best_row = opt_results.iloc[0]
        st.info(f"Top result on this window: **{best_row['Strategy']}** with "
                f"**{best_row['Risk Style']}** ({best_row['Return %']}% return). "
                f"Go validate this combo properly before trusting it.")


st.markdown("---")
st.markdown("### Asset detail")

asset_class_label = st.radio("Asset class:", list(ASSET_CLASS_OPTIONS.keys()), horizontal=True)
asset_class = ASSET_CLASS_OPTIONS[asset_class_label]

if asset_class == "crypto":
    current_asset_options = ASSET_OPTIONS
    asset_label = st.selectbox("Choose an asset:",
                               list(current_asset_options.keys()) + ["Custom (type any symbol)"])
else:
    current_asset_options = INDIAN_STOCK_OPTIONS
    asset_label = st.selectbox("Choose a stock:",
                               list(current_asset_options.keys()) + ["Custom (type any NSE ticker)"])
    st.caption("Data from Yahoo Finance. 'By number of trades' mode isn't available for "
               "stocks yet -- use 'By number of days'.")

if asset_class == "crypto" and asset_label == "Custom (type any symbol)":
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
            st.error(f"'{symbol}' returned no data. Check the symbol is correct.")
            st.stop()
    except Exception as e:
        st.error(f"Couldn't find '{symbol}' on Binance. Check the spelling. (Error: {e})")
        st.stop()

elif asset_class == "stocks" and asset_label == "Custom (type any NSE ticker)":
    custom_stock_input = st.text_input(
        "Enter an NSE ticker symbol:",
        placeholder="e.g. TCS, INFY, RELIANCE, TATASTEEL...",
        help="Just the ticker is fine -- .NS will be added automatically if you leave it off."
    )
    if not custom_stock_input:
        st.info("Type a ticker above to continue (e.g. TCS, INFY, RELIANCE).")
        st.stop()

    symbol = resolve_custom_stock_symbol(custom_stock_input)
    asset_label = symbol

    try:
        _test_df = fetch_stock_data(symbol, limit=5)
        if len(_test_df) == 0:
            st.error(f"'{symbol}' returned no data. Check the ticker is correct.")
            st.stop()
    except Exception as e:
        st.error(f"Couldn't find '{symbol}' on Yahoo Finance. Check the spelling. (Error: {e})")
        st.stop()

elif asset_class == "crypto":
    symbol = ASSET_OPTIONS[asset_label]
else:
    symbol = INDIAN_STOCK_OPTIONS[asset_label]

if asset_class == "crypto" and symbol in ["SOLUSDT", "BNBUSDT"]:
    st.warning(f"Known limitation: the default trailing-stop MA crossover has tested "
               f"noticeably worse on {asset_label} historically due to its higher "
               f"volatility. Consider trying a different strategy/risk combo below.")

scol1, scol2, scol3, scol4 = st.columns(4)
strategy_choice = scol1.selectbox("Core strategy:", list(STRATEGY_FUNCS.keys()), key=f"strategy_{symbol}")
risk_choice = scol2.selectbox("Risk style:", list(RISK_STYLES.keys()), index=2, key=f"risk_{symbol}")
sizing_choice = scol3.selectbox("Position sizing:", list(POSITION_SIZE_OPTIONS.keys()), key=f"sizing_{symbol}")
slippage_choice = scol4.selectbox("Slippage assumption:", list(SLIPPAGE_OPTIONS.keys()),
                                   index=1, key=f"slippage_{symbol}")

strategy_func = STRATEGY_FUNCS[strategy_choice]
risk_kwargs = dict(RISK_STYLES[risk_choice])
risk_per_trade_pct = POSITION_SIZE_OPTIONS[sizing_choice]
risk_kwargs["slippage_pct"] = SLIPPAGE_OPTIONS[slippage_choice]

if risk_per_trade_pct is not None:
    if risk_choice == "No stop-loss":
        st.warning("Risk-based position sizing needs a stop-loss to size against -- "
                   "using 'All-in per trade' instead. Pick a stop-loss risk style above to enable sizing.")
    else:
        risk_kwargs["risk_per_trade_pct"] = risk_per_trade_pct

if asset_class == "crypto":
    mode = st.radio("Backtest window:", ["By number of days", "By number of trades"],
                    horizontal=True, key=f"mode_{symbol}")
else:
    mode = "By number of days"

if mode == "By number of days":
    lookback_days = st.slider("Number of days to analyze/backtest:", min_value=60, max_value=1000,
                              value=200, step=10, key=f"lookback_{symbol}")
    with st.spinner(f"Fetching live data for {symbol}..."):
        if asset_class == "crypto":
            df = fetch_data(symbol, limit=lookback_days)
        else:
            df = fetch_stock_data(symbol, limit=lookback_days)
        df = prep_data(df)
        recommendation, reason, latest, signal_df = get_recommendation(df, strategy_func)
    window_label = f"last {lookback_days} days"
else:
    target_trades = st.number_input("Target number of trades to test:", min_value=5, max_value=500,
                                    value=20, step=5, key=f"target_trades_{symbol}",
                                    help="Keeps pulling further back in history until this many "
                                         "BUY/SELL trades have occurred.")
    with st.spinner(f"Searching history for {target_trades} trades..."):
        signal_df, achieved_trades, days_used = find_data_for_trade_target(
            symbol, strategy_func, target_trades, prep_data
        )
        latest = signal_df.iloc[-1]
        recommendation = latest["signal"]
        reason = reason_for_signal(recommendation)
        df = signal_df

    if achieved_trades < target_trades:
        st.info(f"Only found {achieved_trades} trades in {days_used} days of available history.")
    else:
        st.caption(f"Found {achieved_trades} trades within {days_used} days of history.")
    window_label = f"{achieved_trades} trades ({days_used} days of history)"

currency_symbol = "Rs. " if asset_class == "stocks" else "$"

st.markdown(f"## {badge(recommendation, recommendation)}", unsafe_allow_html=True)
st.write(reason)
st.caption(f"Using: **{strategy_choice}** with **{risk_choice}** on **{asset_class_label}**")

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{currency_symbol}{latest['close']:.2f}")
col2.metric("MA20", f"{currency_symbol}{latest['MA_20']:.2f}")
col3.metric("MA50", f"{currency_symbol}{latest['MA_50']:.2f}")

st.markdown(f"**Regime:** {badge(latest['regime'], latest['regime'])}  |  **RSI:** {latest['RSI']:.1f}",
            unsafe_allow_html=True)
st.caption(f"Based on most recent closed candle: {latest['timestamp'].date()}")

if asset_class == "stocks":
    st.info("Reminder: this strategy's crash-protection and other findings were validated "
            "on crypto only. Treat this recommendation as unvalidated for stocks.")

if recommendation == "HOLD" and latest["regime"] == "CHOPPY":
    st.caption("Context: markets are choppy about 80% of the time historically.")

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
st.caption(f"How **{strategy_choice}** with **{risk_choice}** would have performed historically.")

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
    st.download_button("Download full trade log (CSV)", data=trade_log_df.to_csv(index=False),
                        file_name=f"{symbol}_{strategy_choice.replace(' ', '_')}_trade_log.csv",
                        mime="text/csv")

pdf_bytes = generate_pdf_report(asset_label, symbol, strategy_choice, risk_choice, slippage_choice,
                                recommendation, reason, latest, track_results, window_label,
                                currency=currency_symbol)
st.download_button("Download summary report (PDF)", data=pdf_bytes,
                    file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf")

st.markdown("### Recent signal history")
st.caption(f"The actual BUY/SELL signals **{strategy_choice}** generated for this asset recently.")
history = get_signal_history(signal_df, n=10)
if len(history) > 0:
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button("Download signal history (CSV)", data=history.to_csv(index=False),
                        file_name=f"{symbol}_{strategy_choice.replace(' ', '_')}_signals.csv",
                        mime="text/csv")
else:
    st.write("No BUY/SELL signals in the recent history window.")

st.markdown("---")
st.markdown("### Trade journal")
st.caption("Log what you actually did in real life, to track real outcomes over time.")

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
    st.download_button("Download journal (CSV)", data=asset_journal.to_csv(index=False),
                        file_name=f"{symbol}_journal.csv", mime="text/csv")
else:
    st.write("No journal entries for this asset yet.")

st.markdown("---")
st.caption("Pick a strategy, risk style, and backtest window above for each asset -- "
           "your choices are remembered per asset as you switch between them. "
           "Past performance does not guarantee future results.")