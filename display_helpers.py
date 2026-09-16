"""
Reusable display helpers: custom CSS injection, TradingView-style dual-panel
candlestick chart (price + volume), pill badges, and styled metric card components.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def inject_custom_css():
    """Injects custom CSS styling for a modern, dark financial terminal aesthetic."""
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* App container background */
        .stApp {
            background-color: #090d16;
            color: #f1f5f9;
        }

        /* Main Header Banner */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .hero-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 6px;
        }

        /* Custom Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0f172a;
            padding: 6px 8px;
            border-radius: 12px;
            border: 1px solid #1e293b;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 500;
            padding: 0 16px;
            border: none !important;
            transition: all 0.2s ease-in-out;
        }

        .stTabs [aria-selected="true"] {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        /* Custom Cards */
        .glass-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        /* Metric card styling */
        div[data-testid="stMetric"] {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
        }

        /* Expander styling */
        .stExpander {
            background-color: #0f172a;
            border: 1px solid #1e293b !important;
            border-radius: 12px !important;
        }

        /* Dataframe table styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1e293b;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            border: none;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        .stButton>button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)


def make_candlestick_chart(df, title=""):
    """
    Builds a TradingView-style dual-panel chart:
      - Upper panel: Candlesticks + MA20/MA50 overlays
      - Lower panel: Volume bar chart with color-coded buy/sell volume
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(title, "")
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
            increasing_line_color="#00e676", decreasing_line_color="#ff1744",
            increasing_fillcolor="#00e676", decreasing_fillcolor="#ff1744"
        ),
        row=1, col=1
    )

    # 2. Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["MA_20"], name="MA20",
            line=dict(color="#fbbf24", width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["MA_50"], name="MA50",
            line=dict(color="#38bdf8", width=1.5)
        ),
        row=1, col=1
    )

    # 3. Volume Bar Trace (green for up candles, red for down)
    vol_colors = [
        "#00e676" if close >= open_p else "#ff1744"
        for close, open_p in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["timestamp"], y=df["volume"], name="Volume",
            marker_color=vol_colors, opacity=0.6
        ),
        row=2, col=1
    )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )

    # Gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")

    return fig


def badge(text, kind):
    """Returns HTML for a sleek, glowing pill badge (BUY/SELL/HOLD, TRENDING/CHOPPY)."""
    styles = {
        "BUY":      ("rgba(0, 230, 118, 0.15)",  "#00e676", "rgba(0, 230, 118, 0.3)"),
        "SELL":     ("rgba(255, 23, 68, 0.15)",   "#ff1744", "rgba(255, 23, 68, 0.3)"),
        "HOLD":     ("rgba(148, 163, 184, 0.15)", "#94a3b8", "rgba(148, 163, 184, 0.3)"),
        "TRENDING": ("rgba(56, 189, 248, 0.15)",  "#38bdf8", "rgba(56, 189, 248, 0.3)"),
        "CHOPPY":   ("rgba(251, 191, 36, 0.15)",  "#fbbf24", "rgba(251, 191, 36, 0.3)"),
    }
    bg, fg, border = styles.get(kind, ("rgba(148, 163, 184, 0.15)", "#94a3b8", "rgba(148, 163, 184, 0.3)"))
    return (
        f'<span style="background-color:{bg}; color:{fg}; border: 1px solid {border}; '
        f'padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85em; '
        f'letter-spacing: 0.5px; box-shadow: 0 0 10px {bg};">{text}</span>'
    )


def make_equity_drawdown_chart(backtest_results, title="Backtest Performance & Underwater Drawdown"):
    """
    Builds a 2-panel Plotly chart:
      - Upper panel: Strategy Equity vs Buy & Hold Benchmark Equity
      - Lower panel: Underwater Drawdown % area chart
    """
    df = backtest_results.get("df", None)
    if df is None or "equity" not in df.columns:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.70, 0.30],
        subplot_titles=(title, "Underwater Drawdown (%)")
    )

    timestamps = df["timestamp"] if "timestamp" in df.columns else df.index

    # 1. Strategy Equity
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=df["equity"], name="Strategy Equity",
            line=dict(color="#38bdf8", width=2.5)
        ),
        row=1, col=1
    )

    # 2. Buy & Hold Benchmark Equity
    if "buy_hold_equity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=df["buy_hold_equity"], name="Buy & Hold Baseline",
                line=dict(color="#94a3b8", width=1.5, dash="dash")
            ),
            row=1, col=1
        )

    # 3. Drawdown Area Plot
    running_max = df["equity"].cummax()
    drawdown_pct = ((df["equity"] - running_max) / running_max) * 100

    fig.add_trace(
        go.Scatter(
            x=timestamps, y=drawdown_pct, name="Drawdown %",
            fill="tozeroy", line=dict(color="#ff1744", width=1.5),
            fillcolor="rgba(255, 23, 68, 0.2)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=480,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")

    return fig


def make_monte_carlo_chart(mc_results: dict, title="Monte Carlo 1,000 Path Simulations"):
    """
    Renders Monte Carlo percentile fan chart (5th, 50th, 95th percentiles).
    """
    percentiles_df = mc_results.get("percentiles", None)
    if percentiles_df is None or len(percentiles_df) == 0:
        return go.Figure()

    fig = go.Figure()
    days = list(range(len(percentiles_df)))

    # Helper to retrieve column safely
    col_95 = "95th (Optimistic)" if "95th (Optimistic)" in percentiles_df.columns else ("95th" if "95th" in percentiles_df.columns else percentiles_df.columns[0])
    col_50 = "50th (Median)" if "50th (Median)" in percentiles_df.columns else ("50th" if "50th" in percentiles_df.columns else percentiles_df.columns[0])
    col_5 = "5th (Pessimistic)" if "5th (Pessimistic)" in percentiles_df.columns else ("5th" if "5th" in percentiles_df.columns else percentiles_df.columns[0])

    # 95th Percentile (Optimistic)
    fig.add_trace(go.Scatter(
        x=days, y=percentiles_df[col_95], name="95th Percentile (Bull Case)",
        line=dict(color="#00e676", width=2)
    ))

    # 50th Percentile (Median)
    fig.add_trace(go.Scatter(
        x=days, y=percentiles_df[col_50], name="50th Percentile (Median)",
        line=dict(color="#38bdf8", width=2.5)
    ))

    # 5th Percentile (Pessimistic)
    fig.add_trace(go.Scatter(
        x=days, y=percentiles_df[col_5], name="5th Percentile (Bear Case)",
        line=dict(color="#ff1744", width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Simulated Days",
        yaxis_title="Portfolio Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")

    return fig


def make_correlation_heatmap(corr_df: pd.DataFrame, title="Cross-Asset Price Return Correlation"):
    """
    Renders an interactive correlation heatmap using Plotly.
    """
    if corr_df is None or len(corr_df) == 0:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="Viridis",
        zmin=-1.0, zmax=1.0,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11}
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


def make_comparative_equity_chart(comp_df: pd.DataFrame, title="Multi-Strategy Cumulative Equity Comparison"):
    """
    Renders multiple equity curves overlaid on a single Plotly chart.
    """
    if comp_df is None or len(comp_df) == 0:
        return go.Figure()

    fig = go.Figure()
    timestamps = comp_df["timestamp"] if "timestamp" in comp_df.columns else comp_df.index

    colors = ["#38bdf8", "#00e676", "#fbbf24", "#c084fc", "#f43f5e", "#94a3b8"]

    col_idx = 0
    for col in comp_df.columns:
        if col != "timestamp":
            is_baseline = "Baseline" in col or "Buy" in col
            line_style = dict(color="#94a3b8", width=1.5, dash="dash") if is_baseline else dict(color=colors[col_idx % len(colors)], width=2.5)
            fig.add_trace(go.Scatter(
                x=timestamps, y=comp_df[col], name=col,
                line=line_style
            ))
            if not is_baseline:
                col_idx += 1

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")

    return fig


def make_ga_fitness_chart(history_df: pd.DataFrame, title="Genetic Algorithm Strategy Fitness Evolution"):
    """Renders generation-by-generation GA fitness progression."""
    if history_df is None or len(history_df) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["best_fitness"],
        mode="lines+markers", name="Best Fitness (Sharpe)",
        line=dict(color="#00e676", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["avg_fitness"],
        mode="lines", name="Population Average",
        line=dict(color="#38bdf8", width=1.5, dash="dot")
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    return fig


def make_efficient_frontier_chart(mc_df: pd.DataFrame, max_sharpe: dict, min_vol: dict, title="Markowitz Efficient Frontier"):
    """Renders Markowitz Efficient Frontier scatter with Tangency and Min-Vol portfolios."""
    if mc_df is None or len(mc_df) == 0:
        return go.Figure()

    fig = go.Figure()

    # Scatter Cloud of Monte Carlo simulated portfolios
    fig.add_trace(go.Scatter(
        x=mc_df["Volatility %"],
        y=mc_df["Return %"],
        mode="markers",
        marker=dict(
            size=5,
            color=mc_df["Sharpe Ratio"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name="Simulated Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in mc_df["Sharpe Ratio"]],
        hoverinfo="x+y+text"
    ))

    # Max Sharpe Ratio (Tangency)
    fig.add_trace(go.Scatter(
        x=[max_sharpe["volatility_pct"]],
        y=[max_sharpe["return_pct"]],
        mode="markers+text",
        name="Max Sharpe Portfolio ⭐",
        text=["Max Sharpe ⭐"],
        textposition="top center",
        marker=dict(color="#fbbf24", size=15, symbol="star", line=dict(color="#ffffff", width=1))
    ))

    # Minimum Volatility
    fig.add_trace(go.Scatter(
        x=[min_vol["volatility_pct"]],
        y=[min_vol["return_pct"]],
        mode="markers+text",
        name="Min Volatility Portfolio 🛡️",
        text=["Min Vol 🛡️"],
        textposition="bottom center",
        marker=dict(color="#00e676", size=14, symbol="diamond", line=dict(color="#ffffff", width=1))
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Annualized Volatility (Risk %)",
        yaxis_title="Annualized Expected Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    return fig


def make_feature_importance_chart(fi_df: pd.DataFrame, title="ML Technical Indicator Feature Importance"):
    """Renders horizontal bar chart of technical indicator feature importances."""
    if fi_df is None or len(fi_df) == 0:
        return go.Figure()

    df_sorted = fi_df.sort_values("Importance %", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted["Importance %"],
        y=df_sorted["Feature"],
        orientation="h",
        marker=dict(color="#38bdf8", line=dict(color="#0284c7", width=1))
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Importance Contribution (%)",
        yaxis_title="Indicator Feature"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b")
    return fig




