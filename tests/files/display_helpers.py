"""
Reusable display helpers: the candlestick chart builder and the
colored pill "badge" used for BUY/SELL/HOLD and TRENDING/CHOPPY labels.
"""

import plotly.graph_objects as go


def make_candlestick_chart(df, title=""):
    """Builds a real candlestick chart with MA20/MA50 overlays, dark-themed."""
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
    """Returns HTML for a small colored pill badge (BUY/SELL/HOLD, TRENDING/CHOPPY)."""
    colors = {
        "BUY": ("#0f5132", "#75dd9a"), "SELL": ("#5c1a1a", "#f28b8b"),
        "HOLD": ("#3a3a3a", "#bbbbbb"),
        "TRENDING": ("#0f5132", "#75dd9a"), "CHOPPY": ("#4a3b0f", "#e0c060"),
    }
    bg, fg = colors.get(kind, ("#3a3a3a", "#bbbbbb"))
    return (f'<span style="background-color:{bg}; color:{fg}; padding:4px 12px; '
            f'border-radius:12px; font-weight:600; font-size:0.9em;">{text}</span>')
