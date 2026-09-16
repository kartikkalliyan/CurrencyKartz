"""
PDF summary report generation -- a personal record of what the tool
recommended and why, at a point in time.
"""

from datetime import datetime
from fpdf import FPDF


def generate_pdf_report(asset_label, symbol, strategy_choice, risk_choice, slippage_choice,
                        recommendation, reason, latest, track_results, window_label):
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
