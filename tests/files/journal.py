"""
Trade journal (log what you actually did) and email alert
infrastructure (currently unused in the UI, kept available for later).
"""

import os
import json
import smtplib
import pandas as pd
import streamlit as st
from email.mime.text import MIMEText

JOURNAL_FILE = "trade_journal.csv"
LAST_SIGNAL_FILE = "last_signals.json"


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
    Returns (success: bool, message: str) so the UI can show what
    happened without crashing if email isn't configured.
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
    symbol+strategy combo. If it changed to a real BUY or SELL, sends
    an email alert and updates the stored state.
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
