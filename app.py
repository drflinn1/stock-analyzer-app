# app.py – Streamlit Web App Version of Stock Analyzer Bot with S&P Scan & Full Ticker Selection

import os
import time
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import smtplib
import streamlit as st
import yfinance as yf
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh

VERSION = "0.8.7 (2025-08-14)"

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Trading mode comes from secrets (so you don't have to toggle secrets each time)
# Put TRADING_MODE = "simulate" or "live" in Streamlit secrets.
TRADING_MODE = st.secrets.get("TRADING_MODE", "simulate").strip().lower()

# Optional public app URL for links in Slack/email
APP_URL = st.secrets.get("APP_URL", "")

# Validate optional-but-useful secrets (email/slack)
missing = []
for key in ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_RECEIVER"]:
    if not st.secrets.get(key):
        missing.append(key)
if missing:
    st.sidebar.info(
        "Email notifications are optional. Missing secrets: "
        + ", ".join(missing)
        + " (set in Streamlit Cloud → App → Settings → Secrets)."
    )

WEBHOOK = st.secrets.get("SLACK_WEBHOOK_URL", "")

# Robinhood client is optional; if lib missing or TRADING_MODE != 'live', we stay simulated
try:
    from robin_stocks import robinhood as r
except ImportError:
    r = None  # library not installed; app will still run in simulate mode


# -------------------------
# ▶  Helper to fetch S&P 500 & Top Movers
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url, flavor="bs4", attrs={"class": "wikitable"})
        return df_list[0]["Symbol"].tolist()
    except Exception:
        # Fallback parser if bs4 not wired to read_html
        try:
            from bs4 import BeautifulSoup  # optional import here
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            return [
                row.find_all("td")[0].text.strip()
                for row in table.find_all("tr")[1:]
            ]
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
            return []


def get_top_tickers(n: int) -> List[str]:
    """No caching here so top movers are fresh."""
    symbols = get_sp500_tickers()
    if not symbols:
        return []
    try:
        df = yf.download(symbols, period="2d", progress=False)["Close"]
        # df can be Series if one symbol
        if isinstance(df, pd.Series):
            changes = df.pct_change().iloc[-1:].to_frame().T
        else:
            changes = df.pct_change().iloc[-1]
        return (
            changes.dropna().sort_values(ascending=False).head(n).index.tolist()
        )
    except Exception:
        # slower fallback if the multi-download chokes
        perf = {}
        for sym in symbols:
            try:
                tmp = yf.download(sym, period="2d", progress=False)["Close"]
                if len(tmp) >= 2:
                    perf[sym] = float(tmp.pct_change().iloc[-1])
            except Exception:
                continue
        return sorted(perf, key=lambda k: perf[k], reverse=True)[:n]


# -------------------------
# ▶  Analysis Helpers
# -------------------------
def simple_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = simple_sma(series, window)
    std = series.rolling(window).std()
    return sma, sma + num_std * std, sma - num_std * std


# -------------------------
# ▶  Data Fetch & Indicators
# -------------------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    for _ in range(retries):
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except Exception:
                df = df.iloc[:, df.columns.get_level_values(1) == ticker]
                df.columns = df.columns.get_level_values(0)
        if not df.empty:
            close = df["Close"].squeeze()
            sma20, upper, lower = bollinger_bands(close)
            df["sma_20"], df["bb_upper"], df["bb_lower"] = sma20, upper, lower
            df["sma_50"] = simple_sma(close, 50)
            df["rsi"] = simple_rsi(close)
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data fetched for {ticker}")


# -------------------------
# ▶  Signals & Thresholds
# -------------------------
@st.cache_data(show_spinner=False)
def analyze(
    df: pd.DataFrame, min_rows: int, rsi_ovr: float, rsi_obh: float
) -> Optional[Dict]:
    if not isinstance(df, pd.DataFrame) or len(df) < min_rows:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(cur["rsi"])
    sma20_cur, sma50_cur = float(cur["sma_20"]), float(cur["sma_50"])
    price = float(cur["Close"])

    reasons = []
    if rsi < rsi_ovr:
        reasons.append(f"RSI below {rsi_ovr} (oversold)")
    if rsi > rsi_obh:
        reasons.append(f"RSI above {rsi_obh} (overbought)")
    if prev["sma_20"] < prev["sma_50"] <= sma20_cur:
        reasons.append("20 SMA crossed above 50 SMA (bullish)")
    if prev["sma_20"] > prev["sma_50"] >= sma20_cur:
        reasons.append("20 SMA crossed below 50 SMA (bearish)")
    if price < float(cur["bb_lower"]):
        reasons.append("Price below lower BB")
    if price > float(cur["bb_upper"]):
        reasons.append("Price above upper BB")

    text = "; ".join(reasons).lower()
    signal = "HOLD"
    if any(k in text for k in ["buy", "bullish"]):
        signal = "BUY"
    elif any(k in text for k in ["sell", "bearish"]):
        signal = "SELL"

    return {
        "RSI": round(rsi, 2),
        "20 SMA": round(sma20_cur, 2),
        "50 SMA": round(sma50_cur, 2),
        "Signal": signal,
        "Reasons": "; ".join(reasons),
    }


# -------------------------
# ▶  Notifications
# -------------------------
def notify_slack(tkr: str, summ: Dict, price: float):
    if not WEBHOOK:
        return
    if APP_URL:
        qs = f"?tickers={','.join(st.session_state['tickers'])}&period={st.session_state['period']}"
        link = f" (<{APP_URL}{qs}|View in App>)"
    else:
        link = ""
    text = f"*{summ['Signal']}* {tkr} @ ${price}\n{summ['Reasons']}{link}"
    try:
        requests.post(WEBHOOK, json={"text": text}, timeout=5)
    except Exception:
        pass


def notify_email(tkr: str, summ: Dict, price: float):
    # Optional; requires EMAIL_* secrets
    if not (st.secrets.get("EMAIL_ADDRESS") and st.secrets.get("EMAIL_PASSWORD") and st.secrets.get("EMAIL_RECEIVER")):
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if APP_URL:
        qs = f"?tickers={','.join(st.session_state['tickers'])}&period={st.session_state['period']}"
        link = f"\n\nView in app: {APP_URL}{qs}"
    else:
        link = ""
    body = (
        f"Ticker: {tkr}\nSignal: {summ['Signal']} @ ${price}\n"
        f"Reasons: {summ['Reasons']}\nTime: {now}{link}"
    )
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = f"{summ['Signal']} {tkr}"
    msg["From"] = st.secrets["EMAIL_ADDRESS"]
    msg["To"] = st.secrets["EMAIL_RECEIVER"]
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
            s.send_message(msg)
    except Exception:
        pass


# -------------------------
# ▶  Sidebar UI & Controls
# -------------------------
with st.sidebar:
    st.markdown('## ⚙️ Settings')
    with st.expander('General', expanded=True):
        simulate_mode = st.checkbox('Simulate Trading Mode', True)
        debug_mode = st.checkbox('Show debug logs', False)
        st_autorefresh(interval=3600000, limit=None, key='hour_refresh')

    with st.expander('Analysis Options', expanded=True):
        scan_top = st.checkbox('Scan top N performers', False)
        top_n = st.slider('Top tickers to scan', 10, 100, 50) if scan_top else None
        universe = get_top_tickers(top_n) if scan_top else get_sp500_tickers()

        min_rows = st.slider('Minimum data rows', 10, 100, 30)
        rsi_ovr = st.slider('RSI oversold threshold', 0, 100, 30)
        rsi_obh = st.slider('RSI overbought threshold', 0, 100, 70)

        # --- Query params (new API) ---
        qp = st.query_params  # replaces st.experimental_get_query_params()

        # qp values may be str or list depending on version – normalize safely
        def _first(val):
            if isinstance(val, list):
                return val[0] if val else ''
            return val or ''

        qs_tickers_raw = _first(qp.get('tickers'))
        qs_period_raw  = _first(qp.get('period'))

        options = ['1mo', '3mo', '6mo', '1y', '2y']

        # defaults from query or sidebar defaults
        if qs_tickers_raw:
            default_list = [t.strip() for t in qs_tickers_raw.split(',') if t.strip()]
        else:
            default_list = universe[:2]

        default_period = qs_period_raw if qs_period_raw in options else '6mo'

        # --- Widgets (no pre-setting session_state keys to avoid warning) ---
        tickers = st.multiselect(
            'Choose tickers',
            universe,
            default=default_list
        )
        period = st.selectbox(
            'Date range',
            options,
            index=options.index(default_period)
        )

        # Keep URL & session state in sync AFTER widgets are created
        try:
            st.query_params.update({
                'tickers': ','.join(tickers) if tickers else '',
                'period': period
            })
        except Exception:
            # older Streamlit versions may not support update(); ignore silently
            pass

        st.session_state['tickers'] = tickers
        st.session_state['period'] = period


# -------------------------
# ▶  Main Page
# -------------------------
mode_badge = "🔴 SIM" if simulate_mode else "🟢 LIVE"
st.markdown(f"### {mode_badge} {PAGE_TITLE}")

if st.button("▶ Run Analysis", use_container_width=True):
    if not tickers:
        st.warning("Select at least one ticker")
        st.stop()

    # If we're not in simulate, but Robinhood isn't configured, fall back to simulate
    live_enabled = (not simulate_mode) and (TRADING_MODE == "live") and (r is not None)

    if not live_enabled and not simulate_mode and r is None:
        st.info("Robinhood library not installed or trading disabled; running in simulate mode.")

    results: Dict[str, Dict] = {}
    for tkr in tickers:
        try:
            df = get_data(tkr, period)
            summ = analyze(df, min_rows, rsi_ovr, rsi_obh)
            if summ is None:
                st.warning(f"{tkr}: Not enough data, skipped")
                continue

            results[tkr] = summ
            price = float(df.Close.iloc[-1])

            # Notifications (optional)
            notify_email(tkr, summ, price)
            notify_slack(tkr, summ, price)

            # Chart
            st.markdown(f"#### 📈 {tkr} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_20, name="20 SMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name="50 SMA"))
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.bb_upper,
                    name="BB Upper",
                    line=dict(dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.bb_lower,
                    name="BB Lower",
                    line=dict(dash="dot"),
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            badge_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            st.markdown(f"**{badge_map[summ['Signal']]} {tkr} – {summ['Signal']}**")
            st.json(summ)
            st.divider()
        except Exception as e:
            st.error(f"{tkr} failed: {e}")

    if results:
        res_df = pd.DataFrame(results).T
        st.download_button(
            "⬇ Download CSV",
            res_df.to_csv().encode(),
            "stock_analysis_results.csv",
        )
        st.dataframe(res_df)
        st.markdown("### 📊 Summary of Trade Signals")
        signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        st.bar_chart(pd.Series({k: signal_map[v["Signal"]] for k, v in results.items()}))

# -------------------------
# ▶  Logs & Tax Summary (persistent)
# -------------------------
if os.path.exists("trade_log.csv"):
    trades = pd.read_csv("trade_log.csv")
    st.subheader("🧾 Trade Log")
    st.dataframe(trades)
    st.download_button(
        "⬇ Download Trade Log", trades.to_csv(index=False).encode(), "trade_log.csv"
    )

    trades["Cum P/L"] = trades["Gain/Loss"].cumsum()
    total_pl = trades["Gain/Loss"].sum()
    st.markdown(f"## 💰 **Total Portfolio P/L: ${total_pl:.2f}**")
    tax = trades.groupby("Tax Category")["Gain/Loss"].sum().reset_index()
    st.subheader("Tax Summary")
    st.dataframe(tax)
    st.download_button(
        "⬇ Download Tax Summary",
        tax.to_csv(index=False).encode(),
        "tax_summary.csv",
    )
    st.markdown("### 📈 Portfolio Cumulative Profit Over Time")
    st.line_chart(trades.set_index("Date")["Cum P/L"])

