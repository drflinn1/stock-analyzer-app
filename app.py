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
APP_URL = st.secrets.get('APP_URL', '')
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Optional sanity check for secrets that enable notifications
missing = []
for key in ['EMAIL_ADDRESS', 'EMAIL_RECEIVER', 'SLACK_WEBHOOK_URL']:
    if key not in st.secrets or not str(st.secrets.get(key, "")).strip():
        missing.append(key)
if missing:
    with st.sidebar:
        st.info(
            "Notifications are optional. Add these secrets if you want them: "
            + ", ".join(missing)
        )

# Robinhood client (optional / not used in this file; here so imports don’t crash)
try:
    from robin_stocks import robinhood as r  # noqa: F401
except Exception:
    r = None

# -------------------------
# ▶  Helper to fetch S&P 500 & Top Movers
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> List[str]:
    """Return a list of S&P 500 tickers."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df_list = pd.read_html(url, flavor='bs4', attrs={'class': 'wikitable'})
        return df_list[0]['Symbol'].tolist()
    except Exception:
        # Fallback parser
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch S&P 500 list: {e}")
            return []


def _pct_change_last_close(symbols: List[str]) -> pd.Series:
    """
    Download last 2 days of Close for many tickers and compute % change.
    Works for both stocks and crypto yfinance symbols (e.g. BTC-USD).
    """
    if not symbols:
        return pd.Series(dtype=float)

    try:
        df = yf.download(symbols, period='2d', progress=False)['Close']
        if isinstance(df, pd.Series):
            changes = df.pct_change().iloc[-1:].to_frame().T.squeeze()
        else:
            changes = df.pct_change().iloc[-1]
        return changes.dropna()
    except Exception:
        perf: Dict[str, float] = {}
        for sym in symbols:
            try:
                tmp = yf.download(sym, period='2d', progress=False)['Close']
                if len(tmp) >= 2:
                    perf[sym] = float(tmp.pct_change().iloc[-1])
            except Exception:
                continue
        return pd.Series(perf, dtype=float)


def get_top_stock_tickers(n: int) -> List[str]:
    """Return Top‑N S&P 500 symbols by last‑day % change."""
    if n <= 0:
        return []
    symbols = get_sp500_tickers()
    changes = _pct_change_last_close(symbols)
    return list(changes.sort_values(ascending=False).head(n).index)


# -------------------------
# ▶  Optional Crypto – Top movers by 24h change (CoinGecko → yfinance)
# -------------------------
def get_top_crypto_tickers(m: int) -> List[str]:
    """
    Return Top‑M crypto tickers as yfinance symbols like 'BTC-USD', ranked by 24h % change.
    Requires pycoingecko.
    """
    if m <= 0:
        return []
    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        mkts = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=100,
            page=1
        )
        top = sorted(
            mkts,
            key=lambda x: (x.get('price_change_percentage_24h') or -9e9),
            reverse=True
        )[:m]
        tickers = []
        for row in top:
            sym = (row.get('symbol') or '').upper()
            if sym:
                tickers.append(f"{sym}-USD")
        return tickers
    except Exception as e:
        st.sidebar.warning(f"Crypto ranking failed: {e}")
        return []


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
            close = df['Close'].squeeze()
            sma20, upper, lower = bollinger_bands(close)
            df['sma_20'], df['bb_upper'], df['bb_lower'] = sma20, upper, lower
            df['sma_50'] = simple_sma(close, 50)
            df['rsi'] = simple_rsi(close)
            return df.dropna()
        time.sleep(1)
    raise ValueError(f"No data fetched for {ticker}")


# -------------------------
# ▶  Signals & Thresholds
# -------------------------
@st.cache_data(show_spinner=False)
def analyze(df: pd.DataFrame, min_rows: int, rsi_ovr: float, rsi_obh: float) -> Optional[dict]:
    if not isinstance(df, pd.DataFrame) or len(df) < min_rows:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(cur['rsi'])
    sma20_cur, sma50_cur = float(cur['sma_20']), float(cur['sma_50'])
    price = float(cur['Close'])

    reasons = []
    if rsi < rsi_ovr:
        reasons.append(f'RSI below {rsi_ovr} (oversold)')
    if rsi > rsi_obh:
        reasons.append(f'RSI above {rsi_obh} (overbought)')
    if prev['sma_20'] < prev['sma_50'] <= sma20_cur:
        reasons.append('20 SMA crossed above 50 SMA (bullish)')
    if prev['sma_20'] > prev['sma_50'] >= sma20_cur:
        reasons.append('20 SMA crossed below 50 SMA (bearish)')
    if price < float(cur['bb_lower']):
        reasons.append('Price below lower BB')
    if price > float(cur['bb_upper']):
        reasons.append('Price above upper BB')

    text = "; ".join(reasons).lower()
    signal = 'HOLD'
    if any(k in text for k in ['buy', 'bullish']):
        signal = 'BUY'
    elif any(k in text for k in ['sell', 'bearish']):
        signal = 'SELL'

    return {
        'RSI': round(rsi, 2),
        '20 SMA': round(sma20_cur, 2),
        '50 SMA': round(sma50_cur, 2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }


# -------------------------
# ▶  Notifications & Logging
# -------------------------
WEBHOOK = st.secrets.get('SLACK_WEBHOOK_URL')

def notify_slack(tkr: str, summ: dict, price: float):
    try:
        link = ''
        if WEBHOOK and APP_URL:
            qs = f"?tickers={','.join(st.session_state.get('tickers', []))}&period={st.session_state.get('period', '6mo')}"
            link = f" (<{APP_URL}{qs}|View in App>)"
        text = f"*{summ['Signal']}* {tkr} @ ${price}\n{summ['Reasons']}{link}"
        if WEBHOOK:
            requests.post(WEBHOOK, json={'text': text}, timeout=10)
    except Exception:
        pass


def notify_email(tkr: str, summ: dict, price: float):
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        link = ''
        if APP_URL:
            qs = f"?tickers={','.join(st.session_state.get('tickers', []))}&period={st.session_state.get('period', '6mo')}"
            link = f"\n\nView in app: {APP_URL}{qs}"
        body = (f"Ticker: {tkr}\nSignal: {summ['Signal']} @ ${price}\n"
                f"Reasons: {summ['Reasons']}\nTime: {now}{link}")
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = f"{summ['Signal']} {tkr}"
        msg['From'] = st.secrets.get('EMAIL_ADDRESS', '')
        msg['To'] = st.secrets.get('EMAIL_RECEIVER', '')

        if msg['From'] and st.secrets.get('EMAIL_PASSWORD'):
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.login(st.secrets['EMAIL_ADDRESS'], st.secrets['EMAIL_PASSWORD'])
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
        # Autonomous selection (Step #1)
        scan_top = st.checkbox('Scan top N performers', False)
        top_n = st.slider('Top tickers to scan (stocks)', 5, 100, 50) if scan_top else 0

        # Optional crypto add‑on (Step #2) — OFF by default
        include_crypto = st.checkbox('Include crypto in scan (optional)', False)
        crypto_n = st.slider('Top coins to include', 1, 25, 5) if include_crypto and scan_top else 0

        # Thresholds
        min_rows = st.slider('Minimum data rows', 10, 100, 30)
        rsi_ovr = st.slider('RSI oversold threshold', 0, 100, 30)
        rsi_obh = st.slider('RSI overbought threshold', 0, 100, 70)

        # URL parameters (modern API)
        params = st.query_params
        qs_tickers = params.get('tickers', [])
        qs_period = params.get('period', [])

        period_options = ['1mo', '3mo', '6mo', '1y', '2y']
        stock_universe = get_sp500_tickers()

        # ---------- AUTO MODE ----------
        if scan_top:
            auto_stocks = get_top_stock_tickers(top_n) if top_n > 0 else []
            auto_crypto = get_top_crypto_tickers(crypto_n) if include_crypto and crypto_n > 0 else []
            auto_list = [t for t in (auto_stocks + auto_crypto) if t]

            if auto_list:
                st.caption("Autoselected (Top Movers):")
                st.write(", ".join(auto_list))

            default_period = (qs_period[0] if (qs_period and qs_period[0] in period_options) else '6mo')

            # In auto mode there is no widget; it's safe to set session_state directly.
            st.session_state['tickers'] = auto_list
            st.session_state['period'] = default_period

            tickers = auto_list
            period = default_period

        # ---------- MANUAL MODE ----------
        else:
            if qs_tickers:
                default_list = qs_tickers[0].split(',')
            else:
                default_list = (stock_universe[:2] if len(stock_universe) >= 2 else stock_universe)

            default_period = (qs_period[0] if (qs_period and qs_period[0] in period_options) else '6mo')

            opts = stock_universe if stock_universe else default_list
            safe_defaults = [x for x in default_list if x in opts]
            if not safe_defaults and opts:
                safe_defaults = [opts[0]]

            # Widgets keep st.session_state in sync; DO NOT set session_state after creation.
            tickers = st.multiselect('Choose tickers', opts, default=safe_defaults, key='tickers')
            period = st.selectbox('Date range', period_options, index=period_options.index(default_period), key='period')

# -------------------------
# ▶  Main Page
# -------------------------
mode_badge = '🔴 SIM' if st.session_state.get('simulate_mode', simulate_mode) else '🟢 LIVE'
st.markdown(f"### {mode_badge} {PAGE_TITLE}")

if st.button('▶ Run Analysis', use_container_width=True):
    current_tickers = st.session_state.get('tickers', [])
    current_period = st.session_state.get('period', '6mo')

    if not current_tickers:
        st.warning('No tickers to analyze. Enable "Scan top N performers" or select tickers manually.')
        st.stop()

    results: Dict[str, dict] = {}
    for tkr in current_tickers:
        try:
            df = get_data(tkr, current_period)
            summ = analyze(df, min_rows, rsi_ovr, rsi_obh)
            if summ is None:
                st.warning(f"{tkr}: Not enough data, skipped")
                continue
            results[tkr] = summ

            # Notifications (optional)
            price_now = float(df.Close.iloc[-1])
            notify_email(tkr, summ, price_now)
            notify_slack(tkr, summ, price_now)

            # Chart
            st.markdown(f"#### 📈 {tkr} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_20, name='20 SMA'))
            fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name='50 SMA'))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_upper, name='BB Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df.bb_lower, name='BB Lower', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

            badge_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
            st.markdown(f"**{badge_map.get(summ['Signal'], '🟡')} {tkr} – {summ['Signal']}**")
            if debug_mode:
                st.json(summ)
            st.divider()
        except Exception as e:
            st.error(f"{tkr} failed: {e}")

    if results:
        res_df = pd.DataFrame(results).T
        st.download_button("⬇ Download CSV", res_df.to_csv().encode(), "stock_analysis_results.csv")
        st.dataframe(res_df)
        st.markdown("### 📊 Summary of Trade Signals")
        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        st.bar_chart(pd.Series({k: signal_map[v['Signal']] for k, v in results.items()}))

# -------------------------
# ▶  Logs & Tax Summary (persistent)
# -------------------------
if os.path.exists('trade_log.csv'):
    trades = pd.read_csv('trade_log.csv')
    st.subheader("🧾 Trade Log")
    st.dataframe(trades)
    st.download_button("⬇ Download Trade Log", trades.to_csv(index=False).encode(), "trade_log.csv")

    trades['Cum P/L'] = trades['Gain/Loss'].cumsum()
    total_pl = trades['Gain/Loss'].sum()
    st.markdown(f"## 💰 **Total Portfolio P/L: ${total_pl:.2f}**")
    tax = trades.groupby('Tax Category')['Gain/Loss'].sum().reset_index()
    st.subheader("Tax Summary")
    st.dataframe(tax)
    st.download_button("⬇ Download Tax Summary", tax.to_csv(index=False).encode(), "tax_summary.csv")
    st.markdown("### 📈 Portfolio Cumulative Profit Over Time")
    st.line_chart(trades.set_index('Date')['Cum P/L'])
