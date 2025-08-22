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

VERSION = "0.8.8 (UI polish)"

# -------------------------
# ▶  CONFIG & SECRETS
# -------------------------
PAGE_TITLE = "Stock Analyzer Bot (Live Trading + Tax Logs)"
APP_URL = st.secrets.get('APP_URL', '')
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# Minimal CSS polish: sticky top bar + compact paddings + signal badges
st.markdown(
    """
    <style>
      /* compact body */
      .block-container { padding-top: .6rem; padding-bottom: 2rem; }

      /* sticky header band */
      .sticky-top {
        position: sticky; top: 0; z-index: 1000;
        padding: .6rem .2rem; margin: -1.0rem 0 .8rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        background: var(--background-color);
      }
      .subtle { opacity: .7; font-weight: 500; }

      /* pill badges */
      .badge { padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: .85rem; }
      .buy { background: #eafff3; color: #037d50; border: 1px solid #bdf0d1; }
      .sell { background: #ffecec; color: #b00020; border: 1px solid #ffb3bd; }
      .hold { background: #fff7e6; color: #8a5a00; border: 1px solid #ffd391; }

      /* nicer table spacing */
      table.signal-table { width: 100%; border-collapse: collapse; }
      .signal-table th, .signal-table td { padding: .5rem .6rem; border-bottom: 1px solid rgba(0,0,0,.06); text-align: left; }
      .signal-table th { font-weight: 700; font-size: .9rem; }
      .signal-table td.num { text-align:right; font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Optional sanity check for secrets (notifications are optional)
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

# Robinhood client (optional / not used here; import guarded)
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
        # fallback scraping path
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        except Exception:
            return []


def _pct_change_last_close(symbols: List[str]) -> pd.Series:
    """% change of last close vs previous close for list of tickers."""
    if not symbols:
        return pd.Series(dtype=float)

    try:
        df = yf.download(symbols, period='2d', progress=False)['Close']
        if isinstance(df, pd.Series):
            # single symbol returns Series
            return df.pct_change().iloc[-1:].to_frame().T.squeeze().dropna()
        else:
            return df.pct_change().iloc[-1].dropna()
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
    if n <= 0:
        return []
    syms = get_sp500_tickers()
    changes = _pct_change_last_close(syms)
    return list(changes.sort_values(ascending=False).head(n).index)


# -------------------------
# ▶  Optional Crypto – Top movers (CoinGecko -> yfinance symbols)
# -------------------------
def get_top_crypto_tickers(m: int) -> List[str]:
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
        out = []
        for row in top:
            sym = (row.get('symbol') or '').upper()
            if sym:
                out.append(f"{sym}-USD")
        return out
    except Exception:
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
        'Price': round(price, 2),
        'RSI': round(rsi, 2),
        '20 SMA': round(sma20_cur, 2),
        '50 SMA': round(sma50_cur, 2),
        'Signal': signal,
        'Reasons': '; '.join(reasons)
    }


# -------------------------
# ▶  Notifications (optional)
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
        simulate_mode = st.checkbox('Simulate Trading Mode', True, key='simulate_mode')
        debug_mode = st.checkbox('Show debug logs', False, key='debug_mode')
        st_autorefresh(interval=3600000, limit=None, key='hour_refresh')

    with st.expander('Analysis Options', expanded=True):
        # Autonomous selection
        scan_top = st.checkbox('Scan top N performers', False)
        top_n = st.slider('Top tickers to scan (stocks)', 5, 100, 50) if scan_top else 0

        # Optional crypto add-on
        include_crypto = st.checkbox('Include crypto in scan (optional)', False)
        crypto_n = st.slider('Top coins to include', 1, 25, 5) if include_crypto and scan_top else 0

        # Thresholds
        min_rows = st.slider('Minimum data rows', 10, 100, 30)
        rsi_ovr = st.slider('RSI oversold threshold', 0, 100, 30)
        rsi_obh = st.slider('RSI overbought threshold', 0, 100, 70)

        # URL params
        params = st.query_params
        qs_tickers = params.get('tickers', [])
        qs_period = params.get('period', [])
        period_options = ['1mo', '3mo', '6mo', '1y', '2y']

        stock_universe = get_sp500_tickers()

        if scan_top:
            auto_stocks = get_top_stock_tickers(top_n) if top_n > 0 else []
            auto_crypto = get_top_crypto_tickers(crypto_n) if include_crypto and crypto_n > 0 else []
            auto_list = [t for t in (auto_stocks + auto_crypto) if t]

            default_period = (qs_period[0] if (qs_period and qs_period[0] in period_options) else '6mo')

            st.session_state['tickers'] = auto_list
            st.session_state['period'] = default_period

            # Display current auto-selected universe
            if auto_list:
                st.caption("Autoselected (Top Movers):")
                st.write(", ".join(auto_list))

            tickers = auto_list
            period = default_period
        else:
            # Manual mode widgets (keys populate session state automatically)
            default_list = (qs_tickers[0].split(',') if qs_tickers else (stock_universe[:2] if len(stock_universe) >= 2 else stock_universe))
            default_period = (qs_period[0] if (qs_period and qs_period[0] in period_options) else '6mo')

            opts = stock_universe if stock_universe else default_list
            safe_defaults = [x for x in default_list if x in opts] or (opts[:1] if opts else [])
            tickers = st.multiselect('Choose tickers', opts, default=safe_defaults, key='tickers')
            period = st.selectbox('Date range', period_options, index=period_options.index(default_period), key='period')

# -------------------------
# ▶  Sticky Header + Title
# -------------------------
mode_badge = '🔴 SIM' if st.session_state.get('simulate_mode', True) else '🟢 LIVE'
st.markdown(f"""
<div class="sticky-top">
  <div style="display:flex; gap:.75rem; align-items:center; justify-content:space-between;">
    <div style="font-size:1.05rem; font-weight:700;">
      {mode_badge} {PAGE_TITLE} <span class="subtle">v{VERSION}</span>
    </div>
    <div class="subtle">Tickers: {', '.join(st.session_state.get('tickers', [])) or '—'} · Period: {st.session_state.get('period', '6mo')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# ▶  Main Action
# -------------------------
run_clicked = st.button('▶ Run Analysis', use_container_width=True)

if run_clicked:
    current_tickers = st.session_state.get('tickers', [])
    current_period = st.session_state.get('period', '6mo')

    if not current_tickers:
        st.warning('No tickers to analyze. Enable "Scan top N performers" or select tickers manually.')
        st.stop()

    results: Dict[str, dict] = {}
    price_cache: Dict[str, float] = {}

    for tkr in current_tickers:
        try:
            df = get_data(tkr, current_period)
            summ = analyze(df, min_rows, rsi_ovr, rsi_obh)
            if summ is None:
                st.warning(f"{tkr}: Not enough data, skipped")
                continue

            results[tkr] = summ
            price_now = float(df.Close.iloc[-1])
            price_cache[tkr] = price_now

            # Notifications (optional)
            notify_email(tkr, summ, price_now)
            notify_slack(tkr, summ, price_now)

        except Exception as e:
            st.error(f"{tkr} failed: {e}")

    # ----- Summary at top: badges table -----
    if results:
        rows = []
        for tkr, s in results.items():
            sig = s['Signal']
            badge_class = 'buy' if sig == 'BUY' else ('sell' if sig == 'SELL' else 'hold')
            badge_html = f'<span class="badge {badge_class}">{sig}</span>'
            rows.append({
                'Ticker': tkr,
                'Signal': badge_html,
                'Price': f"${s['Price']:.2f}",
                'RSI': f"{s['RSI']:.2f}",
                '20 SMA': f"{s['20 SMA']:.2f}",
                '50 SMA': f"{s['50 SMA']:.2f}",
            })
        # Render a compact HTML table so the badge HTML shows
        table_html = "<table class='signal-table'><thead><tr>" + \
                     "".join([f"<th>{h}</th>" for h in ['Ticker', 'Signal', 'Price', 'RSI', '20 SMA', '50 SMA']]) + \
                     "</tr></thead><tbody>"
        for r in rows:
            table_html += f"<tr><td>{r['Ticker']}</td><td>{r['Signal']}</td>" \
                          f"<td class='num'>{r['Price']}</td><td class='num'>{r['RSI']}</td>" \
                          f"<td class='num'>{r['20 SMA']}</td><td class='num'>{r['50 SMA']}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("---")

        # ----- Per‑ticker charts (compact, same as before) -----
        for tkr in current_tickers:
            try:
                df = get_data(tkr, current_period)
                st.markdown(f"#### 📈 {tkr} Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df.sma_20, name='20 SMA'))
                fig.add_trace(go.Scatter(x=df.index, y=df.sma_50, name='50 SMA'))
                fig.add_trace(go.Scatter(x=df.index, y=df.bb_upper, name='BB Upper', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df.bb_lower, name='BB Lower', line=dict(dash='dot')))
                st.plotly_chart(fig, use_container_width=True)

                s = results[tkr]
                if st.session_state.get('debug_mode', False):
                    st.json(s)
                st.divider()
            except Exception as e:
                st.error(f"{tkr} chart failed: {e}")

        # CSV + quick bar of signals
        res_df = pd.DataFrame(results).T[['Price','RSI','20 SMA','50 SMA','Signal','Reasons']]
        st.download_button("⬇ Download CSV", res_df.to_csv().encode(), "stock_analysis_results.csv")
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
