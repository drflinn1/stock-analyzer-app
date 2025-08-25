# main.py — headless runner with crypto + equities screeners (autopick)
import os
import math
import pandas as pd
import yfinance as yf

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# Optional broker adapters
try:
    from trader.broker_robinhood import RobinhoodBroker
except Exception:
    RobinhoodBroker = None

try:
    from trader.broker_crypto_ccxt import CCXTCryptoBroker
except Exception:
    CCXTCryptoBroker = None

# CCXT for crypto screener
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None


# ---------- helpers ----------
def download_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(how="all")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not TA_AVAILABLE:
        return df
    close = df['Close'] if 'Close' in df.columns else None
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    rsi = ta.momentum.RSIIndicator(close=close).rsi()
    bb = ta.volatility.BollingerBands(close=close)
    out = df.copy()
    out['Close'] = close
    out['RSI'] = rsi
    out['BB_high'] = bb.bollinger_hband()
    out['BB_low'] = bb.bollinger_lband()
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or 'Close' not in df.columns:
        return df
    if not TA_AVAILABLE or 'RSI' not in df.columns:
        df = df.copy(); df['Signal'] = ""; return df

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    sig = []
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        bb_l = df['BB_low'].iloc[i]
        bb_h = df['BB_high'].iloc[i]
        c = close.iloc[i]
        if pd.notna(c) and pd.notna(bb_l) and pd.notna(rsi) and float(c) < float(bb_l) and float(rsi) < 30:
            sig.append("Buy")
        elif pd.notna(c) and pd.notna(bb_h) and pd.notna(rsi) and float(c) > float(bb_h) and float(rsi) > 70:
            sig.append("Sell")
        else:
            sig.append("")
    out = df.copy(); out['Signal'] = sig; return out


def is_crypto_symbol(sym: str) -> bool:
    return '-' in sym and sym.upper().endswith('-USD')


def ccxt_to_dash(sym: str) -> str:
    # 'BTC/USD' -> 'BTC-USD'
    return sym.replace('/', '-')


# ---------- CRYPTO SCREENER ----------
def autopick_top_crypto(exchange_id: str,
                        min_quote_vol_usd: float,
                        top_n: int,
                        rank_by: str = "pct") -> list[str]:
    if ccxt is None:
        print("SCREENER: ccxt not available; skipping autopick.")
        return []
    try:
        ex_class = getattr(ccxt, exchange_id)
    except Exception:
        print(f"SCREENER: Unknown exchange id '{exchange_id}'; skipping.")
        return []

    ex = ex_class({'enableRateLimit': True})
    try:
        markets = ex.load_markets()
    except Exception as e:
        print(f"SCREENER: load_markets failed: {e}; skipping.")
        return []

    usd_syms = [s for s, m in markets.items()
                if m.get('active') and m.get('spot') and s.endswith('/USD')]
    if not usd_syms:
        print("SCREENER: no USD spot markets discovered; skipping.")
        return []

    # fetch tickers
    tickers = {}
    try:
        if hasattr(ex, "fetch_tickers"):
            tickers = ex.fetch_tickers(usd_syms)
        else:
            for s in usd_syms:
                tickers[s] = ex.fetch_ticker(s)
    except Exception as e:
        print(f"SCREENER: fetch_tickers failed: {e}; skipping.")
        return []

    STABLES = {'USDT', 'USDC', 'DAI', 'TUSD', 'FDUSD', 'PYUSD', 'GUSD', 'USD', 'EUR'}

    rows = []
    for s, t in tickers.items():
        base = s.split('/')[0]
        if base in STABLES:
            continue
        last = t.get('last')
        pct = t.get('percentage')  # 24h %
        qvol = t.get('quoteVolume')
        if qvol is None:
            bvol = t.get('baseVolume')
            qvol = (bvol * last) if (bvol is not None and last is not None) else None
        if qvol is None or qvol < float(min_quote_vol_usd):
            continue
        rows.append((s, float(pct) if pct is not None else 0.0, float(qvol)))

    if not rows:
        print("SCREENER: no markets passed filters; skipping.")
        return []

    rows.sort(key=(lambda x: x[2] if rank_by.lower() == "vol" else x[1]), reverse=True)
    picked = [ccxt_to_dash(s) for s, _, _ in rows[:max(0, int(top_n))]]

    preview = ", ".join([f"{ccxt_to_dash(s)}(+{pct:.2f}% / ${qvol:,.0f})"
                         for s, pct, qvol in rows[:max(0, int(top_n))]])
    print(f"SCREENER: top {top_n} -> {preview}" if preview else "SCREENER: none")
    return picked


# ---------- EQUITY SCREENER ----------
# Curated liquid US universe (50-ish tickers)
EQUITY_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","COST",
    "LLY","JPM","V","UNH","XOM","JNJ","WMT","MA","PG","HD",
    "ORCL","BAC","MRK","KO","PEP","CRM","CVX","ABBV","ADBE","NFLX",
    "LIN","ACN","AMD","DIS","CSCO","TMO","NEE","MCD","WFC","TXN",
    "IBM","CAT","GE","VZ","PFE","DHR","MS","QCOM","AMAT","HON",
]

def autopick_top_equities(universe: list[str],
                          min_avg_dollar_vol: float,
                          top_n: int,
                          rank_by: str = "pct") -> list[str]:
    """
    Rank a fixed liquid universe by 1-day % move or by avg $ volume (5d).
    Filters by 5-day average dollar volume (Close * Volume).
    """
    if not universe:
        return []

    try:
        data = yf.download(" ".join(universe), period="10d", interval="1d",
                           group_by="ticker", progress=False)
    except Exception as e:
        print(f"EQ-SCREENER: download failed: {e}")
        return []

    rows = []
    multi = isinstance(data.columns, pd.MultiIndex)

    def have_ticker(t: str) -> bool:
        if multi:
            return t in data.columns.get_level_values(0)
        return True  # single case

    for t in universe:
        if not have_ticker(t):
            continue
        try:
            sub = data[t] if multi else data
            sub = sub.dropna()
            if len(sub) < 2:
                continue
            last = float(sub["Close"].iloc[-1])
            prev = float(sub["Close"].iloc[-2])
            pct = 0.0 if prev == 0 else (last - prev) / prev * 100.0
            avg_vol = float(pd.to_numeric(sub["Volume"], errors="coerce").tail(5).mean())
            avg_dv = avg_vol * last
            if math.isnan(avg_dv) or avg_dv < float(min_avg_dollar_vol):
                continue
            rows.append((t, pct, avg_dv))
        except Exception:
            continue

    if not rows:
        print("EQ-SCREENER: no equities passed filters; skipping.")
        return []

    rows.sort(key=(lambda x: x[2] if rank_by.lower() == "vol" else x[1]), reverse=True)
    picked = [t for t, _, _ in rows[:max(0, int(top_n))]]
    preview = ", ".join([f"{t}(+{pct:.2f}% / ${dv:,.0f})" for t, pct, dv in rows[:max(0, int(top_n))]])
    print(f"EQ-SCREENER: top {top_n} -> {preview}" if preview else "EQ-SCREENER: none")
    return picked


# ---------- main ----------
def main():
    # Core run knobs
    symbols_input = [s.strip() for s in os.getenv('SYMBOLS', 'AAPL,MSFT,BTC-USD').split(',') if s.strip()]
    start = os.getenv('START', '2023-01-01')
    end = os.getenv('END', '') or None
    dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
    outdir = os.getenv('OUT_DIR', 'out'); os.makedirs(outdir, exist_ok=True)

    # Brokers enabled?
    equity_enabled = os.getenv('EQUITY_BROKER', 'robinhood') == 'robinhood' and RobinhoodBroker is not None
    crypto_exchange_id = os.getenv('CRYPTO_EXCHANGE', '').strip()
    crypto_enabled = crypto_exchange_id != '' and CCXTCryptoBroker is not None

    # Optional: force side
    force_side = os.getenv('FORCE_SIDE', '').strip().lower()
    if force_side not in ('buy', 'sell'):
        force_side = None

    # Autopick (crypto)
    crypto_autopick = os.getenv('CRYPTO_AUTOPICK', 'false').lower() == 'true'
    crypto_pick_n = int(os.getenv('CRYPTO_AUTOPICK_TOP_N', '3'))
    crypto_min_vol = float(os.getenv('CRYPTO_AUTOPICK_MIN_VOL_USD', '500000'))
    crypto_rank_by = os.getenv('CRYPTO_AUTOPICK_RANK', 'pct').lower()

    # Autopick (equities)
    equity_autopick = os.getenv('EQUITY_AUTOPICK', 'false').lower() == 'true'
    equity_pick_n = int(os.getenv('EQUITY_AUTOPICK_TOP_N', '3'))
    equity_min_dv = float(os.getenv('EQUITY_AUTOPICK_MIN_DOLLAR_VOL', '100000000'))  # $100M
    equity_rank_by = os.getenv('EQUITY_AUTOPICK_RANK', 'pct').lower()

    # Instantiate brokers
    rb = None
    if equity_enabled:
        ts = os.getenv('RH_TOTP_SECRET', '').strip() or None
        rb = RobinhoodBroker(
            username=os.getenv('RH_USERNAME', ''),
            password=os.getenv('RH_PASSWORD', ''),
            totp_secret=ts,
            dry_run=dry_run,
        )
        if not dry_run:
            if ts is None:
                print("Robinhood 2FA: No TOTP secret set – will wait for app/SMS approval if challenged…")
            rb.login()

    cb = None
    if crypto_enabled:
        cb = CCXTCryptoBroker(
            exchange_id=crypto_exchange_id,
            api_key=os.getenv('CRYPTO_API_KEY', None),
            api_secret=os.getenv('CRYPTO_API_SECRET', None),
            api_password=os.getenv('CRYPTO_API_PASSPHRASE', None),
            dry_run=dry_run,
        )

    # Notional sizes
    eq_notional = float(os.getenv('EQUITY_DOLLARS_PER_TRADE', '200'))
    cr_notional = float(os.getenv('CRYPTO_DOLLARS_PER_TRADE', '100'))

    # Autopicks
    crypto_picks = autopick_top_crypto(crypto_exchange_id, crypto_min_vol, crypto_pick_n, crypto_rank_by) \
        if (crypto_enabled and crypto_autopick) else []
    equity_picks = autopick_top_equities(EQUITY_UNIVERSE, equity_min_dv, equity_pick_n, equity_rank_by) \
        if equity_autopick else []

    # Final symbols: user list + autopicks (unique, keep order)
    symbols = list(dict.fromkeys(symbols_input + equity_picks + crypto_picks))

    print(
        "CONFIG:",
        f"dry_run={dry_run}",
        f"equity_enabled={bool(rb)}",
        f"crypto_enabled={bool(cb)}",
        f"exchange={crypto_exchange_id or '(none)'}",
        f"force_side={force_side or '(none)'}",
        f"autopick_crypto={crypto_autopick} top_n={crypto_pick_n} min_vol=${crypto_min_vol:,.0f} rank_by={crypto_rank_by}",
        f"autopick_equity={equity_autopick} top_n={equity_pick_n} min_$vol=${equity_min_dv:,.0f} rank_by={equity_rank_by}",
    )

    combined = []
    for sym in symbols:
        df = download_data(sym, start, end)
        df = add_indicators(df)
        df = generate_signals(df)
        df.to_csv(os.path.join(outdir, f"{sym}_analysis.csv"))

        t = df[df.get('Signal', "") != ""].copy()
        t = t.reset_index()
        if 'Date' not in t.columns:
            t.rename(columns={t.columns[0]: 'Date'}, inplace=True)
        t.insert(1, 'Ticker', sym)
        keep_cols = [c for c in ['Date', 'Ticker', 'Close', 'RSI', 'BB_high', 'BB_low', 'Signal'] if c in t.columns]
        t = t[keep_cols]
        t.to_csv(os.path.join(outdir, f"{sym}_trade_log.csv"), index=False)
        combined.append(t)

        # Decide side
        signal = ''
        last = t.tail(1)
        if not last.empty:
            signal = last['Signal'].iloc[0]
        actual_side = force_side or (signal.lower() if signal else '')

        is_cr = is_crypto_symbol(sym)
        print("ROUTE:",
              sym,
              f"is_crypto={is_cr}",
              f"signal={signal or '(none)'}",
              f"actual_side={actual_side or '(none)'}",
              f"eq_notional={eq_notional}",
              f"cr_notional={cr_notional}",
              f"dry_run={dry_run}")

        if not actual_side:
            print(f"SKIP {sym} -> no signal and no FORCE_SIDE.")
            continue

        if is_cr and cb is not None:
            res = cb.place_market_notional(symbol=sym, side=actual_side, notional_usd=cr_notional)
            print(f"CRYPTO {sym} {actual_side} -> {res}")
        elif (not is_cr) and rb is not None:
            res = rb.place_market(symbol=sym, side=actual_side, notional=eq_notional)
            print(f"EQUITY {sym} {actual_side} -> {res}")
        else:
            print(f"No broker configured for {sym}; would {actual_side} "
                  f"${cr_notional if is_cr else eq_notional} (dry_run={dry_run})")

    if any([not t.empty for t in combined]):
        all_trades = pd.concat([t for t in combined if not t.empty], ignore_index=True)
        all_trades.to_csv(os.path.join(outdir, 'combined_trade_log.csv'), index=False)
    print(f"Dry run = {dry_run}. Orders{' NOT' if dry_run else ''} placed. Files saved to {outdir}/")


if __name__ == '__main__':
    main()
