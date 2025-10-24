#!/usr/bin/env python3
import os
import math
import time
import pathlib
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

# ---------- Config from env (with sane defaults) ---------- #
EXCHANGE_ID = os.getenv('EXCHANGE', 'kraken').lower()
MIN_24H_PCT = float(os.getenv('MIN_24H_PCT', '30'))
MIN_BASE_VOL_USD = float(os.getenv('MIN_BASE_VOL_USD', '25000'))
MIN_PRICE_USD = float(os.getenv('MIN_PRICE_USD', '0.01'))
EXCLUDE_STABLES = os.getenv('EXCLUDE_STABLES', 'true').lower() == 'true'
WHITELIST = {s.strip().upper() for s in (os.getenv('WHITELIST') or '').split(',') if s.strip()}
BLACKLIST = {s.strip().upper() for s in (os.getenv('BLACKLIST') or '').split(',') if s.strip()}
TIMEFRAME = os.getenv('TIMEFRAME', '1h')
LOOKBACK_HOURS = int(os.getenv('LOOKBACK_HOURS', '72'))
EMA_WINDOW = int(os.getenv('EMA_WINDOW', '20'))

STATE_DIR = pathlib.Path('.state/spike_scan')
STATE_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = STATE_DIR / 'momentum_candidates.csv'
REPORT_PATH = STATE_DIR / 'momentum_report.txt'

STABLE_TICKERS = {
    'USDT','USDC','DAI','TUSD','FDUSD','GUSD','USDP','EUR','USD','USDE','PYUSD','XAUT','WBTC'
}

# ---------- Helpers ---------- #
def is_stable_symbol(symbol: str) -> bool:
    try:
        base, quote = symbol.split('/')
    except ValueError:
        return False
    return base.upper() in STABLE_TICKERS or quote.upper() in STABLE_TICKERS

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()

def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

# ---------- Load exchange ---------- #
ex_class = getattr(ccxt, EXCHANGE_ID)
ex = ex_class({'enableRateLimit': True})
ex.load_markets()

symbols = [s for s in ex.symbols if "/USD" in s or "/USDT" in s]
all_tickers = ex.fetch_tickers(symbols)

rows = []
now = datetime.now(timezone.utc)

for sym, t in all_tickers.items():
    try:
        base, quote = sym.split('/')
    except ValueError:
        continue

    if EXCLUDE_STABLES and is_stable_symbol(sym):
        continue
    if WHITELIST and base.upper() not in WHITELIST:
        continue
    if BLACKLIST and base.upper() in BLACKLIST:
        continue

    last = t.get('last') or t.get('close')
    if last is None or last < MIN_PRICE_USD:
        continue

    pct24 = t.get('percentage')
    if pct24 is None:
        open_ = t.get('open')
        if open_:
            pct24 = pct_change(last, open_)
        else:
            continue

    quote_vol = t.get('quoteVolume')
    if quote_vol is None:
        base_vol = t.get('baseVolume')
        if base_vol is not None:
            if quote.upper() in {'USD', 'USDT'}:
                quote_vol = base_vol * last
            else:
                continue
        else:
            continue

    if quote_vol < MIN_BASE_VOL_USD:
        continue
    if pct24 < MIN_24H_PCT:
        continue

    # Momentum check via EMA slope
    try:
        limit = min(LOOKBACK_HOURS + 2, 500)
        ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
        closes = np.array([c[4] for c in ohlcv], dtype=float)
        if len(closes) < EMA_WINDOW + 3:
            continue
        ema_arr = ema(closes, EMA_WINDOW)
        ema_slope = (ema_arr[-1] - ema_arr[-EMA_WINDOW]) / EMA_WINDOW
        above_ema = closes[-1] > ema_arr[-1]
        ret_24h_from_hourlies = pct_change(closes[-1], closes[max(0, len(closes)-24)])
    except Exception:
        continue

    rows.append({
        'symbol': sym,
        'base': base,
        'quote': quote,
        'last_price': round(float(last), 10),
        'pct_24h': round(float(pct24), 3),
        'quote_volume_24h': round(float(quote_vol), 2),
        'ema_window': EMA_WINDOW,
        'ema_slope': float(ema_slope),
        'above_ema': bool(above_ema),
        'ret_24h_from_hourlies': round(float(ret_24h_from_hourlies), 3),
        'lookback_hours': LOOKBACK_HOURS,
        'timeframe': TIMEFRAME,
        'scanned_at_utc': now.strftime('%Y-%m-%d %H:%M:%S')
    })

if rows:
    df = pd.DataFrame(rows)
    df = df[(df['above_ema']) & (df['ema_slope'] > 0)]
    df = df.sort_values(['pct_24h', 'quote_volume_24h'], ascending=[False, False])
else:
    df = pd.DataFrame(columns=[
        'symbol','base','quote','last_price','pct_24h','quote_volume_24h',
        'ema_window','ema_slope','above_ema','ret_24h_from_hourlies',
        'lookback_hours','timeframe','scanned_at_utc'
    ])

if not df.empty:
    df.to_csv(CSV_PATH, index=False)
else:
    CSV_PATH.write_text('')

lines = []
lines.append(f"Momentum Spike — Scan Report  |  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
lines.append("")
lines.append(f"Criteria: pct24h ≥ {MIN_24H_PCT}%, vol ≥ ${MIN_BASE_VOL_USD}, price ≥ ${MIN_PRICE_USD}, above rising EMA{EMA_WINDOW}")
lines.append(f"Exchange: {EXCHANGE_ID} | Symbols scanned: {len(symbols)} | Candidates: {0 if df.empty else len(df)}")
lines.append("")

if df.empty:
    lines.append("No candidates passed filters this run. Try lowering thresholds.")
else:
    top = df.head(25)
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"{i:02d}. {row.symbol:<12} +{row.pct_24h:.2f}%  vol≈${row.quote_volume_24h:,.0f}  last=${row.last_price}  "
            f"EMA{row.ema_window}↑{row.ema_slope:.6f}  24hRet≈{row.ret_24h_from_hourlies:.2f}%"
        )

REPORT_PATH.write_text("\n".join(lines))

print("\n".join(lines))
print(f"\nSaved CSV → {CSV_PATH}")
print(f"Saved Report → {REPORT_PATH}")
