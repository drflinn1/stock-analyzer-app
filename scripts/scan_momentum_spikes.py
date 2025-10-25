#!/usr/bin/env python3
import os, pathlib
from datetime import datetime, timezone
import ccxt, pandas as pd, numpy as np

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

STATE = pathlib.Path('.state/spike_scan'); STATE.mkdir(parents=True, exist_ok=True)
CSV = STATE / 'momentum_candidates.csv'
REPORT = STATE / 'momentum_report.txt'

STABLES = {'USDT','USDC','DAI','TUSD','FDUSD','GUSD','USDP','EUR','USD','USDE','PYUSD','XAUT','WBTC'}

def is_stable_symbol(sym:str)->bool:
    try: b,q = sym.split('/')
    except ValueError: return False
    return b.upper() in STABLES or q.upper() in STABLES

def ema(x, span): return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()
def pct(a,b): return 0.0 if b==0 else (a-b)/b*100.0

ex = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
ex.load_markets()
symbols = [s for s in ex.symbols if '/USD' in s or '/USDT' in s]
tickers = ex.fetch_tickers(symbols)

rows, now = [], datetime.now(timezone.utc)
for sym, t in tickers.items():
    try: base, quote = sym.split('/')
    except ValueError: continue

    if EXCLUDE_STABLES and is_stable_symbol(sym): continue
    if WHITELIST and base.upper() not in WHITELIST: continue
    if BLACKLIST and base.upper() in BLACKLIST: continue

    last = t.get('last') or t.get('close')
    if last is None or last < MIN_PRICE_USD: continue

    pct24 = t.get('percentage')
    if pct24 is None:
        open_ = t.get('open')
        if not open_: continue
        pct24 = pct(last, open_)

    qv = t.get('quoteVolume')
    if qv is None:
        bv = t.get('baseVolume')
        if bv is None or quote.upper() not in {'USD','USDT'}: continue
        qv = bv * last
    if qv < MIN_BASE_VOL_USD or pct24 < MIN_24H_PCT: continue

    try:
        limit = min(LOOKBACK_HOURS + 2, 500)
        ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
        closes = np.array([c[4] for c in ohlcv], float)
        if len(closes) < EMA_WINDOW + 3: continue
        ema_arr = ema(closes, EMA_WINDOW)
        ema_slope = (ema_arr[-1] - ema_arr[-EMA_WINDOW]) / EMA_WINDOW
        above_ema = closes[-1] > ema_arr[-1]
        ret_24h = pct(closes[-1], closes[max(0, len(closes)-24)])
    except Exception:
        continue

    rows.append(dict(
        symbol=sym, base=base, quote=quote, last_price=round(float(last),10),
        pct_24h=round(float(pct24),3), quote_volume_24h=round(float(qv),2),
        ema_window=EMA_WINDOW, ema_slope=float(ema_slope), above_ema=bool(above_ema),
        ret_24h_from_hourlies=round(float(ret_24h),3), lookback_hours=LOOKBACK_HOURS,
        timeframe=TIMEFRAME, scanned_at_utc=now.strftime('%Y-%m-%d %H:%M:%S')
    ))

cols = ['symbol','base','quote','last_price','pct_24h','quote_volume_24h',
        'ema_window','ema_slope','above_ema','ret_24h_from_hourlies',
        'lookback_hours','timeframe','scanned_at_utc']
df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
if not df.empty:
    df = df[(df['above_ema']) & (df['ema_slope'] > 0)].sort_values(
        ['pct_24h','quote_volume_24h'], ascending=[False, False]
    )
    df.to_csv(CSV, index=False)
else:
    CSV.write_text('')

lines = [
    f"Momentum Spike — Scan Report  |  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
    "",
    f"Criteria: pct24h ≥ {MIN_24H_PCT}%, vol ≥ ${MIN_BASE_VOL_USD}, price ≥ ${MIN_PRICE_USD}, above rising EMA{EMA_WINDOW}",
    f"Exchange: {EXCHANGE_ID} | Symbols scanned: {len(symbols)} | Candidates: {0 if df.empty else len(df)}",
    ""
]
if df.empty:
    lines.append("No candidates passed filters this run. Try lowering thresholds.")
else:
    for i, r in enumerate(df.head(25).itertuples(index=False), 1):
        lines.append(f"{i:02d}. {r.symbol:<12} +{r.pct_24h:.2f}%  vol≈${r.quote_volume_24h:,.0f}  "
                     f"last=${r.last_price}  EMA{r.ema_window}↑{r.ema_slope:.6f}  "
                     f"24hRet≈{r.ret_24h_from_hourlies:.2f}%")
REPORT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved CSV → {CSV}")
print(f"Saved Report → {REPORT}")
