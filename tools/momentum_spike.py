#!/usr/bin/env python3
# Read-only Momentum Spike Scanner (Kraken via ccxt)
# - Safe: public data only, no API keys needed
# - Robust env parsing: strips inline comments and commas
# - Outputs: .state/spike_candidates.csv (ranked)

import os
import sys
import csv
import logging
from typing import List, Dict, Tuple

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("spike")

def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    raw = raw.split("#", 1)[0].replace(",", "").strip()
    try:
        return float(raw)
    except Exception:
        return float(default)

def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    raw = raw.split("#", 1)[0].replace(",", "").strip()
    try:
        return int(float(raw))
    except Exception:
        return int(default)

def parse_str_list_env(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

MIN_24H_PCT         = parse_float_env("MIN_24H_PCT", 0.0)
MIN_BASE_VOL_USD    = parse_float_env("MIN_BASE_VOL_USD", 25000)
EMA_WINDOW          = parse_int_env  ("MOMENTUM_EMA_WINDOW", 10)
MAX_CANDIDATES      = parse_int_env  ("MAX_CANDIDATES", 10)
QUOTE_WHITELIST     = parse_str_list_env("QUOTE_WHITELIST", "USD,USDT")
OHLCV_TIMEFRAME     = os.getenv("OHLCV_TIMEFRAME", "15m")
MAX_MARKETS         = parse_int_env("MAX_MARKETS", 500)

OUTPUT_DIR          = os.getenv("OUTPUT_DIR", ".state")
OUTPUT_FILE         = os.path.join(OUTPUT_DIR, "spike_candidates.csv")

try:
    import ccxt
except Exception as e:
    print("Missing dependency: ccxt. Add 'ccxt' to requirements.txt", file=sys.stderr)
    raise

def ema(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values[:]
    k = 2 / (window + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else (v * k + prev * (1 - k))
        out.append(prev)
    return out

def load_universe_kraken(exchange) -> List[str]:
    markets = exchange.load_markets()
    pairs = []
    for m in markets.values():
        if not m.get("active", True):
            continue
        base = (m.get("base") or "").upper()
        quote = (m.get("quote") or "").upper()
        if not base or not quote:
            continue
        if quote not in QUOTE_WHITELIST:
            continue
        pairs.append(f"{base}/{quote}")
        if len(pairs) >= MAX_MARKETS:
            break
    return sorted(set(pairs))

def fetch_24h(exchange, pairs: List[str]) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    tickers = exchange.fetch_tickers(pairs)
    for p, t in tickers.items():
        pct  = t.get("percentage") or 0.0
        bvol = t.get("baseVolume") or 0.0
        last = t.get("last") or t.get("close") or 0.0
        out[p] = (float(pct), float(bvol), float(last))
    return out

def fetch_ohlcv_close(exchange, pair: str, limit: int) -> List[float]:
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=OHLCV_TIMEFRAME, limit=limit)
        return [row[4] for row in ohlcv if row and len(row) >= 5]
    except Exception as e:
        log.debug("OHLCV fetch failed for %s: %s", pair, str(e))
        return []

def score_pair(exchange, pair: str, pct24: float, base_vol: float, last: float) -> Tuple[float, Dict[str, float]]:
    usd_vol = base_vol * (last or 0.0)
    if pct24 < MIN_24H_PCT:
        return -1e9, {}
    if usd_vol < MIN_BASE_VOL_USD:
        return -1e9, {}
    closes = fetch_ohlcv_close(exchange, pair, limit=max(EMA_WINDOW * 3, 60))
    if len(closes) < EMA_WINDOW + 5:
        return -1e9, {}
    e = ema(closes, EMA_WINDOW)
    slope = (e[-1] - e[-5]) / max(e[-5], 1e-8)
    score = (pct24) + (min(usd_vol, 1_000_000) / 200_000.0) + (slope * 50)
    return score, {"pct24": pct24, "usd_vol": usd_vol, "ema_slope": slope}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("MomentumSpike — min%%: %.1f | min vol: $%,.0f | EMA:%d@%s",
             MIN_24H_PCT, MIN_BASE_VOL_USD, EMA_WINDOW, OHLCV_TIMEFRAME)
    exchange = ccxt.kraken({"enableRateLimit": True, "options": {"fetchOHLCVWarning": False}})
    universe = load_universe_kraken(exchange)
    log.info("Universe: %d pairs (quotes: %s)", len(universe), ",".join(QUOTE_WHITELIST))
    tick = fetch_24h(exchange, universe)
    ranked: List[Tuple[float, str, Dict[str, float]]] = []
    for pair in universe:
        pct24, base_vol, last = tick.get(pair, (0.0, 0.0, 0.0))
        score, details = score_pair(exchange, pair, pct24, base_vol, last)
        if score > -1e8:
            ranked.append((score, pair, details))
    ranked.sort(reverse=True, key=lambda x: x[0])
    top = ranked[:MAX_CANDIDATES]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "score", "pct24", "usd_vol", "ema_slope"])
        for score, pair, d in top:
            w.writerow([pair, f"{score:.3f}", f"{d['pct24']:.2f}", f"{d['usd_vol']:.0f}", f"{d['ema_slope']:.4f}"])
    print(f"SUMMARY: MomentumSpike — found {len(top)} candidates (min %: {MIN_24H_PCT}, min vol: ${int(MIN_BASE_VOL_USD):,}, EMA:{EMA_WINDOW}@{OHLCV_TIMEFRAME}).")
    for i, (score, pair, d) in enumerate(top, start=1):
        print(f"{i:2d}. {pair:10s} ↑  {d['pct24']:6.2f}%  vol $ {int(d['usd_vol']):,}")
    print(f"ARTIFACT: wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
