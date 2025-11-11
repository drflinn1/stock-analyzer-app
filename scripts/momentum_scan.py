#!/usr/bin/env python3
"""
Momentum Scan (USD-only) for Kraken spot pairs.

Outputs: .state/momentum_candidates.csv
Schema:  pair,score,pct24,usd_vol,ema_slope

Score = pct24 * log10(usd_vol + 1)
- pct24 ≈ (last - open_today)/open_today * 100
- usd_vol ≈ last_price * base_volume_24h   (approx from ticker)
- ema_slope: slope of EMA(20) over last ~40 closes from OHLC

ENV (optional via repo Variables):
  TOP_N (default 40)
  MIN_USD_VOL (default 25000)
  EXCLUDE (comma list of base assets to skip, default 'USDT,USDC,DAI,FDUSD,PYUSD,TUSD,USD')
  OHLC_INTERVAL (minutes, default 5)
"""

from __future__ import annotations
import os, math, csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
import requests

API = "https://api.kraken.com/0/public"
STATE = Path(".state")
STATE.mkdir(exist_ok=True)
OUT = STATE / "momentum_candidates.csv"

TOP_N = int(os.getenv("TOP_N", "40"))
MIN_USD_VOL = float(os.getenv("MIN_USD_VOL", "25000"))
EXCLUDE = set(x.strip().upper() for x in os.getenv("EXCLUDE", "USDT,USDC,DAI,FDUSD,PYUSD,TUSD,USD").split(",") if x.strip())
OHLC_INTERVAL = int(os.getenv("OHLC_INTERVAL", "5"))

def get_asset_pairs() -> List[Dict[str, Any]]:
    r = requests.get(f"{API}/AssetPairs", timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for k, v in (data.get("result") or {}).items():
        ws = v.get("wsname") or ""   # "SOL/USD"
        alt = v.get("altname") or "" # "SOLUSD"
        if not ws or "/USD" not in ws:
            continue
        base = ws.split("/")[0].replace("X","").replace("Z","").upper()
        if base in EXCLUDE:          # skip stablecoins, etc.
            continue
        out.append({"code": k, "wsname": ws, "altname": alt})
    return out

def get_ticker(pairs_alt: List[str]) -> Dict[str, Dict[str, Any]]:
    if not pairs_alt:
        return {}
    joined = ",".join(pairs_alt)
    r = requests.get(f"{API}/Ticker", params={"pair": joined}, timeout=25)
    r.raise_for_status()
    return r.json().get("result") or {}

def pct_change_24(last: float, open_today: float) -> float:
    if open_today and open_today > 0:
        return (last - open_today) / open_today * 100.0
    return 0.0

def ema(values: List[float], period: int = 20) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for x in values[1:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def get_ohlc_closes(altname: str, interval: int) -> List[float]:
    try:
        r = requests.get(f"{API}/OHLC", params={"pair": altname, "interval": interval}, timeout=30)
        r.raise_for_status()
        res = r.json().get("result") or {}
        for key, rows in res.items():
            if key == "last":
                continue
            return [float(x[4]) for x in rows[-60:]]  # last ~60 bars
    except Exception:
        pass
    return []

def main():
    pairs = get_asset_pairs()
    if not pairs:
        OUT.write_text("pair,score,pct24,usd_vol,ema_slope\n")
        print("No USD pairs after filter; wrote empty CSV.")
        return

    alt_list = [p["altname"] for p in pairs if p.get("altname")]
    tdata = get_ticker(alt_list)

    rows: List[Tuple[str, float, float, float, float]] = []
    for p in pairs:
        alt = p["altname"]
        ws = p["wsname"]        # "SOL/USD"
        t = tdata.get(alt)
        if not t:
            continue
        try:
            last = float(t["c"][0])
            open_today = float(t["o"])
            base_vol_24h = float(t["v"][1])  # 24h base volume
            usd_vol = last * base_vol_24h
            pct24 = pct_change_24(last, open_today)
            if usd_vol < MIN_USD_VOL:
                continue

            closes = get_ohlc_closes(alt, OHLC_INTERVAL)
            slope = 0.0
            if len(closes) >= 25:
                ev = ema(closes, period=20)
                if len(ev) >= 5:
                    # relative slope over ~5 bars
                    denom = abs(ev[-5]) if abs(ev[-5]) > 1e-12 else 1e-12
                    slope = (ev[-1] - ev[-5]) / denom

            score = pct24 * math.log10(usd_vol + 1.0)
            rows.append((ws, score, pct24, usd_vol, slope))
        except Exception:
            continue

    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:TOP_N]

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "score", "pct24", "usd_vol", "ema_slope"])
        for ws, score, pct24, usd_vol, slope in rows:
            w.writerow([ws, f"{score:.6f}", f"{pct24:.4f}", f"{usd_vol:.2f}", f"{slope:.6f}"])

    print(f"Wrote {len(rows)} USD pairs → {OUT}")

if __name__ == "__main__":
    main()
