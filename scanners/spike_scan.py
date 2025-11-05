#!/usr/bin/env python3
"""
scanners/spike_scan.py
Heuristic spike list from Kraken USD pairs:
- Rank by 24h % change
- Keep those also close to their 24h high (freshness proxy)
- Filter out very low USD volume (rough proxy from ticker 'v' * last price)

Environment:
  MIN_24H_PCT (default 10)    -> minimum 24h pct change
  MIN_USD_VOL (default 5000)  -> min rough USD value traded today
  TOP_K_SPIKES (default 8)
"""
import json, os, urllib.request, urllib.parse, time
from pathlib import Path

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
OUT = STATE / "spike_candidates.csv"
API = "https://api.kraken.com"

def http_get(url: str, timeout: int = 25):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def usd_pairs():
    data = http_get(f"{API}/0/public/AssetPairs")
    out = []
    for k, v in data.get("result", {}).items():
        ws = v.get("wsname") or ""
        if "/USD" in ws and ".d" not in k.lower():
            out.append((k, ws))
    return out

def fetch_ticker(pairs):
    q = ",".join([p[0] for p in pairs])
    data = http_get(f"{API}/0/public/Ticker?pair={urllib.parse.quote(q)}")
    return data.get("result", {})

def pct(o, c):
    try:
        o = float(o); c = float(c)
        return (c - o) / o * 100.0 if o > 0 else 0.0
    except: return 0.0

def main():
    min_pct = float(os.getenv("MIN_24H_PCT", "10"))
    min_usd = float(os.getenv("MIN_USD_VOL", "5000"))
    top_k   = int(os.getenv("TOP_K_SPIKES", "8"))

    pairs = usd_pairs()
    if not pairs:
        OUT.write_text("symbol\n")
        return

    candidates = []
    for i in range(0, len(pairs), 20):
        chunk = pairs[i:i+20]
        t = fetch_ticker(chunk)
        for pkey, ws in chunk:
            row = t.get(pkey)
            if not row: continue
            last = float(row["c"][0])
            o    = float(row.get("o", 0) or 0)
            h24  = float(row["h"][1])   # 24h high
            v24  = float(row["v"][1])   # 24h volume (base)
            usd_volume = v24 * last
            change = pct(o, last)
            # Near-high freshness proxy: last price within 5% of 24h high
            near_high = (h24 > 0) and (last >= 0.95 * h24)
            if change >= min_pct and usd_volume >= min_usd and near_high:
                sym = ws.replace("/", "")  # "HONEY/USD" -> "HONEYUSD"
                candidates.append((change, usd_volume, sym))
        time.sleep(0.2)

    # Sort by 24h % first, then USD volume
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    keep = [sym for _, _, sym in candidates[:top_k]]

    with OUT.open("w") as f:
        f.write("symbol\n")
        for s in keep:
            f.write(f"{s}\n")

if __name__ == "__main__":
    main()
