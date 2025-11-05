#!/usr/bin/env python3
"""
scanners/momentum_scan.py
Builds .state/momentum_candidates.csv from Kraken USD pairs by 24h % change.
- No external deps (stdlib only).
- Uses /0/public/AssetPairs to discover USD pairs
- Uses /0/public/Ticker to pull last price and open price
- Sorts by 24h % change desc, writes top UNIVERSE_TOP_K symbols
"""
import json, os, urllib.request, urllib.parse, time
from pathlib import Path

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
OUT = STATE / "momentum_candidates.csv"
API = "https://api.kraken.com"

def http_get(url: str, timeout: int = 25):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def usd_pairs():
    data = http_get(f"{API}/0/public/AssetPairs")
    pairs = []
    for k, v in data.get("result", {}).items():
        ws = v.get("wsname") or ""
        # prefer visible USD spot markets like "HONEY/USD"
        if "/USD" in ws and ".d" not in k.lower():
            pairs.append((k, ws))
    return pairs

def fetch_ticker(pairs):
    # Kraken Ticker accepts comma-separated pairs
    q = ",".join([p[0] for p in pairs])
    data = http_get(f"{API}/0/public/Ticker?pair={urllib.parse.quote(q)}")
    return data.get("result", {})

def pct(o, c):
    try:
        o = float(o); c = float(c)
        return (c - o) / o * 100.0 if o > 0 else 0.0
    except: return 0.0

def main():
    top_k = int(os.getenv("UNIVERSE_TOP_K", "12"))
    pairs = usd_pairs()
    if not pairs:
        OUT.write_text("symbol\n")  # empty but valid
        return
    # Split into chunks of ~20 to avoid URL size issues
    symbols = []
    for i in range(0, len(pairs), 20):
        chunk = pairs[i:i+20]
        t = fetch_ticker(chunk)
        for pkey, ws in chunk:
            row = t.get(pkey)
            if not row: continue
            c = row["c"][0]           # last trade price
            o = row.get("o")          # today's opening price
            change = pct(o, c)
            # Prefer Kraken "altname" (like HONEYUSD)
            alt = pkey
            # Convert wsname "HONEY/USD" -> "HONEYUSD"
            sym = ws.replace("/", "")
            symbols.append((change, sym))
        time.sleep(0.2)  # polite pacing

    symbols.sort(key=lambda x: x[0], reverse=True)
    keep = [sym for _, sym in symbols[:top_k]]

    with OUT.open("w") as f:
        f.write("symbol\n")
        for s in keep:
            f.write(f"{s}\n")

if __name__ == "__main__":
    main()
