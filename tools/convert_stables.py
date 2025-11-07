#!/usr/bin/env python3
"""
tools/convert_stables.py (v1.0)
- Sells stablecoins (USDT, USDC) into USD on Kraken Spot.
- MARKET first; if blocked, retry LIMIT at bid*(1-slip), price clamped to 5 dp.
- Skips if no balance or if no USD pair exists.

ENV:
  KRAKEN_KEY, KRAKEN_SECRET
  SLIP_PCT        -> "3.0"   # limit fallback depth %
  STABLES         -> "USDT,USDC" (comma list)
  MIN_USD         -> "0.50"  # don't bother if est value < MIN_USD
  WRITE_SUMMARY   -> "ON"    # append to .state/run_summary.md
"""

import base64, hashlib, hmac, json, os, time
from pathlib import Path
from typing import Dict, Tuple, List
from urllib.parse import urlencode
import urllib.request

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_MD = STATE_DIR / "run_summary.md"
API_URL = "https://api.kraken.com"

def _kraken_request(uri_path: str, data: Dict = None, key: str = "", secret: str = "") -> Dict:
    data = data or {}
    postdata = urlencode(data).encode()

    if uri_path.startswith("/0/private/"):
        if not key or not secret:
            raise RuntimeError("[ERROR] Missing Kraken API credentials.")
        nonce = str(int(1000 * time.time()))
        data.update({"nonce": nonce})
        postdata = urlencode(data).encode()

        sha256 = hashlib.sha256((nonce + urlencode(data)).encode()).digest()
        mac = hmac.new(base64.b64decode(secret), uri_path.encode() + sha256, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())

        headers = {
            "API-Key": key,
            "API-Sign": sigdigest.decode(),
            "User-Agent": "convert-stables/1.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(API_URL + uri_path, data=postdata, headers=headers)
    else:
        req = urllib.request.Request(API_URL + uri_path, headers={"User-Agent": "convert-stables/1.0"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode())
        except Exception:
            return {"raw": raw.decode()}

def _pairs_usd() -> Dict[str, str]:
    """Return map BASE -> ALTNAME for all BASEUSD pairs (e.g., {'USDT':'USDTUSD'})."""
    res = _kraken_request("/0/public/AssetPairs")
    out: Dict[str, str] = {}
    for _, info in (res.get("result") or {}).items():
        alt = info.get("altname", "")
        if alt.endswith("USD"):
            out[alt[:-3]] = alt
    return out

def _best_bid_usd(altname: str) -> float:
    res = _kraken_request("/0/public/Ticker?pair=" + altname)
    if "result" not in res:
        return 0.0
    k = next(iter(res["result"]))
    return float(res["result"][k]["b"][0])

def _balances(key: str, secret: str) -> Dict[str, float]:
    res = _kraken_request("/0/private/Balance", {}, key, secret)
    if res.get("error"):
        raise RuntimeError(f"[ERROR] Balance: {res['error']}")
    raw = res.get("result", {})
    return {k.replace("Z","").replace("X",""): float(v) for k, v in raw.items()}

def _sell(pair_alt: str, vol: float, slip_pct: float, key: str, secret: str) -> Tuple[bool, str]:
    # MARKET
    data = {"ordertype":"market","type":"sell","pair":pair_alt,"volume":f"{vol:.8f}","oflags":"viqc"}
    r1 = _kraken_request("/0/private/AddOrder", data, key, secret)
    if not r1.get("error"):
        return True, f"[STABLE] MARKET {vol:.8f} {pair_alt} -> OK"

    err = r1.get("error", [])
    err_text = ", ".join(err) if isinstance(err, list) else str(err)

    # LIMIT fallback at 5 dp
    bid = _best_bid_usd(pair_alt)
    px = round(bid * (1.0 - slip_pct / 100.0), 5)
    data2 = {"ordertype":"limit","type":"sell","pair":pair_alt,"volume":f"{vol:.8f}","price":f"{px:.5f}"}
    r2 = _kraken_request("/0/private/AddOrder", data2, key, secret)
    if not r2.get("error"):
        return True, f"[STABLE] LIMIT {vol:.8f} {pair_alt} @ ~{px:.5f} -> OK (fallback after: {err_text})"
    err2 = r2.get("error", [])
    err2_text = ", ".join(err2) if isinstance(err2, list) else str(err2)
    return False, f"[STABLE][ERROR] MARKET failed: {err_text} | LIMIT failed: {err2_text}"

def _append(lines: List[str]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# Run Summary\n\n", encoding="utf-8")
    block = [f"### Convert Stables — {ts}", "", *lines, ""]
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

def main():
    key = os.getenv("KRAKEN_KEY","")
    sec = os.getenv("KRAKEN_SECRET","")
    slip = float(os.getenv("SLIP_PCT","3.0") or "3.0")
    stables = [s.strip().upper() for s in (os.getenv("STABLES","USDT,USDC")).split(",") if s.strip()]
    min_usd = float(os.getenv("MIN_USD","0.50") or "0.50")
    write_summary = (os.getenv("WRITE_SUMMARY","ON").upper() == "ON")

    usdmap = _pairs_usd()
    bals = _balances(key, sec)
    out: List[str] = []
    any_sold = False

    for base in stables:
        vol = 0.0
        for k, v in bals.items():
            if k.upper() == base or k.upper().endswith(base):
                vol = max(vol, v)
        if vol <= 0:
            out.append(f"- {base}: no balance; skipped.")
            continue
        pair = usdmap.get(base)
        if not pair:
            out.append(f"- {base}: no USD pair; skipped.")
            continue
        bid = _best_bid_usd(pair)
        est = bid * vol
        if est < min_usd:
            out.append(f"- {base}: est ${est:.2f} < ${min_usd:.2f}; skipped.")
            continue
        # leave tiny hair for fees; ensure non-zero 8dp
        adj = max(0.0, round(vol * 0.999, 8))
        if adj <= 0:
            out.append(f"- {base}: too small after precision; skipped.")
            continue
        ok, msg = _sell(pair, adj, slip, key, sec)
        out.append(f"- {msg}")
        any_sold = any_sold or ok

    if write_summary:
        _append(out)

    for line in out:
        print(line)
    print("[RESULT] One or more converts placed." if any_sold else "[RESULT] No converts placed.")

if __name__ == "__main__":
    main()
