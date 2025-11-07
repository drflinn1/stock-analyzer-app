#!/usr/bin/env python3
"""
tools/force_sell.py
- One-off or batch ("ALL") liquidations on Kraken Spot.
- Tries MARKET first; if price-protection blocks it, retries as LIMIT
  'slip' % through the book to force a fill.
- Writes an entry to .state/run_summary.md every time it runs.

Inputs via env:
  KRAKEN_KEY, KRAKEN_SECRET
  INPUT_SYMBOL   -> "SOON", "SOONUSD", "SOON/USD", or "ALL"
  INPUT_SLIP     -> e.g. "3.0"  (percent depth for LIMIT fallback)

Safe assumptions:
- Only sells spot balances into USD (pairs like SOONUSD, SOLUSD, etc.)
- Ignores tiny dust (< $0.50 est) and assets without a USD pair.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
import urllib.request

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_MD = STATE_DIR / "run_summary.md"

API_URL = "https://api.kraken.com"

def _kraken_request(uri_path: str, data: Dict = None, key: str = "", secret: str = "") -> Dict:
    """Minimal Kraken REST client (no external deps)."""
    data = data or {}
    postdata = urlencode(data).encode()

    if uri_path.startswith("/0/private/"):
        if not key or not secret:
            raise RuntimeError("[ERROR] Missing Kraken API credentials.")
        nonce = str(int(1000 * time.time()))
        data.update({"nonce": nonce})
        postdata = urlencode(data).encode()

        sha256 = hashlib.sha256((nonce + urlencode(data)).encode()).digest()
        hmac_data = uri_path.encode() + sha256
        mac = hmac.new(base64.b64decode(secret), hmac_data, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())

        headers = {
            "API-Key": key,
            "API-Sign": sigdigest.decode(),
            "User-Agent": "force-sell/1.1",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(API_URL + uri_path, data=postdata, headers=headers)
    else:
        req = urllib.request.Request(API_URL + uri_path, headers={"User-Agent": "force-sell/1.1"})

    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode())
        except Exception:
            return {"raw": raw.decode()}

def _get_pairs_usd_map() -> Dict[str, str]:
    """Map BASE -> ALTNAME (USD pair), e.g. {'SOON': 'SOONUSD'}"""
    res = _kraken_request("/0/public/AssetPairs")
    if "result" not in res:
        return {}
    out = {}
    for _, info in res["result"].items():
        alt = info.get("altname", "")
        if alt.endswith("USD"):
            base = alt[:-3]  # strip 'USD'
            out[base] = alt
    return out

def _best_bid_usd(pair_altname: str) -> float:
    """Get best bid for a USD pair."""
    res = _kraken_request("/0/public/Ticker?pair=" + pair_altname)
    if "result" not in res:
        return 0.0
    first_key = next(iter(res["result"]))
    bid = float(res["result"][first_key]["b"][0])
    return bid

def _account_balances(key: str, secret: str) -> Dict[str, float]:
    res = _kraken_request("/0/private/Balance", {}, key, secret)
    if res.get("error"):
        raise RuntimeError(f"[ERROR] Balance: {res['error']}")
    raw = res.get("result", {})
    return {k.replace("Z", "").replace("X", ""): float(v) for k, v in raw.items()}

def _format_sym(user_sym: str) -> str:
    s = user_sym.strip().upper().replace(" ", "")
    s = s.replace("/", "")
    if s.endswith("USD"):
        s = s[:-3]
    return s  # base only (e.g., 'SOON')

def _append_summary(lines: List[str]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    block = [
        f"### Force Sell — {ts}",
        "",
        *lines,
        "",
    ]
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

def _sell_one(pair_altname: str, vol: float, slip_pct: float, key: str, secret: str) -> Tuple[bool, str]:
    """Try MARKET, then LIMIT at bid*(1-slip). Returns (ok, message)."""
    # 1) MARKET
    data = {
        "ordertype": "market",
        "type": "sell",
        "pair": pair_altname,
        "volume": f"{vol:.8f}",
        "oflags": "viqc",
    }
    res = _kraken_request("/0/private/AddOrder", data, key, secret)
    if not res.get("error"):
        return True, f"[SELL] MARKET {vol} {pair_altname} -> OK"

    err_list = res.get("error", [])
    err_text = ", ".join(err_list) if isinstance(err_list, list) else str(err_list)

    # 2) LIMIT fallback if price protection / similar
    bid = _best_bid_usd(pair_altname)
    # ---- precision clamp: 5 decimals to satisfy Kraken tick-size rules
    limit_px = round(bid * (1.0 - slip_pct / 100.0), 5)
    data2 = {
        "ordertype": "limit",
        "type": "sell",
        "pair": pair_altname,
        "volume": f"{vol:.8f}",
        "price": f"{limit_px:.5f}",   # format to 5 dp as well
    }
    res2 = _kraken_request("/0/private/AddOrder", data2, key, secret)
    if not res2.get("error"):
        return True, f"[SELL] LIMIT {vol} {pair_altname} @ ~{limit_px:.5f} -> OK (fallback after: {err_text})"
    err2 = res2.get("error", [])
    err2_text = ", ".join(err2) if isinstance(err2, list) else str(err2)
    return False, f"[ERROR] MARKET failed: {err_text} | LIMIT fallback failed: {err2_text}"

def main():
    key = os.getenv("KRAKEN_KEY", "")
    secret = os.getenv("KRAKEN_SECRET", "")
    sym_in = os.getenv("INPUT_SYMBOL", "").strip()
    slip_str = os.getenv("INPUT_SLIP", "3.0").strip() or "3.0"
    slip_pct = float(slip_str)

    if not sym_in:
        raise SystemExit("[ERROR] INPUT_SYMBOL is required.")

    pairs_map = _get_pairs_usd_map()  # BASE -> ALTNAME
    results: List[str] = []
    sold_any = False

    if sym_in.upper() != "ALL":
        base = _format_sym(sym_in)
        if base not in pairs_map:
            raise SystemExit(f"[ERROR] No USD pair found for {base}.")
        bals = _account_balances(key, secret)
        vol = 0.0
        for k, v in bals.items():
            if k.upper() == base or k.upper().endswith(base):
                vol = max(vol, v)
        if vol <= 0:
            raise SystemExit(f"[INFO] No balance to sell for {base}.")
        vol *= 0.999  # leave hair for fees
        pair_alt = pairs_map[base]
        print(f"[INFO] Selling: {base}/USD  (slip={slip_pct:.1f}%)")
        ok, msg = _sell_one(pair_alt, vol, slip_pct, key, secret)
        results.append(msg)
        sold_any = sold_any or ok
    else:
        print(f"[INFO] Selling: ALL  (slip={slip_pct:.1f}%)")
        bals = _account_balances(key, secret)
        skip = []
        for base, pair_alt in pairs_map.items():
            vol = 0.0
            for k, v in bals.items():
                if k.upper() == base or k.upper().endswith(base):
                    vol = max(vol, v)
            if vol <= 0:
                continue
            bid = _best_bid_usd(pair_alt)
            if bid * vol < 0.5:
                skip.append(f"{base} (dust)")
                continue
            vol *= 0.999
            ok, msg = _sell_one(pair_alt, vol, slip_pct, key, secret)
            results.append(msg)
            sold_any = sold_any or ok

        if not results:
            results.append("[INFO] Nothing to sell (no balances with USD pairs).")
        if skip:
            results.append("[INFO] Skipped tiny dust: " + ", ".join(skip))

    _append_summary([
        f"- Request: symbol='{sym_in}', slip='{slip_pct} %'",
        *[f"- {line}" for line in results],
    ])

    for line in results:
        print(line)
    if sold_any:
        print("[RESULT] One or more sell orders placed.")
    else:
        print("[RESULT] No orders placed.")

if __name__ == "__main__":
    main()
