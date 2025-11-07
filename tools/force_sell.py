#!/usr/bin/env python3
"""
tools/force_sell.py  (v1.3 — Smart Dust Clamp)

- One-off or batch ("ALL") liquidations on Kraken Spot.
- MARKET first; if rejected (price-protection etc.), retry LIMIT
  at bid*(1-slip), clamped to 5 decimals (Kraken tick size).
- "Dust sweeper" options to clear tiny balances while avoiding
  Kraken's minimum order-size errors.

ENV (read from workflow env):
  KRAKEN_KEY, KRAKEN_SECRET
  INPUT_SYMBOL      -> "SOON", "SOONUSD", "SOON/USD", or "ALL"
  INPUT_SLIP        -> e.g. "3.0"  (percent depth for LIMIT fallback)
  DUST_MIN_USD      -> default "0.50" (skip anything smaller unless FORCE_DUST=ON)
  FORCE_DUST        -> "ON" to sell even below DUST_MIN_USD
  STABLES           -> comma list to ignore (default: "USD,ZUSD,USDT,USDC")

Writes a block to .state/run_summary.md every run.
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

# ---------------- Kraken REST (minimal, stdlib only) ----------------
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
            "User-Agent": "force-sell/1.3",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(API_URL + uri_path, data=postdata, headers=headers)
    else:
        req = urllib.request.Request(API_URL + uri_path, headers={"User-Agent": "force-sell/1.3"})

    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode())
        except Exception:
            return {"raw": raw.decode()}

def _pairs_usd_and_ordermin() -> Dict[str, Dict[str, float]]:
    """
    Returns map: BASE -> { 'alt': 'BASEUSD', 'ordermin': float or default }
    """
    res = _kraken_request("/0/public/AssetPairs")
    out: Dict[str, Dict[str, float]] = {}
    if "result" not in res:
        return out
    for _, info in res["result"].items():
        alt = info.get("altname", "")
        if not alt or not alt.endswith("USD"):
            continue
        base = alt[:-3]
        # Kraken sometimes provides 'ordermin' (min base volume)
        try:
            ordermin = float(info.get("ordermin", "0.0001"))
        except Exception:
            ordermin = 0.0001
        out[base] = {"alt": alt, "ordermin": max(ordermin, 1e-8)}
    return out

def _best_bid_usd(pair_altname: str) -> float:
    res = _kraken_request("/0/public/Ticker?pair=" + pair_altname)
    if "result" not in res:
        return 0.0
    first = next(iter(res["result"]))
    return float(res["result"][first]["b"][0])

def _account_balances(key: str, secret: str) -> Dict[str, float]:
    res = _kraken_request("/0/private/Balance", {}, key, secret)
    if res.get("error"):
        raise RuntimeError(f"[ERROR] Balance: {res['error']}")
    raw = res.get("result", {})
    # Normalize Krakenified asset codes a bit (ZZZ/XZZ)
    return {k.replace("Z", "").replace("X", ""): float(v) for k, v in raw.items()}

# ---------------- Helpers ----------------
def _format_sym(user_sym: str) -> str:
    s = user_sym.strip().upper().replace(" ", "").replace("/", "")
    return s[:-3] if s.endswith("USD") else s  # BASE

def _append_summary(lines: List[str]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# Run Summary\n\n", encoding="utf-8")
    block = [f"### Force Sell — {ts}", "", *lines, ""]
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

def _smart_adj_volume(base: str, vol: float, ordermin: float) -> Tuple[float, str]:
    """
    Apply fee hair, clamp against Kraken 'ordermin', and ensure 8dp string won't be zero.
    Returns (adj_volume, reason_if_skipped_or_empty_string).
    """
    if vol <= 0:
        return 0.0, "no balance"
    # leave hair for fees
    v = vol * 0.999
    # enforce minimum order size if provided
    if v < ordermin:
        return 0.0, f"too small for Kraken min (ordermin={ordermin})"
    # ensure it doesn't round to 0 at 8 dp (Kraken accepts up to 8 dp on volume)
    if round(v, 8) <= 0.0:
        return 0.0, "too small for 8dp precision"
    return v, ""

def _sell_one(pair_altname: str, vol: float, slip_pct: float, key: str, secret: str) -> Tuple[bool, str]:
    """Try MARKET, then LIMIT at bid*(1-slip) — LIMIT price clamped to 5 dp."""
    # MARKET
    data = {
        "ordertype": "market",
        "type": "sell",
        "pair": pair_altname,
        "volume": f"{vol:.8f}",
        "oflags": "viqc",
    }
    res = _kraken_request("/0/private/AddOrder", data, key, secret)
    if not res.get("error"):
        return True, f"[SELL] MARKET {vol:.8f} {pair_altname} -> OK"

    err1 = res.get("error", [])
    err1_text = ", ".join(err1) if isinstance(err1, list) else str(err1)

    # LIMIT fallback
    bid = _best_bid_usd(pair_altname)
    limit_px = round(bid * (1.0 - slip_pct / 100.0), 5)  # clamp price to 5 dp
    data2 = {
        "ordertype": "limit",
        "type": "sell",
        "pair": pair_altname,
        "volume": f"{vol:.8f}",
        "price": f"{limit_px:.5f}",
    }
    res2 = _kraken_request("/0/private/AddOrder", data2, key, secret)
    if not res2.get("error"):
        return True, f"[SELL] LIMIT {vol:.8f} {pair_altname} @ ~{limit_px:.5f} -> OK (fallback after: {err1_text})"
    err2 = res2.get("error", [])
    err2_text = ", ".join(err2) if isinstance(err2, list) else str(err2)
    return False, f"[ERROR] MARKET failed: {err1_text} | LIMIT failed: {err2_text}"

# ---------------- Main ----------------
def main():
    key = os.getenv("KRAKEN_KEY", "")
    secret = os.getenv("KRAKEN_SECRET", "")
    sym_in = os.getenv("INPUT_SYMBOL", "").strip()
    slip_pct = float((os.getenv("INPUT_SLIP", "3.0") or "3.0"))
    dust_min = float((os.getenv("DUST_MIN_USD", "0.50") or "0.50"))
    force_dust = (os.getenv("FORCE_DUST", "OFF").upper() == "ON")
    stables = [s.strip().upper() for s in (os.getenv("STABLES", "USD,ZUSD,USDT,USDC")).split(",") if s.strip()]

    if not sym_in:
        raise SystemExit("[ERROR] INPUT_SYMBOL is required.")

    pairs = _pairs_usd_and_ordermin()  # BASE -> {'alt','ordermin'}
    results: List[str] = []
    sold_any = False

    def process_one(base: str, vol: float):
        nonlocal sold_any
        info = pairs.get(base)
        if not info:
            results.append(f"[INFO] No USD pair found for {base}; skipped.")
            return
        pair_alt = info["alt"]
        ordermin = float(info.get("ordermin", 0.0001))

        bid = _best_bid_usd(pair_alt)
        est = bid * vol
        if est < dust_min and not force_dust:
            results.append(f"[INFO] {base}: est ${est:.2f} < ${dust_min:.2f}; skipped (dust).")
            return

        adj_vol, why = _smart_adj_volume(base, vol, ordermin)
        if adj_vol <= 0:
            # Give a helpful reason (too small for ordermin/precision/etc.)
            results.append(f"[INFO] {base}: {why}; skipped.")
            return

        ok, msg = _sell_one(pair_alt, adj_vol, slip_pct, key, secret)
        results.append(msg)
        sold_any = sold_any or ok

    # ---------- Path A: specific symbol ----------
    if sym_in.upper() != "ALL":
        base = _format_sym(sym_in)
        if base in stables:
            raise SystemExit(f"[INFO] Skipping stable {base}.")
        bals = _account_balances(key, secret)
        vol = 0.0
        for k, v in bals.items():
            if k.upper() == base or k.upper().endswith(base):
                vol = max(vol, v)
        if vol <= 0:
            raise SystemExit(f"[INFO] No balance to sell for {base}.")
        process_one(base, vol)

    # ---------- Path B: ALL balances ----------
    else:
        bals = _account_balances(key, secret)
        # Iterate through known USD pairs only
        for base in sorted(pairs.keys()):
            if base in stables:
                continue
            # Find any matching key in balances
            vol = 0.0
            for k, v in bals.items():
                if k.upper() == base or k.upper().endswith(base):
                    vol = max(vol, v)
            if vol > 0:
                process_one(base, vol)

        if not results:
            results.append("[INFO] Nothing to sell (no non-stable balances).")

    lines = [
        f"- Request: symbol='{sym_in}', slip='{slip_pct} %'",
        f"- Dust rule: min=${dust_min:.2f}, FORCE_DUST={'ON' if force_dust else 'OFF'}",
        f"- Ignoring stables: {', '.join(stables)}",
        *[f"- {line}" for line in results],
    ]
    _append_summary(lines)

    for line in results:
        print(line)
    print("[RESULT] One or more sell orders placed." if sold_any else "[RESULT] No orders placed.")

if __name__ == "__main__":
    main()
