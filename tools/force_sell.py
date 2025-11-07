#!/usr/bin/env python3
"""
tools/force_sell.py  (v1.2 Dust-Sweeper ready)

- One-off or batch ("ALL") liquidations on Kraken Spot.
- Tries MARKET first; if price-protection blocks it, retries as LIMIT
  'slip' % through the book to force a fill.
- Can sweep tiny "dust" positions by lowering the DUST_MIN_USD threshold
  or turning FORCE_DUST=ON to sell *everything* non-USD.

ENV (read from workflow env):
  KRAKEN_KEY, KRAKEN_SECRET
  INPUT_SYMBOL      -> "SOON", "SOONUSD", "SOON/USD", or "ALL"
  INPUT_SLIP        -> e.g. "3.0"  (percent depth for LIMIT fallback)
  DUST_MIN_USD      -> default "0.50" (skip anything smaller)
  FORCE_DUST        -> "ON" to sell even below DUST_MIN_USD
  STABLES           -> comma list of assets to ignore (default: "USD,ZUSD,USDT,USDC")

Writes an entry to .state/run_summary.md every time it runs.
"""

import base64, hashlib, hmac, json, os, time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
import urllib.request

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_MD = STATE_DIR / "run_summary.md"
API_URL = "https://api.kraken.com"

# ---------------- Kraken REST (minimal) ----------------
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
            "User-Agent": "force-sell/1.2",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        req = urllib.request.Request(API_URL + uri_path, data=postdata, headers=headers)
    else:
        req = urllib.request.Request(API_URL + uri_path, headers={"User-Agent": "force-sell/1.2"})

    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode())
        except Exception:
            return {"raw": raw.decode()}

def _get_pairs_usd_map() -> Dict[str, str]:
    """Map BASE -> ALTNAME (USD pair), e.g. {'SOON':'SOONUSD'}"""
    res = _kraken_request("/0/public/AssetPairs")
    out = {}
    for _, info in (res.get("result") or {}).items():
        alt = info.get("altname", "")
        if alt.endswith("USD"):
            out[alt[:-3]] = alt
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
    return {k.replace("Z","").replace("X",""): float(v) for k, v in raw.items()}

# ---------------- Helpers ----------------
def _format_sym(user_sym: str) -> str:
    s = user_sym.strip().upper().replace(" ", "").replace("/", "")
    return s[:-3] if s.endswith("USD") else s  # return BASE

def _append_summary(lines: List[str]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    block = [f"### Force Sell — {ts}", "", *lines, ""]
    with SUMMARY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

def _sell_one(pair_altname: str, vol: float, slip_pct: float, key: str, secret: str) -> Tuple[bool, str]:
    """Try MARKET; if rejected, retry LIMIT at bid*(1-slip)."""
    # 1) MARKET
    data = {"ordertype":"market","type":"sell","pair":pair_altname,"volume":f"{vol:.8f}","oflags":"viqc"}
    res = _kraken_request("/0/private/AddOrder", data, key, secret)
    if not res.get("error"):
        return True, f"[SELL] MARKET {vol} {pair_altname} -> OK"

    err1 = res.get("error", [])
    err1_text = ", ".join(err1) if isinstance(err1, list) else str(err1)

    # 2) LIMIT fallback — clamp to 5 dp for Kraken tick rules
    bid = _best_bid_usd(pair_altname)
    limit_px = round(bid * (1.0 - slip_pct / 100.0), 5)
    data2 = {
        "ordertype":"limit","type":"sell","pair":pair_altname,
        "volume":f"{vol:.8f}","price":f"{limit_px:.5f}"
    }
    res2 = _kraken_request("/0/private/AddOrder", data2, key, secret)
    if not res2.get("error"):
        return True, f"[SELL] LIMIT {vol} {pair_altname} @ ~{limit_px:.5f} -> OK (fallback after: {err1_text})"

    err2 = res2.get("error", [])
    err2_text = ", ".join(err2) if isinstance(err2, list) else str(err2)
    return False, f"[ERROR] MARKET failed: {err1_text} | LIMIT fallback failed: {err2_text}"

# ---------------- Main ----------------
def main():
    key = os.getenv("KRAKEN_KEY", "")
    secret = os.getenv("KRAKEN_SECRET", "")
    sym_in = os.getenv("INPUT_SYMBOL", "").strip()
    slip_pct = float((os.getenv("INPUT_SLIP", "3.0") or "3.0"))
    dust_min = float((os.getenv("DUST_MIN_USD", "0.50") or "0.50"))
    force_dust = (os.getenv("FORCE_DUST", "OFF").upper() == "ON")
    stables = [s.strip().upper() for s in (os.getenv("STABLES","USD,ZUSD,USDT,USDC")).split(",") if s.strip()]

    if not sym_in:
        raise SystemExit("[ERROR] INPUT_SYMBOL is required.")

    pairs_map = _get_pairs_usd_map()  # BASE -> ALTNAME
    results: List[str] = []
    sold_any = False

    if sym_in.upper() != "ALL":
        base = _format_sym(sym_in)
        if base in stables:
            raise SystemExit(f"[INFO] Skipping stable {base}.")
        if base not in pairs_map:
            raise SystemExit(f"[ERROR] No USD pair found for {base}.")
        bals = _account_balances(key, secret)
        vol = 0.0
        for k, v in bals.items():
            if k.upper() == base or k.upper().endswith(base):
                vol = max(vol, v)
        if vol <= 0:
            raise SystemExit(f"[INFO] No balance to sell for {base}.")
        bid = _best_bid_usd(pairs_map[base])
        est = bid * vol
        if est < dust_min and not force_dust:
            results.append(f"[INFO] {base} est ${est:.2f} < ${dust_min:.2f}; skipped (dust).")
        else:
            vol *= 0.999
            ok, msg = _sell_one(pairs_map[base], vol, slip_pct, key, secret)
            results.append(msg); sold_any |= ok
    else:
        bals = _account_balances(key, secret)
        skip = []
        for base, pair_alt in pairs_map.items():
            if base.upper() in stables:
                continue
            vol = 0.0
            for k, v in bals.items():
                if k.upper() == base or k.upper().endswith(base):
                    vol = max(vol, v)
            if vol <= 0:
                continue
            bid = _best_bid_usd(pair_alt)
            est = bid * vol
            if est < dust_min and not force_dust:
                skip.append(f"{base} (dust ${est:.2f})")
                continue
            vol *= 0.999
            ok, msg = _sell_one(pair_alt, vol, slip_pct, key, secret)
            results.append(msg); sold_any |= ok

        if not results:
            results.append("[INFO] Nothing to sell (no non-stable balances).")
        if skip:
            results.append("[INFO] Skipped as dust: " + ", ".join(skip))

    # Write summary block
    lines = [
        f"- Request: symbol='{sym_in}', slip='{slip_pct} %'",
        f"- Dust rule: min=${dust_min:.2f}, FORCE_DUST={'ON' if force_dust else 'OFF'}",
        f"- Ignoring stables: {', '.join(stables)}",
        *[f"- {line}" for line in results],
    ]
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# Run Summary\n\n", encoding="utf-8")
    _append_summary(lines)

    for line in results:
        print(line)
    print("[RESULT] One or more sell orders placed." if sold_any else "[RESULT] No orders placed.")

if __name__ == "__main__":
    main()
