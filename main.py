#!/usr/bin/env python3
"""
main.py — Unified crypto rotation runner (LIVE/DRY-RUN) for Kraken.

Features:
- Loads promoted candidates from .state/momentum_candidates.csv (spike scan promotes here)
- Single-position rotation with:
    • 1% stop-loss
    • 5% take-profit
    • "<3% in 1 hour" rotate-out rule
    • Cool-down to avoid instant re-buys of the same symbol
- Always writes .state/run_summary.md and .state/run_summary.json (even on risk-off)
- LIVE mode: places Kraken market orders via REST (no external libs needed)
- DRY_RUN mode: simulates orders but updates .state/position.json so behavior is testable

Environment (commonly set via workflow inputs/vars):
- DRY_RUN: "ON"|"OFF"               (default: "ON")
- BUY_USD: amount in USD to buy     (default: "10")
- UNIVERSE_PICK: specific pair to force buy (e.g., "ALCX/USD") (default: "")
- COOLDOWN_MIN: minutes to avoid re-buying a symbol after exit (default: "60")

Kraken secrets (for LIVE):
- KRAKEN_API_KEY
- KRAKEN_API_SECRET (base64)
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import urllib.parse
import urllib.request

# ---------- Paths ----------
STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "position.json"
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD = STATE / "run_summary.md"
LAST_EXIT = STATE / "last_exit_code.txt"
RISK_FILE = STATE / "last_risk_signal.txt"      # optional; if exists and says OFF => skip
COOLDOWN_FILE = STATE / "cooldown.json"         # { "SYMBOL": "2025-11-05T22:00:00Z", ... }
CANDIDATES = STATE / "momentum_candidates.csv"  # spike scan promotes here

# ---------- Config helpers ----------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return "" if v is None else str(v)

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, ""))
    except Exception:
        return default

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()

# ---------- Kraken minimal REST client ----------
KRAKEN_BASE = "https://api.kraken.com"

def _kraken_request(path: str, data: Dict[str, Any], key: str, secret_b64: str) -> Dict[str, Any]:
    """
    POST to Kraken REST.
    """
    url = f"{KRAKEN_BASE}{path}"
    nonce = str(int(time.time() * 1000))
    post_data = {"nonce": nonce}
    post_data.update(data)
    post_bytes = urllib.parse.urlencode(post_data).encode()

    sha256 = hashlib.sha256((nonce + urllib.parse.urlencode(data)).encode()).digest()
    msg = path.encode() + sha256
    secret = base64.b64decode(secret_b64)
    sig = hmac.new(secret, msg, hashlib.sha512).digest()
    headers = {
        "API-Key": key,
        "API-Sign": base64.b64encode(sig),
        "User-Agent": "stock-analyzer-app",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    req = urllib.request.Request(url, data=post_bytes, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
        payload = json.loads(body.decode())
        return payload

def kraken_public_ticker(pair: str) -> Optional[float]:
    """
    Returns last price for pair like 'ALCXUSD' or 'ALCX/USD' (slash allowed; we normalize).
    """
    kr_pair = pair.replace("/", "").upper()
    url = f"{KRAKEN_BASE}/0/public/Ticker?pair={urllib.parse.quote(kr_pair)}"
    req = urllib.request.Request(url, headers={"User-Agent": "stock-analyzer-app"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            if data.get("error"):
                return None
            result = data.get("result", {})
            if not result:
                return None
            # result key can be normalized or mapped; pick first key
            first_key = next(iter(result.keys()))
            # 'c' field -> last trade [price, lot volume]
            last = result[first_key]["c"][0]
            return float(last)
    except Exception:
        return None

def kraken_place_market(key: str, secret_b64: str, pair: str, side: str, volume: float) -> Tuple[bool, str]:
    """
    Place a market order. side in {"buy","sell"}.
    Returns (ok, txid_or_error).
    """
    pair_norm = pair.replace("/", "").upper()
    data = {
        "pair": pair_norm,
        "type": side,
        "ordertype": "market",
        "volume": f"{volume:.8f}",
        "oflags": "viqc",  # volume in quote currency conversion if needed
    }
    try:
        resp = _kraken_request("/0/private/AddOrder", data, key, secret_b64)
        if resp.get("error"):
            return False, ";".join(resp["error"])
        txs = resp.get("result", {}).get("txid", [])
        return True, ",".join(txs) if txs else "OK"
    except Exception as e:
        return False, f"EXC:{e}"

# ---------- Files ----------
def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))

def write_text(path: Path, text: str) -> None:
    path.write_text(text)

# ---------- Cooldown ----------
def load_cooldown() -> Dict[str, str]:
    return read_json(COOLDOWN_FILE, {})

def save_cooldown(cd: Dict[str, str]) -> None:
    write_json(COOLDOWN_FILE, cd)

def in_cooldown(symbol: str, cd: Dict[str, str]) -> bool:
    until = cd.get(symbol.upper())
    if not until:
        return False
    try:
        return datetime.fromisoformat(until.replace("Z", "+00:00")) > now_utc()
    except Exception:
        return False

def set_cooldown(symbol: str, minutes: int, cd: Dict[str, str]) -> None:
    until = now_utc() + timedelta(minutes=minutes)
    cd[symbol.upper()] = iso(until).replace("+00:00", "Z")

# ---------- Candidates ----------
def load_candidates(path: Path) -> List[str]:
    """
    Reads first column from CSV and returns list of pairs, e.g., ["ALCX/USD", "1INCH/USD", ...]
    Accepts header names like 'pair', 'symbol' or any first column.
    """
    if not path.exists():
        return []
    out: List[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return out
        # Detect header if non-symbol-like
        start_idx = 0
        if rows and rows[0]:
            head = rows[0][0].lower()
            if any(k in head for k in ("pair", "symbol")):
                start_idx = 1
        for r in rows[start_idx:]:
            if not r:
                continue
            sym = r[0].strip()
            if not sym:
                continue
            # normalize like "ALCXUSD"
            sym_norm = sym.replace(" ", "").replace("-", "").upper()
            # keep slash form for presentation; use no-slash for Kraken endpoints
            if "/" not in sym_norm and len(sym_norm) > 3:
                # keep both; we will render "AAA/BBB" in summary, but Kraken uses noslash
                sym_disp = f"{sym_norm[:-3]}/{sym_norm[-3:]}"
            else:
                sym_disp = sym.replace(" ", "").upper()
            out.append(sym_disp)
    # de-dup preserving order
    seen = set()
    unique = []
    for s in out:
        key = s.replace("/", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique

# ---------- Position ----------
def load_position() -> Optional[Dict[str, Any]]:
    return read_json(POS_FILE, None)

def save_position(pos: Optional[Dict[str, Any]]) -> None:
    if pos is None:
        if POS_FILE.exists():
            POS_FILE.unlink(missing_ok=True)
    else:
        write_json(POS_FILE, pos)

# ---------- Risk-Off gate (optional) ----------
def risk_off() -> bool:
    try:
        if not RISK_FILE.exists():
            return False
        txt = RISK_FILE.read_text().strip().upper()
        return "OFF" in txt
    except Exception:
        return False

# ---------- Core logic ----------
def decide_exit(pos: Dict[str, Any], price_now: float) -> Tuple[bool, str, float]:
    """
    Returns (should_exit, reason, gain_pct_signed)
    """
    avg = float(pos.get("avg_price", 0.0) or 0.0)
    if avg <= 0 or price_now <= 0:
        return False, "no-price", 0.0
    pnl = (price_now - avg) / avg
    # Rules
    if pnl <= -0.01:
        return True, "STOP_1pct", pnl
    if pnl >= 0.05:
        return True, "TAKE_PROFIT_5pct", pnl
    # 1h rotate if gain < 3%
    entry_iso = pos.get("entry_ts")
    try:
        entry_dt = datetime.fromisoformat(entry_iso.replace("Z", "+00:00"))
    except Exception:
        entry_dt = now_utc() - timedelta(hours=99)
    age = (now_utc() - entry_dt).total_seconds()
    if age >= 3600 and pnl < 0.03:
        return True, "ROTATE_LT3pct_AFTER_1h", pnl
    return False, "HOLD", pnl

def pick_next(cands: List[str], cooldown: Dict[str, str]) -> Optional[str]:
    for sym in cands:
        key = sym.replace("/", "")
        if not in_cooldown(key, cooldown):
            return sym
    return None

def md_header(lines: List[str]) -> None:
    lines.append(f"**When:** {iso(now_utc())}")
    lines.append(f"**Mode:** {'DRY_RUN' if env_str('DRY_RUN','ON').upper()!='OFF' else 'LIVE'}")
    lines.append(f"**Status:** (pending)")
    lines.append(f"**Universe:** AUTO (UNIVERSE_PICK not set)" if not env_str("UNIVERSE_PICK","") else f"**Universe:** PICK={env_str('UNIVERSE_PICK','')}")
    lines.append("")

def write_summary(status: str,
                  details: Dict[str, Any],
                  candidates_preview: List[str]) -> None:
    data = {
        "when": iso(now_utc()),
        "status": status,
        **details
    }
    write_json(SUMMARY_JSON, data)

    lines: List[str] = []
    md_header(lines)
    lines[2] = f"**Status:** {status}"
    if candidates_preview:
        lines.append("**Candidate audit (top of momentum_candidates.csv):**")
        lines.append("")
        lines.append("```")
        for s in candidates_preview[:10]:
            lines.append(s)
        lines.append("```")
        lines.append("")
    if details.get("last_action_json"):
        lines.append("```json")
        lines.append(json.dumps(details["last_action_json"], indent=2))
        lines.append("```")
    write_text(SUMMARY_MD, "\n".join(lines))
    write_text(LAST_EXIT, "0" if status.endswith("OK") else "1")

# ---------- Runner ----------
def main() -> int:
    dry_run = env_str("DRY_RUN", "ON").upper() != "OFF"
    buy_usd = env_float("BUY_USD", 10.0)
    force_pick = env_str("UNIVERSE_PICK", "").strip()
    cooldown_min = int(env_float("COOLDOWN_MIN", 60))

    # Load candidates
    cands = load_candidates(CANDIDATES)
    cand_preview = cands[:]
    if force_pick:
        # Put forced pick at the top if present; else insert at top
        f_disp = force_pick.replace(" ", "").upper()
        if "/" not in f_disp and len(f_disp) > 3:
            f_disp = f"{f_disp[:-3]}/{f_disp[-3:]}"
        cands = [f_disp] + [c for c in cands if c.replace("/", "") != f_disp.replace("/", "")]
    cooldown = load_cooldown()

    # Risk off?
    if risk_off():
        write_summary("RISK_OFF — no trades", {"reason": "risk_off_file"}, cand_preview)
        return 0

    api_key = env_str("KRAKEN_API_KEY", "")
    api_secret = env_str("KRAKEN_API_SECRET", "")

    pos = load_position()

    # -------- EXIT PHASE --------
    last_action: Dict[str, Any] = {}
    if pos:
        sym = pos["symbol"]  # "ALCXUSD"
        disp = f"{sym[:-3]}/{sym[-3:]}" if "/" not in sym else sym
        # fetch price
        px = kraken_public_ticker(sym)
        if px is None:
            # If public price fails, hold to avoid blind exits
            write_summary("HOLD (price unavailable)", {"position": pos, "symbol": sym}, cand_preview)
            return 0
        should_exit, reason, pnl = decide_exit(pos, px)
        if should_exit:
            # SELL
            qty = float(pos.get("qty", 0.0))
            ok, tx = (True, "DRY_TX") if dry_run else kraken_place_market(api_key, api_secret, sym, "sell", qty)
            last_action = {
                "action": "SELL",
                "symbol": sym,
                "qty": qty,
                "price_now": px,
                "reason": reason,
                "pnl_pct": round(pnl * 100, 3),
                "txid": tx,
                "ok": ok
            }
            # set cooldown
            set_cooldown(sym, cooldown_min, cooldown)
            save_cooldown(cooldown)
            # clear position if sell ok
            if ok:
                save_position(None)
            status = f"LIVE SELL OK" if not dry_run and ok else ("DRY SELL OK" if dry_run and ok else "SELL ERROR")
            write_summary(status, {"last_action_json": last_action}, cand_preview)
            # After a sell we fall through to BUY phase (rotation) if ok
            pos = load_position()

    # -------- BUY PHASE --------
    if pos:
        # Already holding => nothing to do
        write_summary("HOLD (position open)", {"position": pos}, cand_preview)
        return 0

    # pick next candidate not in cooldown
    nxt = pick_next(cands, cooldown)
    if not nxt:
        write_summary("NO-CANDIDATE (all cooled down or empty list)", {"cooldown_keys": list(cooldown.keys())}, cand_preview)
        return 0

    sym_noslash = nxt.replace("/", "")
    # get price for sizing
    px = kraken_public_ticker(sym_noslash)
    if px is None or px <= 0:
        write_summary("SKIP (no price for candidate)", {"candidate": nxt}, cand_preview)
        return 0
    volume = max(buy_usd / px, 0.00000001)

    ok, tx = (True, "DRY_TX") if dry_run else kraken_place_market(api_key, api_secret, sym_noslash, "buy", volume)
    last_action = {
        "action": "BUY",
        "symbol": sym_noslash,
        "qty": volume,
        "avg_price": px,
        "txid": tx,
        "ok": ok
    }
    if ok:
        pos = {
            "symbol": sym_noslash,
            "qty": volume,
            "avg_price": px,
            "entry_ts": iso(now_utc()).replace("+00:00", "Z")
        }
        save_position(pos)

    status = f"LIVE BUY OK" if not dry_run and ok else ("DRY BUY OK" if dry_run and ok else "BUY ERROR")
    write_summary(status, {"last_action_json": last_action, "position": pos}, cand_preview)
    return 0

# ---------- Entry ----------
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Ensure summary exists even on crash
        write_summary("FATAL ERROR", {"error": str(e)}, [])
        raise
