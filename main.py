#!/usr/bin/env python3
"""
main.py — Crypto 1-coin rotation for Kraken (LIVE/DRY-RUN)

Rules (unchanged):
  • 1% stop-loss
  • 5% take-profit
  • "< 3% gain after 1 hour" => rotate out

Enhancements:
  • Always writes run_summary.md/json with rich diagnostics (PnL, prices, entry age, reasons)
  • Accepts legacy position schemas:
        - {"symbol","qty","avg_price","entry_ts"}  (ISO)
        - {"symbol","qty","entry_price","opened_at"} (opened_at = epoch float)
  • Optional fallback-rotate (disabled by default):
        If FALLBACK_ROTATE=ON and age >= AGE_FALLBACK_MIN and pnl <= ROTATE_IF_PNL_BELOW
        and top candidate != current and not in cooldown => rotate

Env:
  DRY_RUN: "ON"|"OFF"                (default "ON")
  BUY_USD: float, USD per buy        (default "10")
  UNIVERSE_PICK: "ALCX/USD" etc      (default "")
  COOLDOWN_MIN: int minutes          (default "60")

  FALLBACK_ROTATE: "ON"|"OFF"        (default "OFF")
  AGE_FALLBACK_MIN: int minutes      (default "60")
  ROTATE_IF_PNL_BELOW: float (frac)  (default "0.0")  # 0% or worse after fallback age

Kraken LIVE:
  KRAKEN_API_KEY
  KRAKEN_API_SECRET (base64)
"""

import base64
import hashlib
import hmac
import json
import os
import time
import csv
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- Paths ----------
STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "position.json"
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD = STATE / "run_summary.md"
LAST_EXIT = STATE / "last_exit_code.txt"
RISK_FILE = STATE / "last_risk_signal.txt"
COOLDOWN_FILE = STATE / "cooldown.json"
CANDIDATES = STATE / "momentum_candidates.csv"

# ---------- Helpers ----------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, ""))
    except Exception:
        return default

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()

# ---------- Kraken minimal client ----------
KRAKEN_BASE = "https://api.kraken.com"

def _kraken_request(path: str, data: Dict[str, Any], key: str, secret_b64: str) -> Dict[str, Any]:
    url = f"{KRAKEN_BASE}{path}"
    nonce = str(int(time.time() * 1000))
    pdata = {"nonce": nonce}
    pdata.update(data)
    enc = urllib.parse.urlencode(pdata)
    post_bytes = enc.encode()

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
        return json.loads(resp.read().decode())

def kraken_public_ticker(pair: str) -> Optional[float]:
    pair_ns = pair.replace("/", "").upper()
    url = f"{KRAKEN_BASE}/0/public/Ticker?pair={urllib.parse.quote(pair_ns)}"
    req = urllib.request.Request(url, headers={"User-Agent": "stock-analyzer-app"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            if data.get("error"):
                return None
            res = data.get("result", {})
            if not res:
                return None
            k = next(iter(res.keys()))
            last = res[k]["c"][0]
            return float(last)
    except Exception:
        return None

def kraken_place_market(key: str, secret_b64: str, pair_ns: str, side: str, volume: float) -> Tuple[bool, str]:
    data = {
        "pair": pair_ns,
        "type": side,
        "ordertype": "market",
        "volume": f"{volume:.8f}",
        "oflags": "viqc",
    }
    try:
        resp = _kraken_request("/0/private/AddOrder", data, key, secret_b64)
        if resp.get("error"):
            return False, ";".join(resp["error"])
        txs = resp.get("result", {}).get("txid", [])
        return True, ",".join(txs) if txs else "OK"
    except Exception as e:
        return False, f"EXC:{e}"

# ---------- File I/O ----------
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

def in_cooldown(symbol_ns: str, cd: Dict[str, str]) -> bool:
    until = cd.get(symbol_ns.upper())
    if not until:
        return False
    try:
        return datetime.fromisoformat(until.replace("Z", "+00:00")) > now_utc()
    except Exception:
        return False

def set_cooldown(symbol_ns: str, minutes: int, cd: Dict[str, str]) -> None:
    until = now_utc() + timedelta(minutes=minutes)
    cd[symbol_ns.upper()] = iso(until).replace("+00:00", "Z")

# ---------- Candidates ----------
def load_candidates(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open(newline="") as f:
        r = csv.reader(f)
        rows = list(r)
        if not rows:
            return out
        start = 0
        if rows[0] and any(k in rows[0][0].lower() for k in ("pair", "symbol")):
            start = 1
        for row in rows[start:]:
            if not row:
                continue
            s = row[0].strip().upper().replace(" ", "")
            if not s:
                continue
            if "/" not in s and len(s) > 3:
                s = f"{s[:-3]}/{s[-3:]}"
            out.append(s)
    # de-dupe
    seen, uniq = set(), []
    for s in out:
        k = s.replace("/", "")
        if k in seen: 
            continue
        seen.add(k)
        uniq.append(s)
    return uniq

# ---------- Position ----------
def load_position() -> Optional[Dict[str, Any]]:
    """
    Returns normalized position dict:
      {"symbol":"ALCXUSD","qty":float,"avg_price":float,"entry_ts": ISO str}
    """
    raw = read_json(POS_FILE, None)
    if not raw:
        return None
    sym = raw.get("symbol") or raw.get("pair") or ""
    sym = sym.replace("/", "").upper()
    qty = float(raw.get("qty", 0) or 0)
    avg_price = raw.get("avg_price", None)
    entry_ts = raw.get("entry_ts", None)

    # Legacy compatibility
    if avg_price is None and "entry_price" in raw:
        avg_price = float(raw["entry_price"])
    if not entry_ts and "opened_at" in raw:
        try:
            # epoch (float or int) -> ISO
            entry_ts = iso(datetime.fromtimestamp(float(raw["opened_at"]), tz=timezone.utc))
        except Exception:
            entry_ts = None

    if not sym or qty <= 0 or avg_price is None:
        return None
    if not entry_ts:
        entry_ts = iso(now_utc())  # best-effort

    return {"symbol": sym, "qty": float(qty), "avg_price": float(avg_price), "entry_ts": entry_ts}

def save_position(pos: Optional[Dict[str, Any]]) -> None:
    if pos is None:
        POS_FILE.unlink(missing_ok=True)
    else:
        write_json(POS_FILE, pos)

# ---------- Risk gate ----------
def risk_off() -> bool:
    try:
        if not RISK_FILE.exists():
            return False
        return "OFF" in RISK_FILE.read_text().strip().upper()
    except Exception:
        return False

# ---------- Decision ----------
def decide_exit(pos: Dict[str, Any], price_now: float, top_candidate_ns: Optional[str], cfg: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    avg = float(pos["avg_price"])
    pnl = (price_now - avg) / avg if avg > 0 else 0.0

    try:
        entry_dt = datetime.fromisoformat(pos["entry_ts"].replace("Z", "+00:00"))
    except Exception:
        entry_dt = now_utc()
    age_sec = (now_utc() - entry_dt).total_seconds()
    age_min = age_sec / 60.0

    # Primary rules
    if pnl <= -0.01:
        return True, "STOP_1pct", {"pnl": pnl, "age_min": age_min}
    if pnl >= 0.05:
        return True, "TAKE_PROFIT_5pct", {"pnl": pnl, "age_min": age_min}
    if age_sec >= 3600 and pnl < 0.03:
        return True, "ROTATE_LT3pct_AFTER_1h", {"pnl": pnl, "age_min": age_min}

    # Optional fallback rotation (disabled by default)
    if cfg["FALLBACK_ROTATE"]:
        if age_min >= cfg["AGE_FALLBACK_MIN"] and pnl <= cfg["ROTATE_IF_PNL_BELOW"]:
            if top_candidate_ns and top_candidate_ns != pos["symbol"]:
                return True, "FALLBACK_ROTATE", {"pnl": pnl, "age_min": age_min}

    return False, "HOLD", {"pnl": pnl, "age_min": age_min}

# ---------- Summary ----------
def write_summary(status: str, details: Dict[str, Any], cand_preview: List[str]) -> None:
    data = {"when": iso(now_utc()), "status": status, **details}
    write_json(SUMMARY_JSON, data)

    lines = []
    lines.append(f"**When:** {data['when']}")
    lines.append(f"**Mode:** {'DRY_RUN' if details.get('dry_run') else 'LIVE'}")
    lines.append(f"**Status:** {status}")
    lines.append(f"**Universe:** {'PICK=' + details['forced'] if details.get('forced') else 'AUTO (UNIVERSE_PICK not set)'}")
    lines.append("")
    if cand_preview:
        lines.append("**Candidate audit (top of momentum_candidates.csv):**")
        lines.append("")
        lines.append("```")
        for s in cand_preview[:10]:
            lines.append(s)
        lines.append("```")
        lines.append("")
    # Diagnostics
    diag = details.get("diagnostics")
    if diag:
        lines.append("**Diagnostics:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(diag, indent=2))
        lines.append("```")
        lines.append("")
    if details.get("last_action_json"):
        lines.append("```json")
        lines.append(json.dumps(details["last_action_json"], indent=2))
        lines.append("```")
    write_text(SUMMARY_MD, "\n".join(lines))
    write_text(LAST_EXIT, "0" if status.endswith("OK") or status.startswith("HOLD") else "1")

# ---------- Runner ----------
def main() -> int:
    dry_run = env_str("DRY_RUN", "ON").upper() != "OFF"
    buy_usd = env_float("BUY_USD", 10.0)
    force_pick = env_str("UNIVERSE_PICK", "").strip()
    cooldown_min = int(env_float("COOLDOWN_MIN", 60))

    # Fallback rotate config
    cfg = {
        "FALLBACK_ROTATE": env_str("FALLBACK_ROTATE", "OFF").upper() == "ON",
        "AGE_FALLBACK_MIN": int(env_float("AGE_FALLBACK_MIN", 60)),
        "ROTATE_IF_PNL_BELOW": env_float("ROTATE_IF_PNL_BELOW", 0.0),
    }

    # Candidates
    cands = load_candidates(CANDIDATES)
    cand_preview = cands[:]
    if force_pick:
        fp = force_pick.replace(" ", "").upper()
        if "/" not in fp and len(fp) > 3:
            fp = f"{fp[:-3]}/{fp[-3:]}"
        cands = [fp] + [c for c in cands if c.replace("/", "") != fp.replace("/", "")]
    cooldown = load_cooldown()

    # Risk-off
    if risk_off():
        write_summary("RISK_OFF — no trades", {"dry_run": dry_run}, cand_preview)
        return 0

    api_key = env_str("KRAKEN_API_KEY", "")
    api_secret = env_str("KRAKEN_API_SECRET", "")

    pos = load_position()
    top_candidate_ns = cands[0].replace("/", "") if cands else None

    # -------- EXIT PHASE --------
    last_action: Dict[str, Any] = {}
    if pos:
        sym_ns = pos["symbol"]
        px = kraken_public_ticker(sym_ns)
        diag = {
            "position": pos,
            "price_now": px,
            "top_candidate": top_candidate_ns,
            "cooldown_active": in_cooldown(sym_ns, cooldown),
            "fallback_rotate_enabled": cfg["FALLBACK_ROTATE"],
        }
        if px is None:
            write_summary("HOLD (price unavailable)", {"dry_run": dry_run, "diagnostics": diag}, cand_preview)
            return 0

        should_exit, reason, metrics = decide_exit(pos, px, top_candidate_ns, cfg)
        diag.update(metrics)

        if should_exit:
            qty = float(pos["qty"])
            ok, tx = (True, "DRY_TX") if dry_run else kraken_place_market(api_key, api_secret, sym_ns, "sell", qty)
            last_action = {
                "action": "SELL",
                "symbol": sym_ns,
                "qty": qty,
                "price_now": px,
                "reason": reason,
                "pnl_pct": round(metrics["pnl"] * 100, 3),
                "age_min": round(metrics["age_min"], 1),
                "txid": tx,
                "ok": ok
            }
            set_cooldown(sym_ns, cooldown_min, cooldown)
            save_cooldown(cooldown)
            if ok:
                save_position(None)
            status = f"{'DRY' if dry_run else 'LIVE'} SELL OK" if ok else "SELL ERROR"
            write_summary(status, {"dry_run": dry_run, "last_action_json": last_action, "diagnostics": diag}, cand_preview)
            pos = load_position()  # should be None now

        else:
            write_summary("HOLD (position open)", {"dry_run": dry_run, "diagnostics": diag, "position": pos}, cand_preview)
            return 0  # hold => stop here (single-position bot)

    # -------- BUY PHASE --------
    if pos:
        write_summary("HOLD (position open)", {"dry_run": dry_run, "position": pos}, cand_preview)
        return 0

    # pick next (not in cooldown)
    nxt = None
    for s in cands:
        ns = s.replace("/", "")
        if not in_cooldown(ns, cooldown):
            nxt = s
            break
    if not nxt:
        write_summary("NO-CANDIDATE (all cooled down or none)", {"dry_run": dry_run, "cooldown": list(cooldown.keys())}, cand_preview)
        return 0

    nxt_ns = nxt.replace("/", "")
    px = kraken_public_ticker(nxt_ns)
    if px is None or px <= 0:
        write_summary("SKIP (no price for candidate)", {"dry_run": dry_run, "candidate": nxt}, cand_preview)
        return 0

    volume = max(buy_usd / px, 0.00000001)
    ok, tx = (True, "DRY_TX") if dry_run else kraken_place_market(api_key, api_secret, nxt_ns, "buy", volume)
    last_action = {
        "action": "BUY",
        "symbol": nxt_ns,
        "qty": volume,
        "avg_price": px,
        "txid": tx,
        "ok": ok
    }
    if ok:
        pos = {
            "symbol": nxt_ns,
            "qty": volume,
            "avg_price": px,
            "entry_ts": iso(now_utc()).replace("+00:00", "Z")
        }
        save_position(pos)

    status = f"{'DRY' if dry_run else 'LIVE'} BUY OK" if ok else "BUY ERROR"
    write_summary(status, {"dry_run": dry_run, "last_action_json": last_action, "forced": force_pick or "", "position": pos}, cand_preview)
    return 0

# ---------- Entry ----------
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        write_summary("FATAL ERROR", {"error": str(e), "dry_run": env_str("DRY_RUN","ON").upper()!='OFF'}, [])
        raise
