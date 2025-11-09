#!/usr/bin/env python3
"""
Sell Guard for Crypto — 1-Coin Rotation

Rules (all configurable via env/kwargs):
- STOP_PCT:   sell if drop >= STOP_PCT since entry (applies at all times).
- TP_PCT:     sell if gain >= TP_PCT since entry.
- SLOW_GAIN:  after WINDOW_MIN minutes, if gain < SLOW_GAIN_REQ → sell (rotate).

Artifacts written:
- .state/sell_log.md     (append-only log)
- .state/last_sell.json  (details of the last sell)
- .state/run_summary.md  (updated note appended)
- .state/positions.json  (deleted when sold)
"""

from __future__ import annotations
import json, os, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import ccxt

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
POS_FILE      = STATE / "positions.json"
SELL_LOG_MD   = STATE / "sell_log.md"
LAST_SELL_JSON= STATE / "last_sell.json"
SUMMARY_MD    = STATE / "run_summary.md"

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try: return float(v)
    except (TypeError, ValueError): return float(default)

def _kraken():
    key = env_str("KRAKEN_API_KEY", env_str("KRAKEN_KEY",""))
    sec = env_str("KRAKEN_API_SECRET", env_str("KRAKEN_SECRET",""))
    ex = ccxt.kraken({
        "apiKey": key,
        "secret": sec,
        "enableRateLimit": True,
    })
    ex.load_markets()
    return ex, bool(key and sec)

def _parse_when(s: str) -> Optional[datetime]:
    try:
        # "YYYY-MM-DD HH:MM:SS"
        if "UTC" in s: s = s.replace(" UTC", "")
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.fromisoformat(s.replace("Z","")).replace(tzinfo=timezone.utc)
        except Exception:
            return None

def _append_summary_note(text: str) -> None:
    try:
        if SUMMARY_MD.exists():
            SUMMARY_MD.write_text(SUMMARY_MD.read_text() + f"\n{text}\n")
    except Exception:
        pass

def _append_sell_log(line: str) -> None:
    header = (
        "# Sell Log\n"
        "When (UTC) | Pair | Amount | Entry | Price | Pct | Reason | Mode | OrderId\n"
        "---|---|---:|---:|---:|---:|---|---|---\n"
    )
    if not SELL_LOG_MD.exists():
        SELL_LOG_MD.write_text(header)
    with SELL_LOG_MD.open("a") as f:
        f.write(line + "\n")

def run_sell_guard(
    *,
    dry_run: Optional[bool] = None,
    tp_pct: Optional[float] = None,
    stop_pct: Optional[float] = None,
    window_min: Optional[int] = None,
    slow_gain_req: Optional[float] = None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Returns: {"action": "hold"|"sold", "note": "..."}
    """
    # Defaults from env
    if dry_run is None:
        dry_run = (env_str("DRY_RUN", "ON").upper() != "OFF")
    tp  = tp_pct if tp_pct is not None else env_float("TP_PCT", 5.0)
    sl  = stop_pct if stop_pct is not None else env_float("STOP_PCT", 1.0)
    win = window_min if window_min is not None else int(env_float("WINDOW_MIN", 30.0))
    slow_req = slow_gain_req if slow_gain_req is not None else env_float("SLOW_GAIN_REQ", 3.0)

    if not POS_FILE.exists():
        return {"action": "hold", "note": "No open position."}

    pos = json.loads(POS_FILE.read_text())
    pair   = pos.get("pair")
    amount = float(pos.get("amount", 0))
    est_cost = float(pos.get("est_cost", 0))
    when_s = pos.get("when", "")
    if not pair or amount <= 0 or est_cost <= 0:
        return {"action": "hold", "note": "Position file incomplete; skipping sell."}

    entry = est_cost / amount
    opened = _parse_when(when_s)
    age_min = 0.0
    if opened:
        age_min = max(0.0, (datetime.now(timezone.utc) - opened).total_seconds() / 60.0)

    ex, have_keys = _kraken()
    # Current price
    try:
        t = ex.fetch_ticker(pair)
        price = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        price = float(price)
    except Exception:
        return {"action": "hold", "note": f"Could not fetch price for {pair}; holding."}

    pct = (price / entry - 1.0) * 100.0

    reason = None
    if pct <= -abs(sl):               # hard stop — applies any time
        reason = f"STOP {sl:.2f}% hit"
    elif pct >= abs(tp):              # take profit
        reason = f"TP {tp:.2f}% hit"
    elif age_min >= float(win) and pct < slow_req:  # slow gain after window
        reason = f"Slow gain (<{slow_req:.2f}% after {win}m)"

    if not reason:
        return {"action": "hold", "note": f"Holding {pair}: +{pct:.2f}% (age {age_min:.1f}m)."}

    # Execute SELL
    mode = "DRY" if (dry_run or not have_keys) else "LIVE"
    order_id = "-"
    try:
        if mode == "LIVE":
            order = ex.create_market_sell_order(pair, amount)
            order_id = order.get("id", "-")
    except ccxt.BaseError as e:
        note = f"[SELL ERROR] {pair} amount={amount:.8f} @~{price:.8f}: {e}"
        _append_summary_note(note)
        return {"action": "hold", "note": note}

    # Log & cleanup
    when_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{when_utc} | {pair} | {amount:.8f} | {entry:.8f} | {price:.8f} | {pct:.2f} | {reason} | {mode} | {order_id}"
    _append_sell_log(line)

    LAST_SELL_JSON.write_text(json.dumps({
        "when": when_utc, "pair": pair, "amount": amount,
        "entry": round(entry, 10), "price": round(price, 10), "pct": round(pct, 4),
        "reason": reason, "mode": mode, "order_id": order_id
    }, indent=2))

    try:
        POS_FILE.unlink(missing_ok=True)  # no longer holding
    except Exception:
        pass

    _append_summary_note(f"[{mode}] SOLD {pair}: {reason} at {price:.8f} (from {entry:.8f}, {pct:.2f}%).")
    return {"action": "sold", "note": f"{reason}; {mode} sell executed."}
