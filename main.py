#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
- One-position rotation (buy top candidate, then manage with sell-guard)
- Resilient quotes via trader.crypto_engine (CSV + public fallback)
- Clear, minimal state in .state/
- DRY_RUN simulates; LIVE validates secrets and logs intended action
- Hardened against corrupt state (e.g., entry_price == 0) and zero-division
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# --- Robust imports even if "trader" isn't a package ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from trader.crypto_engine import (
        load_candidates,
        get_public_quote,
        normalize_pair,
    )
    from trader.sell_guard import evaluate_sell
except Exception as e:
    print(f"FATAL: cannot import trader modules: {e}", file=sys.stderr)
    sys.exit(1)

STATE_DIR = ROOT / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS_JSON = STATE_DIR / "positions.json"   # keeps current open position (single-position model)
SUMMARY_JSON   = STATE_DIR / "run_summary.json"
SUMMARY_MD     = STATE_DIR / "run_summary.md"
LAST_OK        = STATE_DIR / "last_ok.txt"


# --------------------- env helpers ---------------------

def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    if val is None:
        return ""
    return str(val).strip()


def env_float(name: str, default: float) -> float:
    raw = env_str(name, "")
    try:
        return float(raw) if raw != "" else float(default)
    except Exception:
        return float(default)


def is_dry_run() -> bool:
    return env_str("DRY_RUN", "ON").upper() == "ON"


def sanitize_universe_pick(s: str) -> str:
    """
    Treat AUTO/NONE/ANY/blank as no override.
    """
    v = s.strip().upper()
    if v in ("", "AUTO", "NONE", "ANY", "NULL", "NA"):
        return ""
    return v


# --------------------- time helpers ---------------------

def now_ts() -> float:
    return time.time()


def minutes_since(ts: float) -> float:
    try:
        return max(0.0, (now_ts() - float(ts)) / 60.0)
    except Exception:
        return 0.0


# --------------------- PnL helpers ---------------------

def safe_pct_change(current: Optional[float], entry: Optional[float]) -> float:
    """
    Safe PnL% calculator: returns 0.0 if values are invalid.
    """
    try:
        if current is None or entry is None:
            return 0.0
        current = float(current)
        entry = float(entry)
        if current <= 0 or entry <= 0:
            return 0.0
        return (current - entry) / entry * 100.0
    except Exception:
        return 0.0


# --------------------- position model ---------------------

@dataclass
class Position:
    pair: str
    entry_price: float
    entry_ts: float
    buy_usd: float

    @property
    def minutes_held(self) -> float:
        return minutes_since(self.entry_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_position() -> Optional[Position]:
    if not POSITIONS_JSON.exists():
        return None
    try:
        data = json.loads(POSITIONS_JSON.read_text())
        if not isinstance(data, dict):
            return None
        pos = Position(
            pair=str(data.get("pair", "")),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_ts=float(data.get("entry_ts", 0.0)),
            buy_usd=float(data.get("buy_usd", 0.0)),
        )
        # Auto-repair: if entry_price or ts invalid, treat as no position
        if pos.entry_price <= 0 or pos.entry_ts <= 0 or not pos.pair:
            return None
        return pos
    except Exception:
        return None


def write_position(pos: Optional[Position]) -> None:
    if pos is None:
        POSITIONS_JSON.write_text(json.dumps({}, indent=2))
    else:
        POSITIONS_JSON.write_text(json.dumps(pos.to_dict(), indent=2))


# --------------------- summary writers ---------------------

def write_summary(payload: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))
    lines = [
        f"**When:** {payload.get('when','')}",
        f"**DRY_RUN:** {payload.get('dry_run')}",
        f"**Action:** {payload.get('action')}",
        f"**Pair:** {payload.get('pair')}",
        f"**Reason:** {payload.get('reason','')}",
        f"**Entry Price:** {payload.get('entry_price')}",
        f"**Current Price:** {payload.get('current_price')}",
        f"**PnL %:** {payload.get('pnl_pct')}",
        f"**Held (min):** {payload.get('minutes_held')}",
        "",
        "**Env:**",
        f"- BUY_USD={payload.get('env',{}).get('BUY_USD')}",
        f"- TP_PCT={payload.get('env',{}).get('TP_PCT')}",
        f"- SL_PCT={payload.get('env',{}).get('SL_PCT')}",
        f"- SLOW_GAIN_PCT={payload.get('env',{}).get('SLOW_GAIN_PCT')}",
        f"- SLOW_WINDOW_MIN={payload.get('env',{}).get('SLOW_WINDOW_MIN')}",
        f"- UNIVERSE_PICK={payload.get('env',{}).get('UNIVERSE_PICK')}",
    ]
    SUMMARY_MD.write_text("\n".join(lines))
    LAST_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


# --------------------- core utils ---------------------

def pick_pair(universe_pick: str) -> str:
    """
    Returns the chosen pair for BUY.
    If UNIVERSE_PICK set, try that first; otherwise top-ranked from candidates.
    """
    uni = sanitize_universe_pick(universe_pick)
    if uni:
        return normalize_pair(uni)

    candidates = load_candidates()
    if not candidates:
        return "EAT/USD"
    return normalize_pair(candidates[0]["pair"])


def get_quote(pair: str) -> Optional[float]:
    return get_public_quote(pair)


def have_live_secrets() -> bool:
    return bool(env_str("KRAKEN_API_KEY", "")) and bool(env_str("KRAKEN_API_SECRET", ""))


def log_env() -> Dict[str, Any]:
    env = {
        "DRY_RUN": env_str("DRY_RUN", "ON"),
        "BUY_USD": env_float("BUY_USD", 25.0),
        "TP_PCT": env_float("TP_PCT", 5.0),
        "SL_PCT": env_float("SL_PCT", 1.0),
        "SLOW_GAIN_PCT": env_float("SLOW_GAIN_PCT", 3.0),
        "SLOW_WINDOW_MIN": env_float("SLOW_WINDOW_MIN", 60.0),
        "UNIVERSE_PICK": env_str("UNIVERSE_PICK", ""),
    }
    print("=== ENV ===")
    for k, v in env.items():
        print(f"{k} = {v}")
    print("============")
    return env


def simulate_buy(pair: str, buy_usd: float) -> Position:
    q = get_quote(pair)
    if not q or q <= 0:
        raise RuntimeError(f"BUY abort: invalid quote for {pair}")
    pos = Position(pair=pair, entry_price=float(q), entry_ts=now_ts(), buy_usd=float(buy_usd))
    write_position(pos)
    print(f"[DRY] BUY {pair} @ {q:.6f} for ${buy_usd:.2f}")
    return pos


def simulate_sell(pos: Position, reason: str, current_price: float) -> Dict[str, Any]:
    pnl_pct = safe_pct_change(current_price, pos.entry_price)
    write_position(None)
    print(f"[DRY] SELL {pos.pair} @ {current_price:.6f}  PnL={pnl_pct:.3f}%  ({reason})")
    return {"pnl_pct": pnl_pct, "reason": reason}


def live_buy_notice(pair: str, buy_usd: float) -> None:
    if not have_live_secrets():
        print("[LIVE] Missing Kraken API secrets; aborting order.")
        return
    q = get_quote(pair)
    print(f"[LIVE] Would BUY {pair} for ${buy_usd:.2f} at ~{q if q else 'N/A'} (market).")


def live_sell_notice(pair: str, reason: str) -> None:
    if not have_live_secrets():
        print("[LIVE] Missing Kraken API secrets; aborting order.")
        return
    q = get_quote(pair)
    print(f"[LIVE] Would SELL {pair} at ~{q if q else 'N/A'} (market). Reason: {reason}")


# --------------------- main ---------------------

def main() -> None:
    env = log_env()

    dry = is_dry_run()
    buy_usd = float(env["BUY_USD"])
    tp = float(env["TP_PCT"])
    sl = float(env["SL_PCT"])
    slow_pct = float(env["SLOW_GAIN_PCT"])
    slow_win = float(env["SLOW_WINDOW_MIN"])
    uni_pick = str(env["UNIVERSE_PICK"])

    pos = read_position()

    # === If we have a position, run sell-guard ===
    if pos:
        cur = get_quote(pos.pair)
        if not cur or cur <= 0 or pos.entry_price <= 0:
            reason = "HOLD: invalid price(s)"
            print(reason)
            write_summary({
                "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "dry_run": dry,
                "action": "HOLD",
                "pair": pos.pair,
                "reason": reason,
                "entry_price": pos.entry_price if pos.entry_price > 0 else None,
                "current_price": cur if cur and cur > 0 else None,
                "pnl_pct": None,
                "minutes_held": pos.minutes_held,
                "env": env,
            })
            return

        guard = evaluate_sell(
            entry_price=pos.entry_price,
            current_price=cur,
            minutes_since_entry=pos.minutes_held,
            tp_pct=tp,
            sl_pct=sl,
            slow_gain_pct=slow_pct,
            slow_window_min=slow_win,
        )

        action = guard.get("action", "HOLD")
        reason = guard.get("reason", "guard ok")
        pnl_pct = guard.get("pnl_pct")
        if pnl_pct is None:
            pnl_pct = safe_pct_change(cur, pos.entry_price)

        if action == "SELL":
            if dry:
                result = simulate_sell(pos, reason, cur)
                pnl_pct = result["pnl_pct"]
            else:
                live_sell_notice(pos.pair, reason)
                write_position(None)

            write_summary({
                "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "dry_run": dry,
                "action": "SELL",
                "pair": pos.pair,
                "reason": reason,
                "entry_price": pos.entry_price,
                "current_price": cur,
                "pnl_pct": pnl_pct,
                "minutes_held": pos.minutes_held,
                "env": env,
            })
            return
        else:
            print(f"HOLD {pos.pair}: {reason}  (held {pos.minutes_held:.1f}m)")
            write_summary({
                "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "dry_run": dry,
                "action": "HOLD",
                "pair": pos.pair,
                "reason": reason,
                "entry_price": pos.entry_price,
                "current_price": cur,
                "pnl_pct": pnl_pct,
                "minutes_held": pos.minutes_held,
                "env": env,
            })
            return

    # === No open position → choose a pair and BUY ===
    pair = pick_pair(uni_pick)
    q = get_quote(pair)
    if not q or q <= 0:
        reason = "BUY abort: invalid quote"
        print(f"{reason} for {pair}")
        write_summary({
            "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dry_run": dry,
            "action": "ABORT",
            "pair": pair,
            "reason": reason,
            "entry_price": None,
            "current_price": None,
            "pnl_pct": None,
            "minutes_held": 0.0,
            "env": env,
        })
        return

    if dry:
        pos = simulate_buy(pair, buy_usd)
    else:
        live_buy_notice(pair, buy_usd)
        pos = Position(pair=pair, entry_price=float(q), entry_ts=now_ts(), buy_usd=float(buy_usd))
        write_position(pos)

    write_summary({
        "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dry_run": dry,
        "action": "BUY",
        "pair": pos.pair,
        "reason": "opened rotation",
        "entry_price": pos.entry_price,
        "current_price": q,
        "pnl_pct": 0.0,
        "minutes_held": 0.0,
        "env": env,
    })


if __name__ == "__main__":
    main()
