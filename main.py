#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
- One-position rotation (buy top candidate, then manage with sell-guard)
- Resilient quotes via trader.crypto_engine (CSV + public fallback)
- Clear, minimal state in .state/
- DRY_RUN simulates; LIVE validates secrets and logs intended action

Env / Repo Variables expected (strings or numbers):
  DRY_RUN            -> "ON" or "OFF"  (default "ON")
  BUY_USD            -> float dollars  (default 25.0)
  TP_PCT             -> float percent  (default 5.0)
  SL_PCT             -> float percent  (default 1.0)
  SLOW_GAIN_PCT      -> float percent  (default 3.0)
  SLOW_WINDOW_MIN    -> float minutes  (default 60.0)
  UNIVERSE_PICK      -> optional pair override (e.g., "SOL/USD")

Secrets for LIVE (validated; order placement is intentionally no-op here):
  KRAKEN_API_KEY
  KRAKEN_API_SECRET
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


def now_ts() -> float:
    return time.time()


def minutes_since(ts: float) -> float:
    return max(0.0, (now_ts() - float(ts)) / 60.0)


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
        return Position(
            pair=str(data.get("pair", "")),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_ts=float(data.get("entry_ts", 0.0)),
            buy_usd=float(data.get("buy_usd", 0.0)),
        )
    except Exception:
        return None


def write_position(pos: Optional[Position]) -> None:
    if pos is None:
        # clear file
        POSITIONS_JSON.write_text(json.dumps({}, indent=2))
    else:
        POSITIONS_JSON.write_text(json.dumps(pos.to_dict(), indent=2))


def write_summary(payload: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))
    # small human-readable MD
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


def pick_pair(universe_pick: str) -> str:
    """
    Returns the chosen pair for BUY.
    If UNIVERSE_PICK set, try that first; otherwise top-ranked from candidates.
    """
    if universe_pick:
        return normalize_pair(universe_pick)

    candidates = load_candidates()
    if not candidates:
        # crypto_engine guarantees at least 1 with valid quote, but double-guard here
        return "EAT/USD"

    # Already sorted by rank desc; pick the first
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
    if q is None or q <= 0:
        raise RuntimeError(f"BUY abort: invalid quote for {pair}")
    pos = Position(pair=pair, entry_price=float(q), entry_ts=now_ts(), buy_usd=float(buy_usd))
    write_position(pos)
    print(f"[DRY] BUY {pair} @ {q:.6f} for ${buy_usd:.2f}")
    return pos


def simulate_sell(pos: Position, reason: str, current_price: float) -> Dict[str, Any]:
    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100.0
    write_position(None)
    print(f"[DRY] SELL {pos.pair} @ {current_price:.6f}  PnL={pnl_pct:.3f}%  ({reason})")
    return {"pnl_pct": pnl_pct, "reason": reason}


def live_buy_notice(pair: str, buy_usd: float) -> None:
    if not have_live_secrets():
        print("[LIVE] Missing Kraken API secrets; aborting order.")
        return
    # Safe default: log the intended action. (Your existing Kraken adapter can be wired here.)
    q = get_quote(pair)
    print(f"[LIVE] Would BUY {pair} for ${buy_usd:.2f} at ~{q if q else 'N/A'} (market).")


def live_sell_notice(pair: str, reason: str) -> None:
    if not have_live_secrets():
        print("[LIVE] Missing Kraken API secrets; aborting order.")
        return
    q = get_quote(pair)
    print(f"[LIVE] Would SELL {pair} at ~{q if q else 'N/A'} (market). Reason: {reason}")


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
        if not cur or cur <= 0:
            # With new crypto_engine this should be rare, but handle gracefully.
            reason = "HOLD: invalid price(s)"
            print(reason)
            write_summary({
                "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "dry_run": dry,
                "action": "HOLD",
                "pair": pos.pair,
                "reason": reason,
                "entry_price": pos.entry_price,
                "current_price": None,
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
        pnl_pct = guard.get("pnl_pct", (cur - pos.entry_price) / pos.entry_price * 100.0)

        if action == "SELL":
            if dry:
                result = simulate_sell(pos, reason, cur)
                pnl_pct = result["pnl_pct"]
            else:
                live_sell_notice(pos.pair, reason)
                # We don't mutate state in LIVE here; assume your live adapter clears it after a true fill.
                # To keep state consistent even without a live adapter, we'll clear locally:
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
        # With the new engine + YAML seed, this should not happen. Still guard.
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
        # Mirror DRY behavior for continuity of state unless you have a live adapter updating this.
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
