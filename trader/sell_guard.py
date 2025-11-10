#!/usr/bin/env python3
"""
sell_guard.py
Live/Dry sell-guard with:
- Fixed STOP %
- Fixed TAKE-PROFIT %
- Trailing stop (arms after TRAIL_START_PCT gain, trails by TRAIL_BACKOFF_PCT)
- Break-even stop (raises floor to entry once BE_TRIGGER_PCT gain seen)
- Cooldown & no-rebuy support via state helpers (timestamps written by engine)

This module is intentionally self-contained and side-effect free for logic.
Execution (placing orders, writing files) is delegated to call-backs provided
by the engine (crypto_engine.py).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import json
import time
import math

STATE_DIR = Path(".state")
TRAIL_STATE = STATE_DIR / "trail_state.json"

# ---------- small utils ----------

def utc_now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

# ---------- config & state ----------

@dataclass
class GuardCfg:
    stop_pct: float            # e.g., 1.0
    tp_pct: float              # e.g., 5.0
    trail_start_pct: float     # e.g., 3.0 (arm trailing after this gain)
    trail_backoff_pct: float   # e.g., 0.8 (sell if drop >= this % from high)
    be_trigger_pct: float      # e.g., 2.0 (raise stop to at least entry)


@dataclass
class Position:
    pair: str
    amount: float
    entry_px: float            # average entry price


@dataclass
class GuardDecision:
    action: str                # "hold" | "sell"
    reason: str
    new_high: Optional[float] = None   # for external state update


# ---------- trailing/break-even helpers ----------

def _trail_state_get(pair: str) -> Dict[str, float]:
    data = _read_json(TRAIL_STATE, {})
    return data.get(pair, {"high": 0.0, "be_armed": False})

def _trail_state_set(pair: str, high: float, be_armed: bool) -> None:
    data = _read_json(TRAIL_STATE, {})
    data[pair] = {"high": float(high), "be_armed": bool(be_armed)}
    _write_json(TRAIL_STATE, data)

def _pct_change(from_px: float, to_px: float) -> float:
    if from_px <= 0.0:
        return 0.0
    return (to_px / from_px - 1.0) * 100.0


# ---------- core logic ----------

def evaluate_guard(pos: Position, cur_price: float, cfg: GuardCfg) -> GuardDecision:
    """
    Decide whether to sell or hold based on:
      STOP%, TP%, Trailing stop (armed after start gain), Break-even.
    """
    # base floors/ceilings
    stop_floor = pos.entry_px * (1.0 - cfg.stop_pct / 100.0)
    tp_level  = pos.entry_px * (1.0 + cfg.tp_pct   / 100.0)

    # trail state
    t = _trail_state_get(pos.pair)
    high = max(float(t.get("high", 0.0)), pos.entry_px, cur_price)
    be_armed = bool(t.get("be_armed", False))

    # update high-water mark
    if cur_price > high:
        high = cur_price

    # arm BE (raise stop to at least entry) if gain reached
    gain_pct = _pct_change(pos.entry_px, high)
    if gain_pct >= cfg.be_trigger_pct:
        be_armed = True
        stop_floor = max(stop_floor, pos.entry_px)

    # arm trailing once start gain reached
    if gain_pct >= cfg.trail_start_pct:
        trail_floor = high * (1.0 - cfg.trail_backoff_pct / 100.0)
        stop_floor = max(stop_floor, trail_floor)

    # decision
    if cur_price <= stop_floor:
        reason = []
        if cur_price <= pos.entry_px * (1.0 - cfg.stop_pct / 100.0):
            reason.append(f"STOP {cfg.stop_pct:.2f}% hit")
        if cur_price <= high * (1.0 - cfg.trail_backoff_pct / 100.0) and gain_pct >= cfg.trail_start_pct:
            reason.append(f"TRAIL {cfg.trail_backoff_pct:.2f}% from high")
        if be_armed and cur_price <= pos.entry_px:
            reason.append("BE floor")
        rsn = " & ".join(reason) if reason else "Stop floor"
        return GuardDecision(action="sell", reason=rsn, new_high=high)

    if cur_price >= tp_level:
        return GuardDecision(action="sell", reason=f"TP {cfg.tp_pct:.2f}% hit", new_high=high)

    return GuardDecision(action="hold", reason="Hold", new_high=high)


# ---------- execution wrapper (called by engine) ----------

def run_sell_guard(
    pos: Position,
    cur_price: float,
    cfg: GuardCfg,
    mode: str,
    place_sell_fn: Callable[[str, float, str], Tuple[bool, str]],
    write_sell_artifacts_fn: Callable[[Dict], None],
) -> Tuple[bool, str]:
    """
    Evaluates guard and, if needed, places a sell via the provided callback.

    place_sell_fn(pair, amount, mode) -> (ok, order_id_or_msg)
    write_sell_artifacts_fn(payload_dict) -> None     # engine logs MD/JSON, cooldown, etc.

    Returns (did_sell, message)
    """
    decision = evaluate_guard(pos, cur_price, cfg)
    _trail_state_set(pos.pair, decision.new_high or pos.entry_px, gain_reached(pos, decision.new_high or pos.entry_px, cfg))

    if decision.action == "sell":
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        payload = {
            "when": utc_now_str(),
            "pair": pos.pair,
            "amount": round(float(pos.amount), 8),
            "entry": round(float(pos.entry_px), 8),
            "price": round(float(cur_price), 8),
            "pct": round(_pct_change(pos.entry_px, cur_price), 4),
            "reason": decision.reason,
            "mode": mode.upper(),
            "order_id": oid,
        }
        write_sell_artifacts_fn(payload)
        return ok, f"SOLD {pos.pair}: {decision.reason}"
    else:
        return False, decision.reason


def gain_reached(pos: Position, high: float, cfg: GuardCfg) -> bool:
    return _pct_change(pos.entry_px, high) >= cfg.be_trigger_pct
