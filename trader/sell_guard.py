#!/usr/bin/env python3
"""
sell_guard.py
Stateless SELL guard used by the crypto engine.

Rules supported (stateless):
- STOP_PCT: hard stop if % change <= -STOP_PCT
- TP_PCT:   take profit if % change >=  TP_PCT
- TRAILING: once gain >= TRAIL_START_PCT, sell on dip below
            (TRAIL_START_PCT - TRAIL_BACKOFF_PCT) from entry
            (stateless floor version)
- BREAK-EVEN: once gain >= BE_TRIGGER_PCT, sell if price < entry

This module is intentionally light and pure so it can be called from
any engine. Persistence / cooldown / artifacts are handled by the caller.

Interface:
    did_sell, reason = run_sell_guard(
        pos=Position(...),
        cur_price=...,
        cfg=GuardCfg(...),
        mode="ON" or "OFF",              # optional; OFF = LIVE, ON = DRY
        place_sell_fn=callable,          # (pair:str, amount:float, mode:str) -> (ok:bool, order_id:str)
        write_sell_artifacts_fn=callable # (payload:dict) -> None
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import time


# --------- Data types ---------

@dataclass
class Position:
    pair: str
    amount: float
    entry_px: float


@dataclass
class GuardCfg:
    stop_pct: float = 1.0          # e.g., 1.0  -> sell at -1.0%
    tp_pct: float = 5.0            # e.g., 5.0  -> sell at +5.0%
    trail_start_pct: float = 3.0   # gain threshold to arm a trailing floor
    trail_backoff_pct: float = 0.8 # how much under trail_start to allow before sell
    be_trigger_pct: float = 2.0    # once gain >= this, sell if price < entry


# --------- Helpers ---------

def pct_change(cur: float, ref: float) -> float:
    if ref <= 0:
        return 0.0
    return (cur / ref - 1.0) * 100.0


def _write_sell(
    pos: Position,
    cur_px: float,
    mode: str,
    reason: str,
    order_id: str,
    write_sell_artifacts_fn,
) -> None:
    payload = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "pair": pos.pair,
        "amount": round(pos.amount, 8),
        "entry": round(pos.entry_px, 8),
        "price": round(cur_px, 8),
        "pct": round(pct_change(cur_px, pos.entry_px), 4),
        "reason": reason,
        "mode": mode.upper(),
        "order_id": order_id,
    }
    try:
        write_sell_artifacts_fn(payload)
    except Exception:
        # best-effort; guard should never hard-crash the run due to artifacts
        pass


# --------- Core guard ---------

def run_sell_guard(
    *,
    pos: Position,
    cur_price: float,
    cfg: GuardCfg,
    mode: str = "ON",  # OFF = LIVE, ON = DRY
    place_sell_fn: Callable[[str, float, str], Tuple[bool, str]],
    write_sell_artifacts_fn,
) -> Tuple[bool, str]:
    """
    Returns (did_sell, reason)
    """
    entry = float(pos.entry_px or 0.0)
    px = float(cur_price or 0.0)

    if entry <= 0 or px <= 0:
        return False, "HOLD: invalid price(s)"

    gain = pct_change(px, entry)

    # Rule 1: Hard STOP
    if cfg.stop_pct > 0 and gain <= -abs(cfg.stop_pct):
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(pos, px, mode, f"STOP {cfg.stop_pct:.2f}% hit", oid, write_sell_artifacts_fn)
            return True, f"SOLD {pos.pair}: STOP {cfg.stop_pct:.2f}% hit"
        return False, f"HOLD: stop error {oid}"

    # Rule 2: Take-Profit
    if cfg.tp_pct > 0 and gain >= abs(cfg.tp_pct):
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(pos, px, mode, f"TP {cfg.tp_pct:.2f}% hit", oid, write_sell_artifacts_fn)
            return True, f"SOLD {pos.pair}: TP {cfg.tp_pct:.2f}% hit"
        return False, f"HOLD: tp error {oid}"

    # Rule 3: Stateless Trailing floor
    # Once gain >= trail_start, sell if price falls below a floor defined from entry:
    # floor_gain = trail_start - trail_backoff
    # Example: start=3%, backoff=0.8% → sell if gain <= 2.2% (after it first exceeds 3%)
    if cfg.trail_start_pct > 0:
        floor_gain = cfg.trail_start_pct - max(0.0, cfg.trail_backoff_pct)
        if gain >= cfg.trail_start_pct and gain <= floor_gain:
            ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
            if ok:
                _write_sell(pos, px, mode,
                            f"TRAIL armed@{cfg.trail_start_pct:.2f}% → dip to {floor_gain:.2f}%", oid,
                            write_sell_artifacts_fn)
                return True, f"SOLD {pos.pair}: TRAIL dip to {floor_gain:.2f}%"
            return False, f"HOLD: trail error {oid}"

    # Rule 4: Break-even tighten
    # If we ever get to >= be_trigger, we refuse to go below entry (sell if px < entry).
    if cfg.be_trigger_pct > 0 and gain >= cfg.be_trigger_pct and px < entry:
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(pos, px, mode, f"BE tighten @{cfg.be_trigger_pct:.2f}% (px < entry)", oid,
                        write_sell_artifacts_fn)
            return True, f"SOLD {pos.pair}: BREAK-EVEN tighten"
        return False, f"HOLD: break-even error {oid}"

    return False, f"HOLD: gain {gain:.2f}%"
