#!/usr/bin/env python3
"""
sell_guard.py
Stateless SELL guard used by the crypto engine.

Rules supported (stateless):
- STOP_PCT: hard stop if % change <= -STOP_PCT
- TP_PCT:   take profit if % change >=  TP_PCT
- TRAILING (stateless floor version):
  once gain >= TRAIL_START_PCT, treat a fall back to
  (TRAIL_START_PCT - TRAIL_BACKOFF_PCT) as a sell
- BREAK-EVEN:
  once gain >= BE_TRIGGER_PCT, sell if price < entry

Caller handles persistence/cooldowns/artifacts.

Interface:
    did_sell, reason = run_sell_guard(
        pos=Position(...),
        cur_price=<float>,
        cfg=GuardCfg(...),
        mode="ON" or "OFF",              # OFF = LIVE, ON = DRY
        place_sell_fn=callable,          # (pair:str, amount:float, mode:str) -> (ok:bool, order_id:str)
        write_sell_artifacts_fn=callable # (payload:dict) -> None
    )
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import time

# ---------- Data ----------
@dataclass
class Position:
    pair: str
    amount: float
    entry_px: float

@dataclass
class GuardCfg:
    stop_pct: float = 1.0
    tp_pct: float = 5.0
    trail_start_pct: float = 3.0
    trail_backoff_pct: float = 0.8
    be_trigger_pct: float = 2.0

# ---------- Helpers ----------
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
        pass

# ---------- Core ----------
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

    # 1) STOP
    if cfg.stop_pct > 0 and gain <= -abs(cfg.stop_pct):
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(pos, px, mode, f"STOP {cfg.stop_pct:.2f}% hit", oid, write_sell_artifacts_fn)
            return True, f"SOLD {pos.pair}: STOP {cfg.stop_pct:.2f}% hit"
        return False, f"HOLD: stop error {oid}"

    # 2) TAKE-PROFIT
    if cfg.tp_pct > 0 and gain >= abs(cfg.tp_pct):
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(pos, px, mode, f"TP {cfg.tp_pct:.2f}% hit", oid, write_sell_artifacts_fn)
            return True, f"SOLD {pos.pair}: TP {cfg.tp_pct:.2f}% hit"
        return False, f"HOLD: tp error {oid}"

    # 3) TRAILING (stateless floor)
    if cfg.trail_start_pct > 0:
        floor_gain = cfg.trail_start_pct - max(0.0, cfg.trail_backoff_pct)
        if gain >= cfg.trail_start_pct and gain <= floor_gain:
            ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
            if ok:
                _write_sell(
                    pos, px, mode,
                    f"TRAIL armed@{cfg.trail_start_pct:.2f}% → dip to {floor_gain:.2f}%",
                    oid, write_sell_artifacts_fn
                )
                return True, f"SOLD {pos.pair}: TRAIL dip to {floor_gain:.2f}%"
            return False, f"HOLD: trail error {oid}"

    # 4) BREAK-EVEN tighten
    if cfg.be_trigger_pct > 0 and gain >= cfg.be_trigger_pct and px < entry:
        ok, oid = place_sell_fn(pos.pair, pos.amount, mode)
        if ok:
            _write_sell(
                pos, px, mode,
                f"BE tighten @{cfg.be_trigger_pct:.2f}% (px < entry)",
                oid, write_sell_artifacts_fn
            )
            return True, f"SOLD {pos.pair}: BREAK-EVEN tighten"
        return False, f"HOLD: break-even error {oid}"

    return False, f"HOLD: gain {gain:.2f}%"
