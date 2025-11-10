#!/usr/bin/env python3
"""
sell_guard.py
Simple, explicit sell-guard used by the rotation bot.

Rules (tunables via env in your workflow/vars):
- Hard stop-loss SL_PCT (e.g., 1.0 means sell if drop >= 1.0 % from entry).
- Take-profit TP_PCT (e.g., 5.0 means sell if gain >= 5.0 % from entry).
- Slow-window rule: if gain < SLOW_GAIN_PCT within SLOW_WINDOW_MIN minutes since entry, sell.

Exports:
- evaluate_sell(entry_price: float,
                current_price: float,
                minutes_since_entry: float,
                tp_pct: float,
                sl_pct: float,
                slow_gain_pct: float,
                slow_window_min: float) -> dict
  Returns: {"action": "HOLD"|"SELL", "reason": str, "pnl_pct": float}
"""

from __future__ import annotations

def _pct_change(now: float, entry: float) -> float:
    return (now - entry) / entry * 100.0


def evaluate_sell(
    entry_price: float,
    current_price: float,
    minutes_since_entry: float,
    tp_pct: float = 5.0,
    sl_pct: float = 1.0,
    slow_gain_pct: float = 3.0,
    slow_window_min: float = 60.0,
) -> dict:
    if entry_price <= 0 or current_price <= 0:
        return {"action": "HOLD", "reason": "invalid price(s)", "pnl_pct": 0.0}

    pnl = _pct_change(current_price, entry_price)

    # 1) Hard stop first
    if pnl <= -abs(sl_pct):
        return {"action": "SELL", "reason": f"STOP {sl_pct:.2f}%", "pnl_pct": pnl}

    # 2) Take-profit
    if pnl >= abs(tp_pct):
        return {"action": "SELL", "reason": f"TP {tp_pct:.2f}%", "pnl_pct": pnl}

    # 3) Slow-window rule only if within the first window
    if minutes_since_entry <= max(1.0, float(slow_window_min)):
        if pnl < abs(slow_gain_pct):
            return {"action": "SELL", "reason": f"SLOW < {slow_gain_pct:.2f}% in {slow_window_min:.0f}m", "pnl_pct": pnl}

    return {"action": "HOLD", "reason": "guard ok", "pnl_pct": pnl}
