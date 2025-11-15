#!/usr/bin/env python3
"""
kraken_monday_main.py
---------------------------------
Kraken — 1-Coin Rotation (Monday Baseline)
with:

- SELL GUARD SG-2 (TP/SL + soft stop at -SOFT_SL_PCT)
- 1-hour stale rule (time + weak gain → SELL)
- Auto gainer rotation (pick next top candidate when flat/after SELL)

Mode:
- Single-position bot (one coin at a time).
- Uses .state/positions.json to track the open position.
- Uses public quotes via trader.crypto_engine for pricing.
- DRY_RUN = "ON" (default) simulates trades.
- DRY_RUN = "OFF" reserved for future live-trading wiring.

Environment variables (strings):
- BUY_USD       : notional USD per buy when flat   (e.g., "20")
- TP_PCT        : take-profit percent              (e.g., "8")
- SL_PCT        : hard stop-loss percent           (e.g., "2")
- SOFT_SL_PCT   : SG-2 soft stop percent           (e.g., "1.0" for -1%)
- SLOW_GAIN_REQ : stale-rule required gain %       (e.g., "3.0" for +3%)
- STALE_MINUTES : stale-rule age threshold (mins)  (e.g., "60")
- DRY_RUN       : "ON" (default) or "OFF"          ("OFF" reserved for live)
- UNIVERSE_PICK : optional symbol override (e.g., "LSK/USD")

Outputs:
- .state/positions.json: current simulated position (or absent if flat)
- .state/run_summary.md: human-readable run summary
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from trader.crypto_engine import (
    get_public_quote,
    normalize_pair,
    load_candidates,
)

STATE_DIR = Path(".state")
POSITIONS_JSON = STATE_DIR / "positions.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Position:
    symbol: str
    units: float
    entry_price: float
    entry_time: str  # ISO 8601 string (UTC)

    @property
    def entry_dt(self) -> datetime:
        return datetime.fromisoformat(self.entry_time.replace("Z", "+00:00"))


# ----------------------------
# Helpers: file + time
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60.0


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(exist_ok=True)


def load_position() -> Optional[Position]:
    if not POSITIONS_JSON.exists():
        return None
    try:
        data = json.loads(POSITIONS_JSON.read_text())
        return Position(
            symbol=data["symbol"],
            units=float(data["units"]),
            entry_price=float(data["entry_price"]),
            entry_time=data["entry_time"],
        )
    except Exception:
        # Corrupt or unexpected file; treat as flat
        return None


def save_position(pos: Position) -> None:
    ensure_state_dir()
    POSITIONS_JSON.write_text(json.dumps(asdict(pos), indent=2))


def clear_position() -> None:
    if POSITIONS_JSON.exists():
        POSITIONS_JSON.unlink()


def write_summary(summary: Dict[str, Any]) -> None:
    """
    Write a readable run_summary.md summarizing what this run decided.
    """
    ensure_state_dir()
    lines = []
    lines.append(f"# Kraken — 1-Coin Rotation (Monday Baseline)")
    lines.append("")
    lines.append(f"**Timestamp (UTC):** {summary.get('timestamp_utc', 'unknown')}")
    lines.append(f"**Mode:** {'DRY-RUN' if summary.get('dry_run', True) else 'LIVE (reserved)'}")
    lines.append(
        f"**Sell Guard Mode:** SG-2 (TP/SL + soft stop at -{summary.get('soft_sl_pct')}%)"
    )
    lines.append("")

    lines.append("## Config")
    lines.append(f"- BUY_USD: {summary.get('BUY_USD')}")
    lines.append(f"- TP_PCT: {summary.get('TP_PCT')}%")
    lines.append(f"- SL_PCT: {summary.get('SL_PCT')}%")
    lines.append(f"- SOFT_SL_PCT: {summary.get('soft_sl_pct')}%")
    lines.append(f"- SLOW_GAIN_REQ: {summary.get('SLOW_GAIN_REQ')}%")
    lines.append(f"- STALE_MINUTES: {summary.get('STALE_MINUTES')}")
    lines.append(f"- UNIVERSE_PICK: {summary.get('UNIVERSE_PICK') or '(auto)'}")
    lines.append("")

    lines.append("## State / Decision")
    lines.append(f"- Action: {summary.get('action')}")
    lines.append(f"- Symbol: {summary.get('symbol')}")
    lines.append(f"- Current Price: {summary.get('current_price')}")
    lines.append(f"- Entry Price: {summary.get('entry_price')}")
    lines.append(f"- Position Units: {summary.get('units')}")
    lines.append(f"- Position Age (min): {summary.get('position_age_minutes')}")
    lines.append(f"- Unrealized PnL %: {summary.get('pnl_pct')}")
    lines.append(f"- Sell Reason: {summary.get('sell_reason') or '(n/a)'}")
    lines.append(f"- Next Symbol: {summary.get('next_symbol')}")
    lines.append(f"- Next Entry Price: {summary.get('next_entry_price')}")
    lines.append(f"- Next Units: {summary.get('next_units')}")
    lines.append("")
    lines.append(f"Notes: {summary.get('notes') or ''}".strip())

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


# ----------------------------
# Market helpers
# ----------------------------

def choose_symbol(universe_pick: Optional[str]) -> str:
    """
    Choose which symbol to trade when flat.
    Priority:
    1) UNIVERSE_PICK if provided
    2) Top candidate from .state/momentum_candidates.csv (by rank)
    3) Fallback to BTC/USD
    """
    if universe_pick:
        return normalize_pair(universe_pick)

    # Try momentum candidates if available
    try:
        candidates = load_candidates()
        if candidates:
            # Expect fields: symbol, quote, rank — pick max rank as "best"
            def rank_val(row: Dict[str, str]) -> float:
                try:
                    return float(row.get("rank", 0))
                except Exception:
                    return 0.0

            best = max(candidates, key=rank_val)
            sym = best.get("symbol") or "BTC/USD"
            return normalize_pair(sym)
    except Exception:
        pass

    # Ultimate fallback
    return normalize_pair("BTC/USD")


def get_price(symbol: str) -> Optional[float]:
    """
    Unified price getter using trader.crypto_engine public quote.
    """
    try:
        return get_public_quote(symbol)
    except Exception:
        return None


# ----------------------------
# Core decision logic
# ----------------------------

def decide_and_act() -> None:
    ensure_state_dir()

    # ----- Read config from environment -----
    BUY_USD = float(os.getenv("BUY_USD", "20"))
    TP_PCT = float(os.getenv("TP_PCT", "8"))
    SL_PCT = float(os.getenv("SL_PCT", "2"))
    SOFT_SL_PCT = float(os.getenv("SOFT_SL_PCT", "1.0"))  # SG-2 soft stop (default 1%)
    SLOW_GAIN_REQ = float(os.getenv("SLOW_GAIN_REQ", "3.0"))  # stale rule gain requirement
    STALE_MINUTES = float(os.getenv("STALE_MINUTES", "60"))   # stale rule age in minutes
    UNIVERSE_PICK = os.getenv("UNIVERSE_PICK", "").strip() or None
    DRY_RUN_FLAG = os.getenv("DRY_RUN", "ON").strip().upper()
    DRY_RUN = DRY_RUN_FLAG != "OFF"  # anything except "OFF" is treated as DRY

    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    pos = load_position()

    # Determine symbol: if we already hold something, we guard that symbol.
    if pos:
        symbol = normalize_pair(pos.symbol)
    else:
        symbol = choose_symbol(UNIVERSE_PICK)

    price = get_price(symbol)

    base_summary: Dict[str, Any] = {
        "timestamp_utc": now_iso,
        "dry_run": DRY_RUN,
        "BUY_USD": BUY_USD,
        "TP_PCT": TP_PCT,
        "SL_PCT": SL_PCT,
        "soft_sl_pct": SOFT_SL_PCT,
        "SLOW_GAIN_REQ": SLOW_GAIN_REQ,
        "STALE_MINUTES": STALE_MINUTES,
        "UNIVERSE_PICK": UNIVERSE_PICK,
        "symbol": symbol,
        "current_price": price,
        "entry_price": None,
        "units": None,
        "pnl_pct": None,
        "position_age_minutes": None,
        "sell_reason": None,
        "action": "HOLD",
        "next_symbol": None,
        "next_entry_price": None,
        "next_units": None,
        "notes": "",
    }

    # Handle missing/invalid price
    if price is None or price <= 0:
        base_summary["notes"] = "HOLD: invalid or missing price feed for symbol."
        write_summary(base_summary)
        return

    # If we already have a position: apply SELL GUARD + stale rule
    if pos is not None:
        entry = float(pos.entry_price)
        units = float(pos.units)
        pnl_pct = ((price - entry) / entry) * 100.0
        age_minutes = minutes_between(pos.entry_dt, now_utc)

        base_summary["entry_price"] = entry
        base_summary["units"] = units
        base_summary["pnl_pct"] = round(pnl_pct, 4)
        base_summary["position_age_minutes"] = round(age_minutes, 2)

        sell_reason: Optional[str] = None
        notes = []

        # TP condition
        if pnl_pct >= TP_PCT:
            sell_reason = "TP"
            notes.append(f"TP hit: PnL {pnl_pct:.3f}% >= {TP_PCT}%")

        # Hard SL condition
        elif pnl_pct <= -SL_PCT:
            sell_reason = "SL"
            notes.append(f"SL hit: PnL {pnl_pct:.3f}% <= -{SL_PCT}%")

        # Soft SL (SG-2) condition
        elif pnl_pct <= -SOFT_SL_PCT:
            sell_reason = "DIP"
            notes.append(f"Soft stop (SG-2): PnL {pnl_pct:.3f}% <= -{SOFT_SL_PCT}%")

        # Stale rule: age threshold + weak gain
        elif age_minutes >= STALE_MINUTES and pnl_pct < SLOW_GAIN_REQ:
            sell_reason = "STALE"
            notes.append(
                f"Stale rule: Age {age_minutes:.1f} min >= {STALE_MINUTES} min "
                f"and PnL {pnl_pct:.3f}% < {SLOW_GAIN_REQ}%."
            )

        # Decide action based on sell_reason
        if sell_reason is None:
            # HOLD position
            base_summary["action"] = "HOLD"
            base_summary["sell_reason"] = None
            notes.append("SELL GUARD: HOLD (no TP/SL/soft-stop/stale hit).")
            base_summary["notes"] = " ".join(notes)
            write_summary(base_summary)
            return

        # We are SELLING this run (and maybe rotating)
        base_summary["action"] = "SELL"
        base_summary["sell_reason"] = sell_reason

        # Simulated execution of SELL
        if DRY_RUN:
            clear_position()
            notes.append(f"DRY-RUN: would SELL {units} {symbol} at {price}.")
        else:
            # Placeholder for future live-trading integration
            clear_position()
            notes.append(
                "LIVE wiring not implemented yet; treated as DRY SELL (position cleared in .state)."
            )

        # Auto gainer rotation (only if UNIVERSE_PICK is not forcing a symbol)
        next_symbol = choose_symbol(UNIVERSE_PICK)
        next_price: Optional[float] = None
        next_units: Optional[float] = None

        if next_symbol:
            next_price = get_price(next_symbol)
            if next_price is not None and next_price > 0:
                next_units = round(BUY_USD / next_price, 8)
                new_pos = Position(
                    symbol=next_symbol,
                    units=next_units,
                    entry_price=next_price,
                    entry_time=now_iso,
                )
                save_position(new_pos)
                base_summary["action"] = "SELL+BUY"
                base_summary["next_symbol"] = next_symbol
                base_summary["next_entry_price"] = next_price
                base_summary["next_units"] = next_units
                if DRY_RUN:
                    notes.append(
                        f"Rotation: DRY-RUN would BUY ~{next_units} {next_symbol} "
                        f"at {next_price} using ${BUY_USD}."
                    )
                else:
                    notes.append(
                        "Rotation: LIVE wiring not implemented yet; "
                        "position stored in .state as if bought."
                    )
            else:
                notes.append(
                    f"Rotation skipped: no valid price for next symbol candidate ({next_symbol})."
                )
        else:
            notes.append("Rotation skipped: no symbol available from universe.")

        base_summary["notes"] = " ".join(notes)
        write_summary(base_summary)
        return

    # If we are FLAT: BUY a new position (auto gainer selection)
    units = round(BUY_USD / price, 8)
    new_pos = Position(
        symbol=symbol,
        units=units,
        entry_price=price,
        entry_time=now_iso,
    )

    base_summary["entry_price"] = price
    base_summary["units"] = units
    base_summary["pnl_pct"] = 0.0  # new entry
    base_summary["position_age_minutes"] = 0.0
    base_summary["action"] = "BUY"
    base_summary["sell_reason"] = None

    notes = [f"Flat before run; opening new position in {symbol}."]
    if DRY_RUN:
        save_position(new_pos)
        notes.append(f"DRY-RUN: would BUY ~{units} {symbol} at {price} using ${BUY_USD}.")
    else:
        # Placeholder for future live buy logic.
        save_position(new_pos)
        notes.append(
            "LIVE wiring not implemented yet; position stored in .state as if bought."
        )

    base_summary["notes"] = " ".join(notes)
    write_summary(base_summary)


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    decide_and_act()
