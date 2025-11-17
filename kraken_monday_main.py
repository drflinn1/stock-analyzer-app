#!/usr/bin/env python3
"""
kraken_monday_main.py

Kraken — 1-Coin Rotation (Monday Baseline)

Features (Nov 2025 baseline):
- Single-position rotation bot.
- Uses .state/momentum_candidates.csv for real Kraken 24h gainers.
- SG-2 Sell Guard:
  * Take-profit (TP_PCT)
  * Hard stop-loss (SL_PCT)
  * Soft stop at SOFT_SL_PCT (e.g. -1%)
- Stale rule:
  * If position age (minutes) >= STALE_MINUTES
  * AND unrealized PnL % < SLOW_GAIN_REQ (e.g. 3%)
    → SELL + rotate into best gainer.

- Writes:
  * .state/run_summary.md (human summary)
  * .state/positions.json (simulated current position) — DRY-RUN ONLY

LIVE behaviour:
- If DRY_RUN == 'OFF':
  * Uses tools/kraken_live_trade.place_market_order to send real MARKET orders.
  * NEVER overwrites positions.json — that file is owned by
    tools/kraken_position_sync.py (live balance sync).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tools.kraken_live_trade import KrakenLiveError, place_market_order

STATE_DIR = Path(".state")
POS_PATH = STATE_DIR / "positions.json"
CANDIDATES_CSV = STATE_DIR / "momentum_candidates.csv"


# ---------- Helpers & models ----------


def utc_now_iso() -> str:
    """Return current UTC time in ISO8601 Z form, seconds precision."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_iso_z(s: str) -> datetime:
    """Parse an ISO string with optional trailing Z into aware UTC datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


@dataclass
class Position:
    symbol: str
    units: float
    entry_price: float
    entry_time: str  # ISO string

    @property
    def age_minutes(self) -> float:
        try:
            t = parse_iso_z(self.entry_time)
            delta = datetime.now(timezone.utc) - t
            return delta.total_seconds() / 60.0
        except Exception:
            return 0.0


@dataclass
class Candidate:
    symbol: str
    quote: float
    rank: float


def load_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def load_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(exist_ok=True)


# ---------- State I/O ----------


def load_position() -> Optional[Position]:
    if not POS_PATH.exists():
        return None
    try:
        data = json.loads(POS_PATH.read_text())
        return Position(
            symbol=data["symbol"],
            units=float(data["units"]),
            entry_price=float(data["entry_price"]),
            entry_time=str(data["entry_time"]),
        )
    except Exception:
        return None


def save_position(pos: Optional[Position]) -> None:
    """DRY-RUN ONLY: persist simulated position."""
    if pos is None:
        try:
            POS_PATH.unlink()
        except FileNotFoundError:
            pass
        return

    data = {
        "symbol": pos.symbol,
        "units": pos.units,
        "entry_price": pos.entry_price,
        "entry_time": pos.entry_time,
    }
    POS_PATH.write_text(json.dumps(data, indent=2))


def load_candidates() -> List[Candidate]:
    """Load momentum candidates from CSV; fall back to empty list if missing."""
    if not CANDIDATES_CSV.exists():
        return []
    out: List[Candidate] = []
    try:
        with CANDIDATES_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sym = row.get("symbol") or row.get("pair") or ""
                    quote = float(row.get("quote", "0") or 0.0)
                    rank = float(row.get("rank", "0") or 0.0)
                    if sym and quote > 0:
                        out.append(Candidate(symbol=sym.strip(), quote=quote, rank=rank))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def price_map_from_candidates(cands: List[Candidate]) -> Dict[str, float]:
    return {c.symbol: c.quote for c in cands}


def choose_best_candidate(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    # Higher rank is better
    return sorted(cands, key=lambda c: c.rank, reverse=True)[0]


def choose_rotation_target(
    cands: List[Candidate],
    universe_pick: str,
    current_symbol: Optional[str] = None,
) -> Optional[Candidate]:
    """Choose next symbol to rotate into."""
    if universe_pick:
        # Universe override wins; try to find that symbol in candidates.
        for c in cands:
            if c.symbol == universe_pick:
                return c
        # If not found, still synthesize a candidate with quote from map if available.
        pm = price_map_from_candidates(cands)
        q = pm.get(universe_pick)
        if q:
            return Candidate(symbol=universe_pick, quote=q, rank=0.0)
        return None

    # Auto-gainer mode: choose highest-rank candidate, preferably not the current symbol.
    if not cands:
        return None
    sorted_cands = sorted(cands, key=lambda c: c.rank, reverse=True)
    for c in sorted_cands:
        if current_symbol is None or c.symbol != current_symbol:
            return c
    # Fallback: just return top-ranked candidate
    return sorted_cands[0]


# ---------- Core decision logic ----------


def main() -> None:
    ensure_state_dir()

    # Environment / config
    buy_usd = load_env_float("BUY_USD", 20.0)
    tp_pct = load_env_float("TP_PCT", 8.0)
    sl_pct = load_env_float("SL_PCT", 2.0)
    soft_sl_pct = load_env_float("SOFT_SL_PCT", 1.0)
    slow_gain_req = load_env_float("SLOW_GAIN_REQ", 3.0)
    stale_minutes = load_env_float("STALE_MINUTES", 60.0)

    dry_run_flag = load_env_str("DRY_RUN", "ON").upper() != "OFF"
    mode_str = "DRY-RUN" if dry_run_flag else "LIVE"

    universe_pick_raw = load_env_str("UNIVERSE_PICK", "").strip()
    universe_pick = universe_pick_raw if universe_pick_raw else ""

    timestamp = utc_now_iso()

    # Load state + candidates
    position = load_position()
    candidates = load_candidates()
    price_map = price_map_from_candidates(candidates)

    sell_guard_mode = f"SG-2 (TP/SL + soft stop at -{soft_sl_pct:.1f}%)"
    universe_display = universe_pick if universe_pick else "(auto)"

    # Decision variables to report
    action = "HOLD"
    symbol = position.symbol if position else None
    current_price: Optional[float] = None
    entry_price = position.entry_price if position else None
    pos_units = position.units if position else 0.0
    pos_age_min = position.age_minutes if position else 0.0
    pnl_pct = 0.0
    sell_reason = "(n/a)"
    next_symbol: Optional[str] = None
    next_entry_price: Optional[float] = None
    next_units: Optional[float] = None
    notes: str = ""

    # --------- Price lookup helpers ---------

    def get_price(pair: str) -> Optional[float]:
        p = price_map.get(pair)
        if p is not None:
            return p
        # As a fallback, just return None; public API quote is handled elsewhere.
        return None

    # --------- Case 1: FLAT → BUY best gainer ---------
    if position is None:
        target = choose_rotation_target(candidates, universe_pick, current_symbol=None)
        if target is None:
            action = "HOLD"
            symbol = None
            notes = "No momentum candidates available; staying flat."
        else:
            symbol = target.symbol
            current_price = target.quote
            entry_price = current_price
            pos_units = buy_usd / current_price if current_price > 0 else 0.0
            pos_age_min = 0.0
            sell_reason = "(n/a)"
            next_symbol = "None"
            next_entry_price = None
            next_units = None

            if dry_run_flag:
                action = "BUY"
                notes = (
                    f"Flat before run; opening new position in {symbol}. "
                    f"{mode_str}: would BUY ~{pos_units:.6f} {symbol} at "
                    f"{current_price:.6f} using ${buy_usd:.2f}."
                )
                # DRY-RUN: update simulated position
                new_pos = Position(
                    symbol=symbol,
                    units=pos_units,
                    entry_price=entry_price,
                    entry_time=timestamp,
                )
                save_position(new_pos)
            else:
                # LIVE: send market BUY to Kraken, do NOT touch positions.json
                try:
                    result = place_market_order(symbol, "buy", pos_units)
                    txids = result.get("txid") or result.get("descr") or ""
                    action = "BUY"
                    notes = (
                        f"Flat before run; LIVE: SENT MARKET BUY ~{pos_units:.6f} "
                        f"{symbol} at {current_price:.6f} using ${buy_usd:.2f}. "
                        f"Kraken result: {txids}"
                    )
                except KrakenLiveError as e:
                    action = "HOLD"
                    notes = (
                        f"LIVE ERROR: failed to place BUY order for {symbol}: {e}. "
                        "Leaving on-exchange balances unchanged."
                    )

    # --------- Case 2: Already in a position ---------
    else:
        symbol = position.symbol
        entry_price = float(position.entry_price)
        pos_units = float(position.units)
        pos_age_min = position.age_minutes
        current_price = get_price(symbol)

        if current_price is None or current_price <= 0:
            # No valid price → hold and log that sell guard is disabled
            action = "HOLD"
            pnl_pct = 0.0
            sell_reason = "(n/a)"
            notes = (
                "SELL GUARD: HOLD (invalid or missing price; "
                "no TP/SL/soft-stop checks)."
            )
        else:
            pnl_pct = (current_price - entry_price) / entry_price * 100.0

            # ---- SG-2 Sell Guard ----
            guard_reason: Optional[str] = None
            if pnl_pct >= tp_pct:
                guard_reason = "TP"
            elif pnl_pct <= -sl_pct:
                guard_reason = "SL"
            elif pnl_pct <= -soft_sl_pct:
                guard_reason = "SOFT_STOP"

            # ---- Stale Rule ----
            stale_triggered = False
            if stale_minutes > 0:
                if pos_age_min >= stale_minutes and pnl_pct < slow_gain_req:
                    stale_triggered = True

            # Decide final action priority:
            # 1) Sell Guard (TP / SL / soft) if hit.
            # 2) Stale rule.
            # 3) Otherwise HOLD.

            if guard_reason is not None:
                sell_reason = guard_reason
                target = choose_rotation_target(
                    candidates, universe_pick, current_symbol=symbol
                )

                if dry_run_flag:
                    if target is None:
                        action = "SELL"
                        next_symbol = "None"
                        next_entry_price = None
                        next_units = None
                        notes = (
                            f"SELL GUARD: {guard_reason}. {mode_str}: would SELL "
                            f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                            "No rotation target available; would stay flat."
                        )
                        save_position(None)
                    else:
                        action = "SELL+BUY"
                        next_symbol = target.symbol
                        next_entry_price = target.quote
                        next_units = (
                            buy_usd / next_entry_price if next_entry_price > 0 else 0.0
                        )
                        notes = (
                            f"SELL GUARD: {guard_reason}. {mode_str}: would SELL "
                            f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                            f"Rotation: {mode_str} would BUY ~{next_units:.6f} "
                            f"{next_symbol} at {next_entry_price:.6f} "
                            f"using ${buy_usd:.2f}."
                        )
                        new_pos = Position(
                            symbol=next_symbol,
                            units=next_units,
                            entry_price=next_entry_price,
                            entry_time=timestamp,
                        )
                        save_position(new_pos)
                else:
                    # LIVE branch: send real SELL (and BUY if target exists)
                    try:
                        # First, SELL current position
                        sell_result = place_market_order(symbol, "sell", pos_units)
                        if target is None:
                            action = "SELL"
                            next_symbol = "None"
                            next_entry_price = None
                            next_units = None
                            notes = (
                                f"SELL GUARD: {guard_reason}. LIVE: SENT MARKET SELL "
                                f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                                f"Kraken sell result: {sell_result}. "
                                "No rotation target available; staying flat."
                            )
                        else:
                            # Then BUY rotation target
                            next_symbol = target.symbol
                            next_entry_price = target.quote
                            next_units = (
                                buy_usd / next_entry_price
                                if next_entry_price > 0
                                else 0.0
                            )
                            buy_result = place_market_order(
                                next_symbol, "buy", next_units
                            )
                            action = "SELL+BUY"
                            notes = (
                                f"SELL GUARD: {guard_reason}. LIVE: SENT MARKET SELL "
                                f"{pos_units:.6f} {symbol} at {current_price:.6f} "
                                f"and MARKET BUY ~{next_units:.6f} {next_symbol} "
                                f"at {next_entry_price:.6f} using ${buy_usd:.2f}. "
                                f"Kraken results: sell={sell_result}, "
                                f"buy={buy_result}."
                            )
                    except KrakenLiveError as e:
                        action = "HOLD"
                        notes = (
                            f"LIVE ERROR: failed to execute SELL/rotation for "
                            f"{symbol}: {e}. Leaving on-exchange balances unchanged."
                        )

            elif stale_triggered:
                sell_reason = "STALE"
                target = choose_rotation_target(
                    candidates, universe_pick, current_symbol=symbol
                )

                if dry_run_flag:
                    if target is None:
                        action = "SELL"
                        next_symbol = "None"
                        next_entry_price = None
                        next_units = None
                        notes = (
                            f"Stale rule: Age {pos_age_min:.1f} min >= "
                            f"{stale_minutes:.1f} min and PnL {pnl_pct:.3f}% "
                            f"< {slow_gain_req:.1f}%. {mode_str}: would SELL "
                            f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                            "No rotation target available; would stay flat."
                        )
                        save_position(None)
                    else:
                        action = "SELL+BUY"
                        next_symbol = target.symbol
                        next_entry_price = target.quote
                        next_units = (
                            buy_usd / next_entry_price if next_entry_price > 0 else 0.0
                        )
                        notes = (
                            f"Stale rule: Age {pos_age_min:.1f} min >= "
                            f"{stale_minutes:.1f} min and PnL {pnl_pct:.3f}% "
                            f"< {slow_gain_req:.1f}%. {mode_str}: would SELL "
                            f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                            f"Rotation: {mode_str} would BUY ~{next_units:.6f} "
                            f"{next_symbol} at {next_entry_price:.6f} "
                            f"using ${buy_usd:.2f}."
                        )
                        new_pos = Position(
                            symbol=next_symbol,
                            units=next_units,
                            entry_price=next_entry_price,
                            entry_time=timestamp,
                        )
                        save_position(new_pos)
                else:
                    # LIVE stale rule
                    try:
                        sell_result = place_market_order(symbol, "sell", pos_units)
                        if target is None:
                            action = "SELL"
                            next_symbol = "None"
                            next_entry_price = None
                            next_units = None
                            notes = (
                                f"Stale rule: Age {pos_age_min:.1f} min >= "
                                f"{stale_minutes:.1f} min and PnL {pnl_pct:.3f}% "
                                f"< {slow_gain_req:.1f}%. LIVE: SENT MARKET SELL "
                                f"{pos_units:.6f} {symbol} at {current_price:.6f}. "
                                f"Kraken sell result: {sell_result}. "
                                "No rotation target available; staying flat."
                            )
                        else:
                            next_symbol = target.symbol
                            next_entry_price = target.quote
                            next_units = (
                                buy_usd / next_entry_price
                                if next_entry_price > 0
                                else 0.0
                            )
                            buy_result = place_market_order(
                                next_symbol, "buy", next_units
                            )
                            action = "SELL+BUY"
                            notes = (
                                f"Stale rule: Age {pos_age_min:.1f} min >= "
                                f"{stale_minutes:.1f} min and PnL {pnl_pct:.3f}% "
                                f"< {slow_gain_req:.1f}%. LIVE: SENT MARKET SELL "
                                f"{pos_units:.6f} {symbol} at {current_price:.6f} "
                                f"and MARKET BUY ~{next_units:.6f} {next_symbol} "
                                f"at {next_entry_price:.6f} using ${buy_usd:.2f}. "
                                f"Kraken results: sell={sell_result}, "
                                f"buy={buy_result}."
                            )
                    except KrakenLiveError as e:
                        action = "HOLD"
                        notes = (
                            f"LIVE ERROR: failed to execute stale SELL/rotation for "
                            f"{symbol}: {e}. Leaving on-exchange balances unchanged."
                        )

            else:
                action = "HOLD"
                sell_reason = "(n/a)"
                notes = (
                    "SELL GUARD: HOLD (no TP/SL/soft-stop hit, "
                    "no stale rule trigger)."
                )

    # ---------- Write run_summary.md ----------

    run_lines: List[str] = []
    run_lines.append("Kraken — 1-Coin Rotation (Monday Baseline)")
    run_lines.append("")
    run_lines.append(f"Timestamp (UTC): {timestamp}")
    run_lines.append(f"Mode: {mode_str}")
    run_lines.append(f"Sell Guard Mode: {sell_guard_mode}")
    run_lines.append("")
    run_lines.append("Config")
    run_lines.append("")
    run_lines.append(f"- BUY_USD: {buy_usd:.1f}")
    run_lines.append(f"- TP_PCT: {tp_pct:.1f}%")
    run_lines.append(f"- SL_PCT: {sl_pct:.1f}%")
    run_lines.append(f"- SOFT_SL_PCT: {soft_sl_pct:.1f}%")
    run_lines.append(f"- SLOW_GAIN_REQ: {slow_gain_req:.1f}%")
    run_lines.append(f"- STALE_MINUTES: {stale_minutes:.1f}")
    run_lines.append(f"- UNIVERSE_PICK: {universe_display}")
    run_lines.append("")
    run_lines.append("State / Decision")
    run_lines.append("")
    run_lines.append(f"- Action: {action}")
    run_lines.append(f"- Symbol: {symbol or 'None'}")
    run_lines.append(
        f"- Current Price: {current_price:.6f}"
        if current_price
        else "- Current Price: (n/a)"
    )
    run_lines.append(
        f"- Entry Price: {entry_price:.6f}"
        if entry_price
        else "- Entry Price: (n/a)"
    )
    run_lines.append(f"- Position Units: {pos_units}")
    run_lines.append(f"- Position Age (min): {pos_age_min:.2f}")
    run_lines.append(f"- Unrealized PnL %: {pnl_pct:.3f}")
    run_lines.append(f"- Sell Reason: {sell_reason}")
    run_lines.append(
        f"- Next Symbol: {next_symbol if next_symbol is not None else 'None'}"
    )
    run_lines.append(
        f"- Next Entry Price: {next_entry_price:.6f}"
        if next_entry_price is not None
        else "- Next Entry Price: None"
    )
    run_lines.append(
        f"- Next Units: {next_units}"
        if next_units is not None
        else "- Next Units: None"
    )
    run_lines.append(f"Notes: {notes}")

    run_summary_path = STATE_DIR / "run_summary.md"
    run_summary_path.write_text("\n".join(run_lines))


if __name__ == "__main__":
    main()
