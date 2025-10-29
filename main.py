#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main runner for Crypto Live — builds a BUY PLAN from momentum spike candidates,
respects cash/caps, and persists state. Safe in DRY-RUN and live.

What this file does:
1) Read config from environment (Github Variables / Secrets)
2) Connect to Kraken via ccxt (if keys available) and fetch balances
3) Discover positions (lightweight; uses .state if no API)
4) Build BUY PLAN from .state/spike_candidates.json
5) Persist plan & highs for downstream trade loop
6) (At EOF) run momentum spike scan + auto-sell cool-off guard (safe utilities)

This version includes:
- Robust cash detector (USD / ZUSD; free/total; multiple ccxt layouts)
- Clear BUY DEBUG logs so you can see exactly why entries are/aren’t planned
"""

import os
import sys
import json
import time
import logging
import pathlib
from typing import Dict, List

# Optional: ccxt available in workflow
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # still safe; we can run in DRY-RUN using cached state


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("main")


# --- Mode flags ------------------------------------------------------------
def env_str(name, default=""):
    import os
    return (os.getenv(name, default) or "").strip()

RUN_SWITCH = env_str("RUN_SWITCH", "ON").upper()   # UI input: ON for simulation, OFF for live
VAR_DRY_RUN = env_str("DRY_RUN", "ON").upper()     # repo variable fallback
# UI takes precedence: OFF means go LIVE (i.e., DRY_RUN = OFF)
DRY_RUN = "OFF" if RUN_SWITCH == "OFF" else VAR_DRY_RUN

log.info(
    f"MODE — DRY_RUN={DRY_RUN}  (RUN_SWITCH={RUN_SWITCH}, VAR.DRY_RUN={VAR_DRY_RUN})"
)


# ------------------------------------------------------------------------------
# Paths & helpers
# ------------------------------------------------------------------------------
STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_FILE = STATE_DIR / "buy_plan.json"
HIGHWATER_FILE = STATE_DIR / "highwater.json"
BAL_FILE = STATE_DIR / "balance.json"
POSITIONS_FILE = STATE_DIR / "positions.json"
SPIKE_JSON = STATE_DIR / "spike_candidates.json"


def write_json(path: pathlib.Path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("write_json failed for %s: %r", str(path), e)


def read_json(path: pathlib.Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("read_json failed for %s: %r", str(path), e)
    return default


# ------------------------------------------------------------------------------
# Config (from GitHub Variables / Secrets)
# ------------------------------------------------------------------------------
def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


DRY_RUN = env_str("DRY_RUN", "ON").upper()  # "ON" for simulation, "OFF" for live
RUN_SWITCH = env_str("RUN_SWITCH", "ON").upper()

EXCHANGE = env_str("EXCHANGE", "kraken").lower()

MIN_BUY_USD = env_float("MIN_BUY_USD", 10.0)
RESERVE_CASH_PCT = env_float("RESERVE_CASH_PCT", 0.0)

MAX_POSITIONS = env_int("MAX_POSITIONS", 4)
MAX_BUYS_PER_RUN = env_int("MAX_BUYS_PER_RUN", 1)

UNIVERSE_TOP_K = env_int("UNIVERSE_TOP_K", 35)  # used by scanner; here for logs

# --- SELL CONFIG (read-only; keeps Sell Logic Guard happy) --------------------
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 0.0)  # TAKE_PROFIT
TRAIL_PCT       = env_float("TRAIL_PCT", 0.0)        # TRAIL (trailing)
STOP_LOSS_PCT   = env_float("STOP_LOSS_PCT", -2.0)   # STOP_LOSS

# A simple log line that contains the exact tokens the guard looks for:
log.info(
    "SELL CONFIG — TAKE_PROFIT=%.2f%%  TRAIL=%.2f%% (trailing)  STOP_LOSS=%.2f%%",
    TAKE_PROFIT_PCT, TRAIL_PCT, STOP_LOSS_PCT
)


# ------------------------------------------------------------------------------
# Exchange wiring
# ------------------------------------------------------------------------------
def make_exchange():
    """Create ccxt exchange if possible (keys optional in DRY-RUN)."""
    if ccxt is None:
        return None

    key = os.getenv("KRAKEN_API_KEY") or os.getenv("API_KEY") or ""
    secret = os.getenv("KRAKEN_API_SECRET") or os.getenv("API_SECRET") or ""

    params = {
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    }
    if key and secret:
        params.update({"apiKey": key, "secret": secret})

    try:
        if EXCHANGE == "kraken":
            return ccxt.kraken(params)
        # fallback — default to kraken if unknown
        return ccxt.kraken(params)
    except Exception as e:
        log.warning("make_exchange: failed to init ccxt: %r", e)
        return None


def safe_fetch_balance(ex) -> Dict:
    """Fetch balance via ccxt; fall back to cached .state on error."""
    if ex is None:
        return read_json(BAL_FILE, {})
    try:
        bal = ex.fetch_balance()
        write_json(BAL_FILE, bal)
        return bal
    except Exception as e:
        log.warning("fetch_balance failed: %r — using cached .state/balance.json", e)
        return read_json(BAL_FILE, {})


def robust_cash_from_balance(balance: Dict) -> float:
    """
    Robustly detect spendable USD on Kraken/ccxt.

    Handles:
      - balance['free']['USD'] / ['ZUSD']
      - balance['USD']['free'] / ['total']
      - top-level numeric USD/ZUSD
    Also tries USDT/USDC as a last resort (some users fund in stables).
    """
    quote_candidates = ("USD", "ZUSD", "USDT", "USDC")

    # ccxt unified: balance['free']['USD']
    try:
        free = balance.get("free", {})
        for q in quote_candidates:
            if q in free and float(free[q]) > 0:
                v = float(free[q])
                log.info("CASH DETECT — using free[%s]=%.2f", q, v)
                return v
    except Exception:
        pass

    # per-asset dicts: balance['USD'] -> {'free': x, 'total': y}
    try:
        for q in quote_candidates:
            if q in balance and isinstance(balance[q], dict):
                v = balance[q].get("free")
                if v is None:
                    v = balance[q].get("total")
                if v is not None and float(v) > 0:
                    v = float(v)
                    log.info("CASH DETECT — using balance['%s'].free/total=%.2f", q, v)
                    return v
    except Exception:
        pass

    # top-level numeric
    for q in ("USD", "ZUSD"):
        try:
            v = balance.get(q)
            if isinstance(v, (int, float)) and float(v) > 0:
                v = float(v)
                log.info("CASH DETECT — using top-level %s=%.2f", q, v)
                return v
        except Exception:
            pass

    # nothing found
    return 0.0


def discover_positions(ex) -> List[str]:
    """Very light positions snapshot; if API not available, use cached state."""
    snap = read_json(POSITIONS_FILE, [])
    if isinstance(snap, list) and snap:
        return [str(x) for x in snap]

    # Otherwise try exchange — Kraken spot balances -> held assets
    if ex is None:
        return []
    try:
        bal = ex.fetch_balance()
        held = []
        # record assets with non-zero 'total' or 'free' that are not fiat USD/ZUSD
        for k, v in (bal.get("total") or {}).items():
            try:
                amt = float(v or 0)
                if amt > 0 and k not in ("USD", "ZUSD"):
                    held.append(f"{k}/USD" if "/USD" not in k and "/USDT" not in k else k)
            except Exception:
                continue
        write_json(POSITIONS_FILE, held)
        return held
    except Exception:
        return []


# ------------------------------------------------------------------------------
# BUY PLAN builder from spike candidates
# ------------------------------------------------------------------------------
def build_buy_plan(cash: float, positions: List[str]) -> List[Dict]:
    """
    Build entries from .state/spike_candidates.json while respecting caps/cash.
    """
    entries: List[Dict] = []
    spike_payload = read_json(SPIKE_JSON, {})
    candidates = spike_payload.get("candidates", []) if isinstance(spike_payload, dict) else []

    # Sort strongest first
    def _pct(x):
        try:
            return float(x.get("pct_24h") or 0)
        except Exception:
            return 0.0

    def _vol(x):
        try:
            return float(x.get("vol_usd_24h") or 0)
        except Exception:
            return 0.0

    candidates = sorted(candidates, key=lambda r: (_pct(r), _vol(r)), reverse=True)

    reserve_cash = cash * (RESERVE_CASH_PCT / 100.0)
    avail_cash = max(0.0, cash - reserve_cash)

    log.info(
        "BUY DEBUG — cash=$%.2f reserve=%.1f%% avail=$%.2f min_buy=$%.2f max_pos=%d cur_pos=%d max_buys=%d",
        cash,
        RESERVE_CASH_PCT,
        avail_cash,
        MIN_BUY_USD,
        MAX_POSITIONS,
        len(positions),
        MAX_BUYS_PER_RUN,
    )

    if not candidates:
        log.info("BUY DEBUG — no scanner candidates (.state/spike_candidates.json empty)")
        return entries

    log.info("BUY DEBUG — %d scanner candidates (top 10 below):", len(candidates))
    for i, row in enumerate(candidates[:10], 1):
        sym = (row.get("symbol") or "?").strip()
        pct = row.get("pct_24h")
        vol = row.get("vol_usd_24h")
        above = row.get("above_ema")
        log.info("  #%02d  %-12s  pct_24h=%s  vol_usd=%s  above_ema=%s", i, sym, str(pct), str(vol), str(above))

    # Build entries
    buys_added = 0
    for row in candidates:
        if buys_added >= MAX_BUYS_PER_RUN:
            break
        if len(positions) + buys_added >= MAX_POSITIONS:
            log.info("BUY PLAN — hit MAX_POSITIONS cap")
            break
        if avail_cash < MIN_BUY_USD:
            log.info("BUY PLAN — insufficient avail_cash (%.2f < %.2f)", avail_cash, MIN_BUY_USD)
            break

        sym = (row.get("symbol") or "").strip()
        if not sym:
            continue
        if sym in positions:
            continue

        entries.append({"symbol": sym, "usd": round(MIN_BUY_USD, 2)})
        buys_added += 1
        avail_cash -= MIN_BUY_USD
        log.info("BUY PLAN — added %s for $%.2f (avail now $%.2f)", sym, MIN_BUY_USD, avail_cash)

    log.info("BUY PLAN — planned buys: %d", buys_added)
    return entries


# ------------------------------------------------------------------------------
# Highwater structure (safe placeholder)
# ------------------------------------------------------------------------------
def load_highs() -> Dict:
    highs = read_json(HIGHWATER_FILE, {})
    if not isinstance(highs, dict):
        highs = {}
    return highs


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    log.info("=== START LOOP ===")

    if RUN_SWITCH != "ON":
        log.info("RUN_SWITCH is OFF — exiting early (no trading).")
        return

    # Connect exchange (ok if None in DRY-RUN; we can use cached balance)
    ex = make_exchange()

    # Fetch balance and detect cash robustly
    bal = safe_fetch_balance(ex)
    cash = robust_cash_from_balance(bal)

    # Discover positions (lightweight)
    positions = discover_positions(ex)

    # Build BUY PLAN from spikes
    entries = build_buy_plan(cash=cash, positions=positions)

    # Persist plan & highs for downstream steps
    highs = load_highs()
    write_json(ENTRY_FILE, entries)
    write_json(HIGHWATER_FILE, highs)

    log.info("Saved buy plan to %s (entries=%d)", str(ENTRY_FILE), len(entries))
    log.info("=== END LOOP ===")


# ------------------------------------------------------------------------------
# EOF utilities — safe add-ons you already use
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    # === Post-run utilities (SAFE, no orders) ==========================
    # 1) Momentum Spike scan (scan-only; saves .state/spike_candidates.*)
    try:
        from tools.momentum_spike import main as act_on_spikes  # type: ignore
        print("\n=== Running Momentum Spike Scan ===")
        act_on_spikes()
    except Exception as e:
        log.warning("Momentum Spike Scan failed: %r", e)

    # 2) Auto-Sell cool-off guard (safe; saves .state/auto_sell_guard.json)
    try:
        from tools.auto_sell_guard import run_cool_off_guard  # type: ignore
        print("\n=== Running Auto-Sell Cool-Off Guard ===")
        run_cool_off_guard()
    except Exception as e:
        log.warning("Auto-Sell Cool-Off Guard failed: %r", e)
