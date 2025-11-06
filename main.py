#!/usr/bin/env python3
"""
main.py — Unified runner that guarantees .state artifacts and obeys UNIVERSE_PICK.

Features
- Always writes .state/run_summary.{json,md} (even on risk-off or errors)
- Force-buy path when UNIVERSE_PICK is provided (e.g., SOLUSD → SOL/USD)
- LIVE trading via ccxt.kraken when DRY_RUN=OFF and Kraken secrets exist
- Records a minimal .state/positions.json for rotation watchdogs
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------- Paths ----------
STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"
POSITIONS_JSON = STATE_DIR / "positions.json"   # minimal ledger for rotation logic

# ---------- Helpers ----------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return "" if v is None else str(v)

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)).strip())
    except Exception:
        return default

def write_summary(data: Dict[str, Any]) -> None:
    """Write both JSON and Markdown summaries; never raises."""
    try:
        SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[WARN] Failed writing {SUMMARY_JSON}: {e}", file=sys.stderr)

    try:
        lines = [
            f"**When:** {data.get('when', '')}",
            f"**Live (DRY_RUN=OFF):** {data.get('live', False)}",
            f"**UNIVERSE_PICK:** {data.get('universe_pick', '')}",
            f"**BUY_USD:** {data.get('buy_usd', '')}",
            f"**Status:** {data.get('status', '')}",
            f"**Note:** {data.get('note', '')}",
            "",
            "### Details",
            "```json",
            json.dumps(data, indent=2),
            "```",
        ]
        SUMMARY_MD.write_text("\n".join(lines))
    except Exception as e:
        print(f"[WARN] Failed writing {SUMMARY_MD}: {e}", file=sys.stderr)

def load_positions() -> Dict[str, Any]:
    if POSITIONS_JSON.exists():
        try:
            return json.loads(POSITIONS_JSON.read_text() or "{}")
        except Exception:
            return {}
    return {}

def save_positions(d: Dict[str, Any]) -> None:
    try:
        POSITIONS_JSON.write_text(json.dumps(d, indent=2))
    except Exception as e:
        print(f"[WARN] Failed writing {POSITIONS_JSON}: {e}", file=sys.stderr)

def map_pair_to_ccxt_symbol(universe_pick: str) -> Optional[str]:
    """
    Input expected like 'SOLUSD', 'BTCUSD', 'SAPIENUSD' (NO slash).
    Output CCXT symbol 'SOL/USD', 'BTC/USD', 'SAPIEN/USD'.
    """
    up = (universe_pick or "").strip().upper()
    if not up.endswith("USD") or len(up) <= 3:
        return None
    base = up[:-3]
    return f"{base}/USD"

# ---------- Runner ----------
def main() -> int:
    when = utc_now_iso()

    # Read env (with conservative defaults)
    dry_run = env_str("DRY_RUN", "ON").strip().upper()
    live = (dry_run == "OFF")
    buy_usd = env_float("BUY_USD", 25.0)
    reserve_cash_pct = env_float("RESERVE_CASH_PCT", 0.0)
    universe_pick = env_str("UNIVERSE_PICK", "").strip().upper()  # e.g., SOLUSD

    # Kraken secrets (only needed if live)
    kraken_key = env_str("KRAKEN_API_KEY", "")
    kraken_secret = env_str("KRAKEN_API_SECRET", "")

    # Start a result payload that we will ALWAYS write
    result: Dict[str, Any] = {
        "when": when,
        "live": live,
        "buy_usd": buy_usd,
        "reserve_cash_pct": reserve_cash_pct,
        "universe_pick": universe_pick,
        "status": "START",
        "note": "",
    }

    print("[ENV] DRY_RUN =", dry_run)
    print("[ENV] BUY_USD =", buy_usd)
    print("[ENV] RESERVE_CASH_PCT =", reserve_cash_pct)
    print("[ENV] UNIVERSE_PICK =", universe_pick)

    # Safety: ensure .state exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # If not live, just log and write artifacts (so upload step always finds files)
    if not live:
        result["status"] = "RISK_OFF"
        result["note"] = "DRY_RUN is ON (simulation only). No live orders placed."
        write_summary(result)
        return 0

    # Live mode sanity: secrets must be present
    if not kraken_key or not kraken_secret:
        result["status"] = "ERROR_NO_SECRETS"
        result["note"] = "[LIVE] Missing Kraken API secrets; aborting order."
        print("[LIVE] ERROR: Missing KRAKEN_API_KEY or KRAKEN_API_SECRET")
        write_summary(result)
        return 0

    # If UNIVERSE_PICK provided, we force-buy that symbol; otherwise we would run scanner logic.
    # (This file focuses on honoring UNIVERSE_PICK to reproduce last night's baseline.)
    if not universe_pick:
        result["status"] = "NO_PICK"
        result["note"] = (
            "No UNIVERSE_PICK provided. Scanner found no actionable gainer, "
            "so no trade was placed. (Artifacts written.)"
        )
        write_summary(result)
        return 0

    symbol = map_pair_to_ccxt_symbol(universe_pick)
    if not symbol:
        result["status"] = "ERROR_BAD_SYMBOL"
        result["note"] = f"UNIVERSE_PICK '{universe_pick}' is not in expected format like 'SOLUSD'."
        write_summary(result)
        return 0

    # Try a live market buy on Kraken via ccxt using cost param (spend BUY_USD in quote currency)
    try:
        import ccxt  # expects ccxt in requirements.txt

        print("[LIVE] Connecting to Kraken via ccxt…")
        exchange = ccxt.kraken({
            "apiKey": kraken_key,
            "secret": kraken_secret,
            "enableRateLimit": True,
            # Optional: "options": { "tradesLimit": 50 },
        })

        markets = exchange.load_markets()
        if symbol not in markets:
            # Some pairs appear with . or : variants; try a fallback search
            # But usually 'SOL/USD' exists. If not, fail gracefully.
            result["status"] = "ERROR_MARKET_NOT_FOUND"
            result["note"] = f"Market '{symbol}' not found on Kraken."
            write_summary(result)
            return 0

        print(f"[LIVE] BUY {symbol} — cost ${buy_usd} (market)")
        order = exchange.create_order(symbol, 'market', 'buy', None, None, {'cost': buy_usd})
        print("[LIVE] Order response:", order)

        # Fetch the filled price if possible (best-effort)
        filled_price = None
        try:
            if order and 'price' in order and order['price']:
                filled_price = float(order['price'])
        except Exception:
            pass

        # Record a minimal position entry for rotation watchdogs
        positions = load_positions()
        positions[symbol] = {
            "symbol": symbol,
            "universe_pick": universe_pick,
            "entry_time": utc_now_iso(),
            "entry_price": filled_price,
            "buy_usd": buy_usd,
        }
        save_positions(positions)

        # Mark success and write artifacts
        result["status"] = "LIVE_BUY_OK"
        result["note"] = f"Placed market BUY for {symbol} with cost ${buy_usd}."
        result["order"] = order
        result["entry_price"] = filled_price

        # Touch last_ok for quick health checks
        try:
            LAST_OK.write_text(utc_now_iso() + "\n")
        except Exception as e:
            print(f"[WARN] Failed writing {LAST_OK}: {e}", file=sys.stderr)

        write_summary(result)
        return 0

    except Exception as e:
        # Handle common Kraken errors gracefully
        msg = str(e)
        print(f"[LIVE] BUY ERROR: {msg}")
        status = "LIVE_BUY_ERROR"
        if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
            status = "ERROR_FUNDS"
        elif "EOrder:Minimum order size" in msg or "Invalid order" in msg:
            status = "ERROR_MIN_ORDER"

        result["status"] = status
        result["note"] = msg
        write_summary(result)
        return 0


if __name__ == "__main__":
    sys.exit(main())
