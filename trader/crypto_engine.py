# trader/crypto_engine.py
#!/usr/bin/env python3
"""
Crypto engine for Hourly 1-Coin Rotation.
- Enforces Kraken pair minimums BEFORE sending orders.
- Writes .state/buy_gates.md explaining skips/bumps.
- Supports whitelist and min-price guard.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import time

# ccxt is preinstalled in your workflows
import ccxt

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
BUY_GATES_MD = STATE_DIR / "buy_gates.md"
POS_FILE     = STATE_DIR / "positions.json"
LAST_EXIT    = STATE_DIR / "last_exit_code.txt"

# ------------- helpers ----------------

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

def load_candidates() -> List[str]:
    """
    Returns candidate symbols like 'SOL/USD'.
    Order of preference:
      1) WHITELIST env (comma-separated)
      2) .state/momentum_candidates.csv (column: symbol or first column)
    """
    wl = env_str("WHITELIST", "")
    if wl:
        return [s.strip() for s in wl.split(",") if s.strip()]

    path = STATE_DIR / "momentum_candidates.csv"
    if path.exists():
        out = []
        with path.open() as f:
            r = csv.reader(f)
            header = next(r, [])
            # Try to find 'symbol' column
            sym_idx = None
            for i, h in enumerate(header):
                if h.strip().lower() == "symbol":
                    sym_idx = i
                    break
            if sym_idx is None:
                # use first column
                sym_idx = 0
                # also treat the first row as data if header was empty
                if len(header) > 0:
                    out.append(header[sym_idx].strip())
            for row in r:
                if not row:
                    continue
                out.append(row[sym_idx].strip())
        # Normalize like 'SOL/USD'
        return [s.replace("-", "/").upper() for s in out if s.strip()]

    return []

def write_buy_gates(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Buy Gates\n",
        "Pair | Price | Kraken Min Amount | Kraken Min Cost | BUY_USD | Decision\n",
        "---|---:|---:|---:|---:|---\n"
    ]
    for r in rows:
        lines.append(
            f"{r['pair']} | {r['price']:.8f} | {r['min_amount']} | ${r['min_cost']:.4f} | ${r['buy_usd']:.2f} | {r['decision']}"
        )
    BUY_GATES_MD.write_text("\n".join(lines))

def write_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    md = [
        f"# Crypto — Hourly 1-Coin Rotation",
        "",
        f"**When:** {data.get('when','')} UTC",
        f"**DRY_RUN:** {data.get('dry_run','')}",
        f"**BUY_USD:** {data.get('buy_usd','')}",
        f"**TP_PCT:** {data.get('tp_pct','')}",
        f"**STOP_PCT:** {data.get('stop_pct','')}",
        f"**WINDOW_MIN:** {data.get('window_min','')}",
        f"**SLOW_GAIN_REQ:** {data.get('slow_gain_req','')}",
        f"**UNIVERSE_PICK:** {data.get('universe_pick','')}",
        f"**Engine:** trader.crypto_engine.run_hourly_rotation",
        f"**Status:** {data.get('status','')}",
        f"**Note:** {data.get('note','')}",
        ""
    ]
    SUMMARY_MD.write_text("\n".join(md))

def _kraken():
    # Accept either naming
    key = env_str("KRAKEN_API_KEY", env_str("KRAKEN_KEY", ""))
    sec = env_str("KRAKEN_API_SECRET", env_str("KRAKEN_SECRET", ""))
    dry = env_str("DRY_RUN", "ON").upper() != "OFF"

    # Use rate-limit safe defaults
    ex = ccxt.kraken({
        "apiKey": key,
        "secret": sec,
        "enableRateLimit": True,
        "options": {
            "fetchMinOrderAmounts": True
        }
    })
    # In DRY-RUN we still want market metadata
    ex.load_markets()
    return ex, dry, bool(key and sec)

def _pair_limits(ex: ccxt.kraken, pair: str, price: float) -> Dict[str, float]:
    """
    Returns dict with min_amount and min_cost (USD).
    Handles cases where one or both are provided by ccxt.
    """
    market = ex.market(pair)
    min_amount = None
    min_cost = None
    try:
        min_amount = float(market.get("limits", {}).get("amount", {}).get("min") or 0.0)
    except Exception:
        min_amount = 0.0
    try:
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min") or 0.0)
    except Exception:
        min_cost = 0.0

    # Derive min_cost via min_amount if cost not present
    implied_cost = (min_amount or 0.0) * float(price or 0.0)
    if (min_cost or 0.0) < implied_cost:
        min_cost = implied_cost

    return {"min_amount": float(min_amount or 0.0), "min_cost": float(min_cost or 0.0)}

def _fetch_price(ex: ccxt.kraken, pair: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(pair)
        # use last or close as fallback
        p = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        return float(p) if p else None
    except Exception:
        return None

def _ensure_pair_format(sym: str) -> str:
    # Normalize to 'BASE/QUOTE' with uppercase
    s = sym.strip().upper().replace("-", "/")
    # Kraken commonly uses 'USD' quote; leave as provided
    return s

# ------------- main entry -------------

def run_hourly_rotation() -> None:
    """
    Minimal rotation buy path with Kraken min checks.
    This function focuses on BUY path (your SELL guard runs elsewhere).
    """
    when_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    buy_usd = env_float("BUY_USD", 25.0)
    min_price = env_float("MIN_PRICE_USD", 0.0)  # e.g., 0.01 to avoid sub-cent tokens

    ex, dry_run, have_keys = _kraken()

    # Load candidates
    candidates = [c for c in load_candidates() if c]
    candidates = [_ensure_pair_format(c) for c in candidates]
    if not candidates:
        write_summary({
            "when": f"{when_utc}",
            "dry_run": "OFF" if not dry_run else "ON",
            "buy_usd": str(buy_usd),
            "tp_pct": env_str("TP_PCT","5"),
            "stop_pct": env_str("STOP_PCT","1"),
            "window_min": env_str("WINDOW_MIN","30"),
            "slow_gain_req": env_str("SLOW_GAIN_REQ","3"),
            "universe_pick": "AUTO",
            "status": "noop",
            "note": "No candidates available (WHITELIST empty and momentum_candidates.csv not found)."
        })
        LAST_EXIT.write_text("0")
        return

    gate_rows: List[Dict[str, Any]] = []
    picked: Optional[str] = None
    final_amount = 0.0
    final_cost   = 0.0

    for sym in candidates:
        price = _fetch_price(ex, sym)
        if price is None or price <= 0:
            gate_rows.append({"pair": sym, "price": 0.0, "min_amount": 0.0, "min_cost": 0.0,
                              "buy_usd": buy_usd, "decision": "Skip — no price"})
            continue

        if min_price > 0 and price < min_price:
            gate_rows.append({"pair": sym, "price": price, "min_amount": 0.0, "min_cost": 0.0,
                              "buy_usd": buy_usd, "decision": f"Skip — price ${price:.8f} < MIN_PRICE_USD"})
            continue

        lim = _pair_limits(ex, sym, price)
        min_amt = lim["min_amount"]
        min_cost = lim["min_cost"]

        # Our intended order
        intended_amount = buy_usd / price

        if intended_amount >= min_amt and buy_usd >= min_cost:
            # Good as-is
            picked = sym
            final_amount = intended_amount
            final_cost = buy_usd
            gate_rows.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                              "buy_usd": buy_usd, "decision": f"OK — will buy ~{final_amount:.8f} for ${final_cost:.2f}"})
            break

        # Try auto-bump to Kraken minimum if affordable within BUY_USD
        bumped_amount = max(intended_amount, min_amt)
        bumped_cost = bumped_amount * price
        # To be strict: only auto-bump when bumped_cost <= BUY_USD
        if bumped_cost <= buy_usd:
            picked = sym
            final_amount = bumped_amount
            final_cost = bumped_cost
            gate_rows.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                              "buy_usd": buy_usd, "decision": f"Auto-bump — buy min {final_amount:.8f} (${final_cost:.2f})"})
            break

        # Not affordable with current BUY_USD
        gate_rows.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                          "buy_usd": buy_usd, "decision": "Skip — Kraken minimum exceeds BUY_USD"})

    # Always write gate audit
    write_buy_gates(gate_rows)

    if not picked:
        write_summary({
            "when": f"{when_utc}",
            "dry_run": "OFF" if not dry_run else "ON",
            "buy_usd": str(buy_usd),
            "tp_pct": env_str("TP_PCT","5"),
            "stop_pct": env_str("STOP_PCT","1"),
            "window_min": env_str("WINDOW_MIN","30"),
            "slow_gain_req": env_str("SLOW_GAIN_REQ","3"),
            "universe_pick": "AUTO",
            "status": "ok",
            "note": "No affordable pairs — all below price floor or exceeded Kraken minimum; see .state/buy_gates.md."
        })
        LAST_EXIT.write_text("0")
        return

    # Execute BUY
    note = ""
    try:
        if dry_run or not have_keys:
            note = "[DRY] Simulated BUY {} amount={} (~${:.2f})".format(picked, final_amount, final_cost)
        else:
            # Kraken requires market symbol in exchange format; ccxt handles mapping.
            order = ex.create_market_buy_order(picked, final_amount)
            note = f"[LIVE] BUY ok {picked} amount={final_amount:.8f} cost≈${final_cost:.2f} order_id={order.get('id','?')}"
        # Lightweight positions file
        pos = {"pair": picked, "amount": final_amount, "est_cost": round(final_cost, 8), "when": when_utc}
        POS_FILE.write_text(json.dumps(pos, indent=2))
        status = "ok"
    except ccxt.BaseError as e:
        status = "error"
        note = f"InvalidOrder: kraken {str(e)}"

    write_summary({
        "when": f"{when_utc}",
        "dry_run": "OFF" if not dry_run else "ON",
        "buy_usd": str(buy_usd),
        "tp_pct": env_str("TP_PCT","5"),
        "stop_pct": env_str("STOP_PCT","1"),
        "window_min": env_str("WINDOW_MIN","30"),
        "slow_gain_req": env_str("SLOW_GAIN_REQ","3"),
        "universe_pick": "AUTO",
        "status": status,
        "note": note
    })
    LAST_EXIT.write_text("0" if status == "ok" else "1")


# Backward-compatible alias if the runner imports by name
def main():
    run_hourly_rotation()

if __name__ == "__main__":
    run_hourly_rotation()
