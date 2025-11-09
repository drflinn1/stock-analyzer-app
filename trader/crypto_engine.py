# trader/crypto_engine.py
#!/usr/bin/env python3
"""
Crypto engine for Hourly 1-Coin Rotation.

- Enforces Kraken pair minimums BEFORE sending orders.
- Accepts kwargs from main.py (dry_run, buy_usd, tp_pct, stop_pct, window_min, slow_gain_req, universe_pick).
- Writes .state/buy_gates.md explaining OK / Auto-bump / Skip.
- Supports WHITELIST and MIN_PRICE_USD guards.
"""
from __future__ import annotations

import csv, json, os, time
from pathlib import Path
from typing import Dict, Any, List, Optional

import ccxt

STATE_DIR = Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
BUY_GATES_MD = STATE_DIR / "buy_gates.md"
POS_FILE     = STATE_DIR / "positions.json"
LAST_EXIT    = STATE_DIR / "last_exit_code.txt"

# ---------------- helpers ----------------

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

def _ensure_pair_format(sym: str) -> str:
    return sym.strip().upper().replace("-", "/")

def load_candidates() -> List[str]:
    """
    Candidate list, priority:
      1) WHITELIST env (comma-separated, e.g., 'SOL/USD,ADA/USD')
      2) .state/momentum_candidates.csv (column 'symbol' or first column)
    """
    wl = env_str("WHITELIST", "")
    if wl:
        return [_ensure_pair_format(s) for s in wl.split(",") if s.strip()]

    p = STATE_DIR / "momentum_candidates.csv"
    if not p.exists():
        return []
    out: List[str] = []
    with p.open() as f:
        r = csv.reader(f)
        header = next(r, [])
        sym_idx = None
        for i, h in enumerate(header):
            if str(h).strip().lower() == "symbol":
                sym_idx = i
                break
        if sym_idx is None:
            # treat header as first data row when no header detected
            if header:
                out.append(str(header[0]).strip())
            sym_idx = 0
        for row in r:
            if row:
                out.append(str(row[sym_idx]).strip())
    return [_ensure_pair_format(s) for s in out if s]

def write_buy_gates(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Buy Gates\n",
        "Pair | Price | Kraken Min Amount | Kraken Min Cost | BUY_USD | Decision\n",
        "---|---:|---:|---:|---:|---\n",
    ]
    for r in rows:
        lines.append(
            f"{r['pair']} | {r['price']:.8f} | {r['min_amount']} | ${r['min_cost']:.4f} | ${r['buy_usd']:.2f} | {r['decision']}"
        )
    BUY_GATES_MD.write_text("\n".join(lines))

def write_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    md = [
        f"# Crypto — Hourly 1-Coin Rotation", "",
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
    key = env_str("KRAKEN_API_KEY", env_str("KRAKEN_KEY", ""))
    sec = env_str("KRAKEN_API_SECRET", env_str("KRAKEN_SECRET", ""))
    dry = env_str("DRY_RUN", "ON").upper() != "OFF"
    ex = ccxt.kraken({
        "apiKey": key,
        "secret": sec,
        "enableRateLimit": True,
        "options": {"fetchMinOrderAmounts": True},
    })
    ex.load_markets()
    return ex, dry, bool(key and sec)

def _fetch_price(ex: ccxt.kraken, pair: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(pair)
        p = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        return float(p) if p else None
    except Exception:
        return None

def _pair_limits(ex: ccxt.kraken, pair: str, price: float) -> Dict[str, float]:
    m = ex.market(pair)
    min_amt = float(m.get("limits", {}).get("amount", {}).get("min") or 0.0)
    min_cost = float(m.get("limits", {}).get("cost", {}).get("min") or 0.0)
    implied = min_amt * float(price or 0.0)
    if min_cost < implied:
        min_cost = implied
    return {"min_amount": float(min_amt), "min_cost": float(min_cost)}

# ---------------- main entry ----------------

def run_hourly_rotation(
    *,
    dry_run: Optional[bool] = None,
    buy_usd: Optional[float] = None,
    tp_pct: Optional[float] = None,
    stop_pct: Optional[float] = None,
    window_min: Optional[int] = None,
    slow_gain_req: Optional[float] = None,
    universe_pick: Optional[str] = None,
    **_kwargs,
) -> None:
    """
    BUY path with Kraken min checks. Accepts kwargs from main.py.
    """
    when_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Env fallbacks (so you can also pass via UI)
    if dry_run is None:
        dry_run = (env_str("DRY_RUN", "ON").upper() != "OFF")
    if buy_usd is None:
        buy_usd = env_float("BUY_USD", 25.0)
    min_price = env_float("MIN_PRICE_USD", 0.0)  # set to 0.01 to avoid sub-cent tokens

    ex, dry_env, have_keys = _kraken()
    # dry_run param overrides env if provided
    if dry_run is None:
        dry_run = dry_env

    # Build candidates (WHITELIST or momentum_candidates.csv)
    candidates = load_candidates()
    if universe_pick and not candidates:
        # compatibility: if main passes a single symbol here
        candidates = [_ensure_pair_format(universe_pick)]

    if not candidates:
        write_summary({
            "when": when_utc,
            "dry_run": "OFF" if not dry_run else "ON",
            "buy_usd": str(buy_usd),
            "tp_pct": str(tp_pct) if tp_pct is not None else env_str("TP_PCT","5"),
            "stop_pct": str(stop_pct) if stop_pct is not None else env_str("STOP_PCT","1"),
            "window_min": str(window_min) if window_min is not None else env_str("WINDOW_MIN","30"),
            "slow_gain_req": str(slow_gain_req) if slow_gain_req is not None else env_str("SLOW_GAIN_REQ","3"),
            "universe_pick": universe_pick or "",
            "status": "noop",
            "note": "No candidates (WHITELIST empty and momentum_candidates.csv not found)."
        })
        LAST_EXIT.write_text("0")
        return

    gates: List[Dict[str, Any]] = []
    picked = None
    final_amount = 0.0
    final_cost   = 0.0

    for sym in candidates:
        price = _fetch_price(ex, sym)
        if price is None or price <= 0:
            gates.append({"pair": sym, "price": 0.0, "min_amount": 0.0, "min_cost": 0.0,
                          "buy_usd": buy_usd, "decision": "Skip — no price"})
            continue

        if min_price > 0 and price < min_price:
            gates.append({"pair": sym, "price": price, "min_amount": 0.0, "min_cost": 0.0,
                          "buy_usd": buy_usd, "decision": f"Skip — price ${price:.8f} < MIN_PRICE_USD"})
            continue

        lim = _pair_limits(ex, sym, price)
        min_amt = lim["min_amount"]
        min_cost = lim["min_cost"]

        intended_amount = float(buy_usd) / price

        if intended_amount >= min_amt and float(buy_usd) >= min_cost:
            picked = sym
            final_amount = intended_amount
            final_cost = float(buy_usd)
            gates.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                          "buy_usd": buy_usd, "decision": f"OK — will buy ~{final_amount:.8f} for ${final_cost:.2f}"})
            break

        bumped_amount = max(intended_amount, min_amt)
        bumped_cost = bumped_amount * price
        # Only auto-bump if the bumped order STILL fits inside BUY_USD
        if bumped_cost <= float(buy_usd):
            picked = sym
            final_amount = bumped_amount
            final_cost = bumped_cost
            gates.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                          "buy_usd": buy_usd, "decision": f"Auto-bump — buy min {final_amount:.8f} (${final_cost:.2f})"})
            break

        gates.append({"pair": sym, "price": price, "min_amount": min_amt, "min_cost": min_cost,
                      "buy_usd": buy_usd, "decision": "Skip — Kraken minimum exceeds BUY_USD"})

    # Always emit audit of decisions
    write_buy_gates(gates)

    if not picked:
        write_summary({
            "when": when_utc,
            "dry_run": "OFF" if not dry_run else "ON",
            "buy_usd": str(buy_usd),
            "tp_pct": str(tp_pct) if tp_pct is not None else env_str("TP_PCT","5"),
            "stop_pct": str(stop_pct) if stop_pct is not None else env_str("STOP_PCT","1"),
            "window_min": str(window_min) if window_min is not None else env_str("WINDOW_MIN","30"),
            "slow_gain_req": str(slow_gain_req) if slow_gain_req is not None else env_str("SLOW_GAIN_REQ","3"),
            "universe_pick": universe_pick or "",
            "status": "ok",
            "note": "No affordable pairs — see .state/buy_gates.md."
        })
        LAST_EXIT.write_text("0")
        return

    # Attempt BUY
    note = ""
    try:
        if dry_run or not have_keys:
            note = f"[DRY] Simulated BUY {picked} amount={final_amount:.8f} (~${final_cost:.2f})"
        else:
            order = ex.create_market_buy_order(picked, final_amount)
            note = f"[LIVE] BUY ok {picked} amount={final_amount:.8f} cost≈${final_cost:.2f} order_id={order.get('id','?')}"
        POS_FILE.write_text(json.dumps({"pair": picked, "amount": final_amount,
                                        "est_cost": round(final_cost, 8),
                                        "when": when_utc}, indent=2))
        status = "ok"
    except ccxt.BaseError as e:
        status = "error"
        note = f"InvalidOrder: kraken {str(e)}"

    write_summary({
        "when": when_utc,
        "dry_run": "OFF" if not dry_run else "ON",
        "buy_usd": str(buy_usd),
        "tp_pct": str(tp_pct) if tp_pct is not None else env_str("TP_PCT","5"),
        "stop_pct": str(stop_pct) if stop_pct is not None else env_str("STOP_PCT","1"),
        "window_min": str(window_min) if window_min is not None else env_str("WINDOW_MIN","30"),
        "slow_gain_req": str(slow_gain_req) if slow_gain_req is not None else env_str("SLOW_GAIN_REQ","3"),
        "universe_pick": universe_pick or "",
        "status": status,
        "note": note
    })
    LAST_EXIT.write_text("0" if status == "ok" else "1")

def main():
    run_hourly_rotation()

if __name__ == "__main__":
    run_hourly_rotation()
