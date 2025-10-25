#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Live runner with Sell-Logic Guard
- Reads PROTECT_JSON from env
- Evaluates TP, SL, TSL, DIP_CUTOFF, AGE_LIMIT, APR_SPIKE_EXIT, MIN_SELL_USD
- Writes artifacts into .state/
- Executes sells if DRY_RUN=OFF and a sell adapter is available

This file tries to be conservative: if it can't find prices/positions or a
sell adapter, it will STILL produce a clear .state/sell_decisions.csv
and .state/sell_orders_todo.json so you can see exactly what it *would* do.
"""

from __future__ import annotations
import os, json, math, time, csv, traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Config / Paths
# --------------------------------------------------------------------------------------
STATE_DIR = Path(os.environ.get("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

DRY_RUN = os.environ.get("DRY_RUN", "ON").upper()  # "ON" == dry run by default
RUN_SWITCH = os.environ.get("RUN_SWITCH", "ON").upper()

# Optional JSON blobs (passed via workflow_dispatch inputs)
ADVANCED_JSON_RAW = os.environ.get("ADVANCED_JSON", "") or ""
PROTECT_JSON_RAW  = os.environ.get("PROTECT_JSON", "") or ""

# Bot knobs from env/vars (with fallbacks)
MIN_SELL_USD      = float(os.environ.get("MIN_SELL_USD", "10"))
DUST_MIN_USD      = float(os.environ.get("DUST_MIN_USD", "2"))
DUST_SKIP_STABLES = (os.environ.get("DUST_SKIP_STABLES", "true").lower() == "true")

# --------------------------------------------------------------------------------------
# Utility: write small artifacts so you can inspect runs
# --------------------------------------------------------------------------------------
def write_text(relpath: str, text: str) -> None:
    p = STATE_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def write_json(relpath: str, data: Any) -> None:
    p = STATE_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def append_csv(relpath: str, header: List[str], rows: List[List[Any]]) -> None:
    p = STATE_DIR / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --------------------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------------------
def parse_json_or_empty(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        write_text("protect_json_error.txt", f"Failed to parse JSON: {e}\nRaw:\n{raw}")
        return {}

PROTECT = parse_json_or_empty(PROTECT_JSON_RAW)
ADVANCED = parse_json_or_empty(ADVANCED_JSON_RAW)

# Persist what we received for transparency
write_json("effective_protect.json", PROTECT)
write_json("effective_advanced.json", ADVANCED)

# Defaults if keys missing
SELL_GUARD = {
    "MIN_SELL_USD": MIN_SELL_USD,
    "TP":  {"enable": True,  "take_profit_pct": 0.08,  "lock_after_minutes": 15},
    "SL":  {"enable": True,  "stop_loss_pct": -0.06},
    "TSL": {"enable": True,  "trail_pct": 0.03, "arm_after_gain_pct": 0.04},
    "DIP_CUTOFF": {"max_drawdown_pct": -0.09, "cooldown_min": 120},
    "AGE_LIMIT":  {"max_hold_hours": 36},
    "APR_SPIKE_EXIT": {"enable": True, "apr_threshold": 0.90, "min_gain_pct": 0.02},
}
if "SELL_GUARD" in PROTECT:
    # Merge shallowly (keep sensible defaults)
    for k, v in PROTECT["SELL_GUARD"].items():
        SELL_GUARD[k] = v

# --------------------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------------------
@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float           # average entry price (USD)
    last_price: float          # current price (USD)
    usd_value: float           # qty * last_price
    opened_at: Optional[str]   # ISO UTC when acquired (if known)
    apr_estimate: Optional[float] = None  # 0..1 range (e.g., 0.95 == 95%)
    realized_pnl_usd: float = 0.0         # optional metric
    trailing_max_price: Optional[float] = None  # for TSL state (from prior runs)
    meta: Dict[str, Any] = None

    def age_hours(self) -> Optional[float]:
        if not self.opened_at:
            return None
        try:
            dt = datetime.fromisoformat(self.opened_at.replace("Z","+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
        except Exception:
            return None

    def gain_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (self.last_price / self.avg_price) - 1.0

# --------------------------------------------------------------------------------------
# Adapters (fetch positions / prices). We try to reuse your existing code if present.
# --------------------------------------------------------------------------------------
def try_import_adapters():
    """
    We try to import your existing adapter functions if they exist:
      - trader.adapters.get_open_positions() -> List[dict]
      - trader.adapters.place_market_sell(symbol: str, qty: float) -> dict
      - trader.adapters.get_price(symbol: str) -> float
    If not present, we fall back to reading .state/positions.json and .state/last_prices.json.
    """
    adapter = {}
    try:
        import importlib
        mod = importlib.import_module("trader.adapters")  # your repo may have this
        adapter["get_open_positions"] = getattr(mod, "get_open_positions", None)
        adapter["place_market_sell"]  = getattr(mod, "place_market_sell", None)
        adapter["get_price"]          = getattr(mod, "get_price", None)
    except Exception:
        pass
    return adapter

ADAPTER = try_import_adapters()

def load_positions_fallback() -> List[Position]:
    """
    Fallback loader: expects .state/positions.json with a list of items containing:
      symbol, qty, avg_price, last_price (optional), opened_at (ISO)
    We'll also consider .state/last_prices.json if last_price missing.
    """
    pos_path = STATE_DIR / "positions.json"
    if not pos_path.exists():
        # As a last resort, treat everything as empty, but log it for you.
        write_text("warnings.txt",
                   "No adapters and no .state/positions.json found; "
                   "Sell-Guard produced only a header artifact.\n")
        return []

    data = json.loads(pos_path.read_text(encoding="utf-8"))
    # last prices map (optional)
    lp = {}
    lp_path = STATE_DIR / "last_prices.json"
    if lp_path.exists():
        lp = json.loads(lp_path.read_text(encoding="utf-8"))

    out: List[Position] = []
    for raw in data:
        symbol = raw.get("symbol")
        qty = float(raw.get("qty", 0) or 0)
        avg_price = float(raw.get("avg_price", 0) or 0)
        last_price = float(raw.get("last_price", lp.get(symbol, 0)) or 0)
        opened_at = raw.get("opened_at")
        apr = raw.get("apr_estimate")
        if qty <= 0 or (avg_price <= 0 and last_price <= 0):
            continue
        usd_value = qty * (last_price if last_price > 0 else avg_price)
        out.append(Position(
            symbol=symbol,
            qty=qty,
            avg_price=avg_price,
            last_price=last_price,
            usd_value=usd_value,
            opened_at=opened_at,
            apr_estimate=apr,
            trailing_max_price=raw.get("trailing_max_price"),
            realized_pnl_usd=float(raw.get("realized_pnl_usd", 0) or 0),
            meta=raw,
        ))
    return out

def fetch_positions() -> List[Position]:
    # Prefer user's adapter if available
    get_open_positions = ADAPTER.get("get_open_positions")
    get_price = ADAPTER.get("get_price")
    if callable(get_open_positions):
        raw_positions = get_open_positions()
        out: List[Position] = []
        for r in raw_positions:
            symbol = r["symbol"]
            qty = float(r.get("qty", 0) or 0)
            avg_price = float(r.get("avg_price", 0) or 0)
            last_price = float(r.get("last_price", 0) or 0)
            if not last_price and callable(get_price):
                try:
                    last_price = float(get_price(symbol) or 0)
                except Exception:
                    last_price = 0.0
            opened_at = r.get("opened_at")
            apr = r.get("apr_estimate")
            if qty <= 0 or (avg_price <= 0 and last_price <= 0):
                continue
            usd_value = qty * (last_price if last_price > 0 else avg_price)
            out.append(Position(
                symbol=symbol,
                qty=qty,
                avg_price=avg_price,
                last_price=last_price,
                usd_value=usd_value,
                opened_at=opened_at,
                apr_estimate=apr,
                trailing_max_price=r.get("trailing_max_price"),
                realized_pnl_usd=float(r.get("realized_pnl_usd", 0) or 0),
                meta=r,
            ))
        return out
    # Fallback to .state files
    return load_positions_fallback()

def place_market_sell(symbol: str, qty: float) -> Dict[str, Any]:
    fn = ADAPTER.get("place_market_sell")
    if callable(fn):
        return fn(symbol, qty)
    # No adapter? We'll just record a TODO order so nothing breaks.
    todo = {"symbol": symbol, "qty": qty, "action": "SELL", "ts": now_iso(), "executed": False}
    todos = []
    p = STATE_DIR / "sell_orders_todo.json"
    if p.exists():
        try:
            todos = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            todos = []
    todos.append(todo)
    write_json("sell_orders_todo.json", todos)
    return {"status": "NO_ADAPTER", "todo_recorded": True, "order": todo}

# --------------------------------------------------------------------------------------
# Guard evaluation
# --------------------------------------------------------------------------------------
def evaluate_sell_decision(p: Position, guard: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Returns (should_sell, reason)
    Order of operations:
      1) Dust skip (optionally skip stables)
      2) Take Profit
      3) Stop Loss
      4) Trailing SL (if armed)
      5) APR spike exit
      6) Max drawdown cutoff (uses current gain)
      7) Age limit
      8) MIN_SELL_USD
    """
    g = guard
    gain = p.gain_pct()  # e.g., +0.08 == +8%

    # Skip tiny dust or stables if configured
    if p.usd_value < DUST_MIN_USD:
        return (False, f"SKIP_DUST(<{DUST_MIN_USD} USD)")

    if DUST_SKIP_STABLES and p.symbol.upper() in {"USDT","USDC","DAI","USD","USDK","TUSD","FDUSD","USDP"}:
        return (False, "SKIP_STABLE")

    # 1) TP
    tp = g.get("TP", {})
    if tp.get("enable", True):
        if gain >= float(tp.get("take_profit_pct", 0.08)):
            return (True, f"TP_HIT({gain:+.2%})")

    # 2) SL
    sl = g.get("SL", {})
    if sl.get("enable", True):
        if gain <= float(sl.get("stop_loss_pct", -0.06)):
            return (True, f"STOP_LOSS({gain:+.2%})")

    # 3) TSL (arm after certain gain)
    tsl = g.get("TSL", {})
    if tsl.get("enable", True):
        arm_after = float(tsl.get("arm_after_gain_pct", 0.04))
        trail = float(tsl.get("trail_pct", 0.03))
        if gain >= arm_after:
            # Without persistent price trail, emulate by requiring price pullback from peak
            # If we don't have trailing_max_price, use last_price as starting peak
            peak = p.meta.get("trailing_max_price") if p.meta else None
            if not peak or peak < p.last_price:
                # update peak file for next run
                peak_map = {}
                path = STATE_DIR / "trailing_peaks.json"
                if path.exists():
                    try:
                        peak_map = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        peak_map = {}
                peak_map[p.symbol] = max(p.last_price, float(peak_map.get(p.symbol, 0) or 0))
                write_json("trailing_peaks.json", peak_map)
            else:
                # if current price is below peak by trail%
                if p.last_price <= (1.0 - trail) * peak:
                    return (True, f"TSL_HIT({gain:+.2%},trail={trail:.2%})")

    # 4) APR spike assisted exit
    apr_cfg = g.get("APR_SPIKE_EXIT", {})
    if apr_cfg.get("enable", True) and p.apr_estimate is not None:
        if p.apr_estimate >= float(apr_cfg.get("apr_threshold", 0.90)):
            if gain >= float(apr_cfg.get("min_gain_pct", 0.02)):
                return (True, f"APR_SPIKE_EXIT(apr={p.apr_estimate:.0%},gain={gain:+.2%})")

    # 5) Max drawdown cutoff (use current gain as proxy if no equity curve)
    dd = g.get("DIP_CUTOFF", {})
    if float(dd.get("max_drawdown_pct", -0.09)) >= -0.0001:
        # if loss beyond threshold, sell
        if gain <= float(dd.get("max_drawdown_pct", -0.09)):
            return (True, f"DRAW_DOWN({gain:+.2%})")

    # 6) Age limit
    age_cfg = g.get("AGE_LIMIT", {})
    age = p.age_hours()
    if age is not None and float(age_cfg.get("max_hold_hours", 36)) > 0:
        if age >= float(age_cfg.get("max_hold_hours", 36)):
            return (True, f"AGE_LIMIT({age:.1f}h)")

    # 7) Min sell USD enforced at the very end (block sells under threshold)
    if p.usd_value < float(g.get("MIN_SELL_USD", MIN_SELL_USD)):
        return (False, f"BLOCK_MIN_SELL(<{g.get('MIN_SELL_USD', MIN_SELL_USD)} USD)")

    return (False, "HOLD")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    # Header
    write_text("run_header.txt", "\n".join([
        "=== Sell-Guard Run ===",
        f"time_utc: {now_iso()}",
        f"DRY_RUN: {DRY_RUN}",
        f"RUN_SWITCH: {RUN_SWITCH}",
        f"MIN_SELL_USD: {SELL_GUARD.get('MIN_SELL_USD', MIN_SELL_USD)}",
    ]))

    if RUN_SWITCH != "ON":
        write_text("run_skipped.txt", "RUN_SWITCH is OFF, skipping.")
        return 0

    positions = fetch_positions()

    # Write snapshot of positions we evaluated
    write_json("positions_evaluated.json", [asdict(p) for p in positions])

    decisions_rows: List[List[Any]] = []
    orders_executed: List[Dict[str, Any]] = []
    orders_recommended: List[Dict[str, Any]] = []

    header = [
        "ts_utc", "symbol", "qty", "avg_price", "last_price", "usd_value",
        "gain_pct", "age_hours", "apr_estimate", "decision", "dry_run"
    ]

    for p in positions:
        should_sell, reason = evaluate_sell_decision(p, SELL_GUARD)
        row = [
            now_iso(), p.symbol, f"{p.qty:.8f}", f"{p.avg_price:.8f}", f"{p.last_price:.8f}",
            f"{p.usd_value:.2f}", f"{p.gain_pct():+.4%}",
            f"{(p.age_hours() if p.age_hours() is not None else -1):.2f}",
            (f"{p.apr_estimate:.2%}" if p.apr_estimate is not None else ""),
            reason, DRY_RUN
        ]
        decisions_rows.append(row)

        if should_sell:
            order = {"symbol": p.symbol, "qty": p.qty, "reason": reason, "ts": now_iso()}
            if DRY_RUN != "OFF":
                orders_recommended.append(order)
            else:
                try:
                    resp = place_market_sell(p.symbol, p.qty)
                    order["exec_response"] = resp
                    orders_executed.append(order)
                except Exception as e:
                    order["exec_error"] = str(e)
                    order["traceback"] = traceback.format_exc()
                    orders_recommended.append(order)

    append_csv("sell_decisions.csv", header, decisions_rows)
    write_json("sell_orders_recommended.json", orders_recommended)
    write_json("sell_orders_executed.json", orders_executed)

    # Light summary
    summary = {
        "time_utc": now_iso(),
        "dry_run": DRY_RUN,
        "positions_checked": len(positions),
        "orders_executed": len(orders_executed),
        "orders_recommended": len(orders_recommended),
    }
    write_json("sell_guard_summary.json", summary)

    # Optional Slack webhook (simple text) if you already store it as secret
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if webhook:
        try:
            import urllib.request
            msg = f"[Sell-Guard] checked {len(positions)} positions. "\
                  f"Exec: {len(orders_executed)}, Reco: {len(orders_recommended)}. DRY_RUN={DRY_RUN}"
            payload = json.dumps({"text": msg}).encode("utf-8")
            req = urllib.request.Request(webhook, data=payload,
                                         headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=6).read()
        except Exception:
            pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
