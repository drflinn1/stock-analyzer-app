#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, csv, time, base64, hashlib, hmac, urllib.parse, urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# Paths / ENV
# =====================================================================
STATE_DIR = Path(os.environ.get("STATE_DIR", ".state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

DRY_RUN    = os.environ.get("DRY_RUN", "ON").upper()         # ON = simulate / validate
RUN_SWITCH = os.environ.get("RUN_SWITCH", "ON").upper()      # OFF = skip logic

ADVANCED_JSON_RAW = os.environ.get("ADVANCED_JSON", "") or ""
PROTECT_JSON_RAW  = os.environ.get("PROTECT_JSON", "")  or ""

# =====================================================================
# Helpers
# =====================================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_text(rel: str, text: str) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def write_json(rel: str, data: Any) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def append_csv(rel: str, header: List[str], rows: List[List[Any]]) -> None:
    p = STATE_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        for r in rows: w.writerow(r)

def parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        write_text("json_parse_error.txt", f"Failed to parse JSON: {e}\nRaw:\n{raw}")
        return {}

PROTECT  = parse_json(PROTECT_JSON_RAW)
ADVANCED = parse_json(ADVANCED_JSON_RAW)
write_json("effective_protect.json", PROTECT)
write_json("effective_advanced.json", ADVANCED)

# =====================================================================
# Guard / Buy config (with sane defaults)
# =====================================================================
SELL_GUARD = {
    "MIN_SELL_USD": 10,
    "TP":  {"enable": True, "take_profit_pct": 0.08, "lock_after_minutes": 15},
    "SL":  {"enable": True, "stop_loss_pct": -0.06},
    "TSL": {"enable": True, "trail_pct": 0.03, "arm_after_gain_pct": 0.04},
    "DIP_CUTOFF": {"max_drawdown_pct": -0.09, "cooldown_min": 120},
    "AGE_LIMIT":  {"max_hold_hours": 36},
    "APR_SPIKE_EXIT": {"enable": True, "apr_threshold": 0.90, "min_gain_pct": 0.02},
}
if "SELL_GUARD" in PROTECT:
    SELL_GUARD.update(PROTECT["SELL_GUARD"])

BUY_CFG = {
    "enable": True,
    "whitelist": ["ANON","BIT","AIR"],  # 👈 set your candidates here (or via ADVANCED_JSON)
    "per_trade_usd": 20.0,
    "min_buy_usd": 10.0,
    "max_positions": 5,
    "cooldown_min": 0
}
if "BUY" in ADVANCED:
    BUY_CFG.update(ADVANCED["BUY"])

# =====================================================================
# Position model
# =====================================================================
@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    last_price: float
    usd_value: float
    opened_at: Optional[str]
    apr_estimate: Optional[float] = None
    trailing_max_price: Optional[float] = None
    realized_pnl_usd: float = 0.0
    meta: Dict[str, Any] = None

    def gain_pct(self) -> float:
        if self.avg_price <= 0: return 0.0
        return (self.last_price / self.avg_price) - 1.0

    def age_hours(self) -> Optional[float]:
        if not self.opened_at: return None
        try:
            dt = datetime.fromisoformat(self.opened_at.replace("Z","+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds()/3600.0)
        except Exception:
            return None

# =====================================================================
# Kraken adapter (single source of truth)
# =====================================================================
try:
    from trader.adapters import (
        place_market_sell as kraken_sell,
        place_market_buy  as kraken_buy,
        last_price_usd    as kraken_last_price,
        get_usd_balance   as kraken_usd_balance,
    )
except Exception as e:
    # Hard fail is okay; it will be recorded in artifacts below
    write_text("adapter_import_error.txt", str(e))
    kraken_sell = kraken_buy = kraken_last_price = kraken_usd_balance = None

# =====================================================================
# Positions persistence
# =====================================================================
def load_positions_file() -> List[Position]:
    p = STATE_DIR / "positions.json"
    if not p.exists(): return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[Position] = []
    for r in raw:
        try:
            sym = r["symbol"].upper()
            qty = float(r.get("qty", 0) or 0)
            avg = float(r.get("avg_price", 0) or 0)
            last = float(r.get("last_price", avg))
            opened = r.get("opened_at")
            usd = qty * (last if last > 0 else avg)
            out.append(Position(sym, qty, avg, last, usd, opened,
                                r.get("apr_estimate"), r.get("trailing_max_price"),
                                float(r.get("realized_pnl_usd", 0) or 0), r))
        except Exception:
            continue
    return out

def save_positions_file(positions: List[Position]) -> None:
    write_json("positions.json", [asdict(p) for p in positions])

# =====================================================================
# SELL evaluation
# =====================================================================
def evaluate_sell(p: Position, g: Dict[str, Any]) -> Tuple[bool, str]:
    gain = p.gain_pct()

    # Dust / stable
    if p.usd_value < float(g.get("MIN_SELL_USD", 10)):
        return (False, f"BLOCK_MIN_SELL(<{g.get('MIN_SELL_USD',10)} USD)")
    if p.symbol.upper() in {"USDT","USDC","DAI","USD","USDK","TUSD","FDUSD","USDP"}:
        return (False, "SKIP_STABLE")

    tp = g.get("TP", {})
    if tp.get("enable", True) and gain >= float(tp.get("take_profit_pct", 0.08)):
        return (True, f"TP_HIT({gain:+.2%})")

    sl = g.get("SL", {})
    if sl.get("enable", True) and gain <= float(sl.get("stop_loss_pct", -0.06)):
        return (True, f"STOP_LOSS({gain:+.2%})")

    tsl = g.get("TSL", {})
    if tsl.get("enable", True):
        arm_after = float(tsl.get("arm_after_gain_pct", 0.04))
        trail = float(tsl.get("trail_pct", 0.03))
        peaks_path = STATE_DIR / "trailing_peaks.json"
        peaks = {}
        if peaks_path.exists():
            try: peaks = json.loads(peaks_path.read_text(encoding="utf-8"))
            except: peaks = {}
        peak = float(peaks.get(p.symbol, 0) or 0)
        if p.last_price > peak:
            peaks[p.symbol] = p.last_price
            write_json("trailing_peaks.json", peaks)
        if gain >= arm_after and peak > 0 and p.last_price <= (1.0 - trail) * peak:
            return (True, f"TSL_HIT({gain:+.2%},trail={trail:.2%})")

    apr = g.get("APR_SPIKE_EXIT", {})
    if apr.get("enable", True) and p.apr_estimate is not None:
        if p.apr_estimate >= float(apr.get("apr_threshold", 0.90)) and gain >= float(apr.get("min_gain_pct", 0.02)):
            return (True, f"APR_SPIKE_EXIT(apr={p.apr_estimate:.0%},gain={gain:+.2%})")

    dd = g.get("DIP_CUTOFF", {})
    if gain <= float(dd.get("max_drawdown_pct", -0.09)):
        return (True, f"DRAW_DOWN({gain:+.2%})")

    age_cfg = g.get("AGE_LIMIT", {})
    age = p.age_hours()
    if age is not None and age >= float(age_cfg.get("max_hold_hours", 36)):
        return (True, f"AGE_LIMIT({age:.1f}h)")

    return (False, "HOLD")

# =====================================================================
# BUY planner
# =====================================================================
def plan_buys(positions: List[Position]) -> List[Dict[str, Any]]:
    if not BUY_CFG.get("enable", True):
        return []
    if not kraken_last_price or not kraken_usd_balance:
        write_text("buy_plan_error.txt", "Adapter not loaded; cannot plan buys.")
        return []

    held = {p.symbol for p in positions if p.qty > 0}
    max_pos = int(BUY_CFG.get("max_positions", 5))
    slots = max(0, max_pos - len(held))
    if slots == 0:
        return []

    usd_cash = float(kraken_usd_balance() or 0.0)
    per_trade = float(BUY_CFG.get("per_trade_usd", 20.0))
    min_usd   = float(BUY_CFG.get("min_buy_usd", 10.0))
    budget_trades = int(usd_cash // per_trade)
    trades = min(slots, budget_trades)
    if trades <= 0:
        return []

    wl: List[str] = [s.upper() for s in BUY_CFG.get("whitelist", [])]
    candidates = [s for s in wl if s not in held][:trades]

    plan = []
    for sym in candidates:
        try:
            px = float(kraken_last_price(sym))
        except Exception as e:
            write_text("buy_price_error.txt", f"{sym}: {e}")
            continue
        qty = per_trade / px if px > 0 else 0.0
        usd_val = qty * px
        if usd_val < min_usd or qty <= 0:
            continue
        plan.append({"symbol": sym, "qty": qty, "price": px, "usd_value": usd_val})
    return plan

# =====================================================================
# Execution wrappers
# =====================================================================
def exec_sell(symbol: str, qty: float) -> Dict[str, Any]:
    if not kraken_sell:
        return {"status": "NO_ADAPTER"}
    validate = (DRY_RUN != "OFF")
    try:
        resp = kraken_sell(symbol, qty, validate=validate, reduce_only=False)
        return {"status": "OK", "validated_only": validate, "kraken": resp}
    except Exception as e:
        return {"status": "ERR", "error": str(e)}

def exec_buy(symbol: str, qty: float) -> Dict[str, Any]:
    if not kraken_buy:
        return {"status": "NO_ADAPTER"}
    validate = (DRY_RUN != "OFF")
    try:
        resp = kraken_buy(symbol, qty, validate=validate)
        return {"status": "OK", "validated_only": validate, "kraken": resp}
    except Exception as e:
        return {"status": "ERR", "error": str(e)}

# =====================================================================
# Main
# =====================================================================
def main() -> int:
    write_text("run_header.txt", "\n".join([
        "=== Live Buy+Sell ===",
        f"time_utc: {now_iso()}",
        f"DRY_RUN: {DRY_RUN}",
        f"RUN_SWITCH: {RUN_SWITCH}",
    ]))

    if RUN_SWITCH != "ON":
        write_text("run_skipped.txt", "RUN_SWITCH is OFF, skipping.")
        return 0

    positions = load_positions_file()
    write_json("positions_evaluated.json", [asdict(p) for p in positions])

    # ---- SELL pass ---------------------------------------------------
    sell_rows: List[List[Any]] = []
    sell_exec, sell_reco = [], []
    sell_header = ["ts_utc","symbol","qty","avg_price","last_price","usd_value","gain_pct","age_hours","decision","dry_run"]

    for p in positions:
        # refresh price for better decisions if adapter available
        try:
            if kraken_last_price:
                p.last_price = float(kraken_last_price(p.symbol))
                p.usd_value  = p.qty * p.last_price
        except Exception:
            pass

        should_sell, reason = evaluate_sell(p, SELL_GUARD)
        sell_rows.append([now_iso(), p.symbol, f"{p.qty:.8f}", f"{p.avg_price:.8f}",
                          f"{p.last_price:.8f}", f"{p.usd_value:.2f}",
                          f"{p.gain_pct():+.4%}",
                          f"{(p.age_hours() if p.age_hours() is not None else -1):.2f}",
                          reason, DRY_RUN])

        if should_sell:
            order = {"symbol": p.symbol, "qty": p.qty, "reason": reason, "ts": now_iso(), "action": "SELL"}
            if DRY_RUN != "OFF":
                sell_reco.append(order)
            else:
                order["exec"] = exec_sell(p.symbol, p.qty)
                sell_exec.append(order)

    append_csv("sell_decisions.csv", sell_header, sell_rows)
    write_json("sell_orders_recommended.json", sell_reco)
    write_json("sell_orders_executed.json", sell_exec)

    # If live sells happened, trim positions snapshot for convenience
    if DRY_RUN == "OFF" and sell_exec:
        newpos = [p for p in positions if p.symbol not in {o["symbol"] for o in sell_exec}]
        positions = newpos

    # ---- BUY pass ----------------------------------------------------
    buy_plan = plan_buys(positions)
    write_json("buy_plan.json", buy_plan)

    buy_rows: List[List[Any]] = []
    buy_exec, buy_reco = [], []
    buy_header = ["ts_utc","symbol","qty","est_price","usd_value","decision","dry_run"]

    for item in buy_plan:
        sym, qty, px, usd_val = item["symbol"], float(item["qty"]), float(item["price"]), float(item["usd_value"])
        decision = "BUY"
        buy_rows.append([now_iso(), sym, f"{qty:.8f}", f"{px:.8f}", f"{usd_val:.2f}", decision, DRY_RUN])

        if DRY_RUN != "OFF":
            buy_reco.append({"symbol": sym, "qty": qty, "price": px, "ts": now_iso(), "action": "BUY"})
        else:
            order = {"symbol": sym, "qty": qty, "price": px, "ts": now_iso(), "action": "BUY"}
            order["exec"] = exec_buy(sym, qty)
            buy_exec.append(order)
            # Update positions snapshot immediately with estimated entry
            positions.append(Position(symbol=sym, qty=qty, avg_price=px, last_price=px,
                                      usd_value=qty*px, opened_at=now_iso()))

    append_csv("buy_decisions.csv", buy_header, buy_rows)
    write_json("buy_orders_recommended.json", buy_reco)
    write_json("buy_orders_executed.json", buy_exec)

    # Persist snapshot for next run
    save_positions_file(positions)

    summary = {
        "time_utc": now_iso(),
        "dry_run": DRY_RUN,
        "positions_checked": len(sell_rows),
        "orders_recommended_sell": len(sell_reco),
        "orders_executed_sell": len(sell_exec),
        "orders_recommended_buy": len(buy_reco),
        "orders_executed_buy": len(buy_exec),
    }
    write_json("run_summary.json", summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
