#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
- Continuous STOP guard (no longer limited to the first window)
- Keeps TP/slow-exit rules
- Single-position auto-guard
- Writes .state/position.json and .state/positions.json + run_summary.{json,md}
"""

from __future__ import annotations
import json, os, time, datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

STATE = Path(".state"); STATE.mkdir(exist_ok=True)
POS_JSON = STATE/"position.json"       # legacy single slot
POSS_JSON = STATE/"positions.json"     # multi-slot for safety
SUM_JSON = STATE/"run_summary.json"
SUM_MD   = STATE/"run_summary.md"
LAST_OK  = STATE/"last_ok.txt"

def env(name:str, default:str=""): 
    v = os.getenv(name, default)
    return "" if v is None else str(v)

def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"+00:00"

# ---------- config from env (same names you already use) ----------
LIVE  = (env("DRY_RUN","ON").upper() == "OFF")
BUY_USD = float(env("BUY_USD","15"))
RESERVE = float(env("RESERVE_CASH_PCT","0"))
TP_PCT  = float(env("TP_PCT","5"))/100.0        # +5%
STOP_PCT= float(env("STOP_PCT","1"))/100.0      # -1% (continuous now)
SLOW_MIN= float(env("SLOW_MIN_PCT","3"))/100.0  # +3% in a window
WINDOW  = int(env("WINDOW_MIN","30"))           # minutes (for slow exit timing only)
PICK    = env("UNIVERSE_PICK","").upper().strip()  # optional manual symbol like "UAIUSD" or "UAI/USD"
DUST_MIN_USD = float(env("DUST_MIN_USD","2"))

# ----------- tiny kraken adapter (expects same env secrets you already set) -----------
from trader.kraken import Kraken  # your existing adapter
K = Kraken(live=LIVE)

def sym_norm(s: str) -> str:
    s = s.upper().replace(" ", "")
    return s if "/" in s else (s.replace("USD","") + "/USD") if s.endswith("USD") else (s + "/USD")

def price_usd(pair: str) -> Optional[float]:
    q = K.quote(pair)  # should return dict with last or ask; your adapter already provides this
    if not q: return None
    return float(q.get("price") or q.get("last") or q.get("ask") or 0) or None

# ----------- state helpers -----------
def load_positions() -> Dict[str, Any]:
    if POSS_JSON.exists():
        return json.loads(POSS_JSON.read_text() or "{}")
    # migrate from legacy single slot if present
    d = {}
    if POS_JSON.exists():
        try:
            p = json.loads(POS_JSON.read_text() or "{}")
            if p.get("symbol"):
                d[p["symbol"]] = {
                    "symbol": p["symbol"],
                    "entry_time": p.get("in_position_since"),
                    "entry_price": p.get("avg_price") or None,
                    "buy_usd": BUY_USD,
                    "amount": p.get("qty") or None,
                    "source": "LEGACY"
                }
        except Exception: pass
    return d

def save_positions(d: Dict[str,Any]) -> None:
    POSS_JSON.write_text(json.dumps(d, indent=2))

def write_summary(data: Dict[str,Any]) -> None:
    SUM_JSON.write_text(json.dumps(data, indent=2))
    md = [
      f"**When:** {data['when']}",
      f"**Live (DRY_RUN=OFF):** {data['live']}",
      f"**Pick Source:** {data.get('pick_source','')}",
      f"**Pick (symbol):** {data.get('symbol','')}",
      f"**BUY_USD:** {data.get('buy_usd')}",
      f"**Status:** {data['status']}",
      f"**Note:** {data.get('note','')}",
      "",
      "### Details",
      "```json",
      json.dumps(data, indent=2),
      "```",
    ]
    SUM_MD.write_text("\n".join(md))
    LAST_OK.write_text(now_utc())

# ----------- core logic -----------
def should_sell_continuous(entry_price: float, last: float) -> bool:
    if not entry_price or not last: return False
    drop = (last - entry_price) / entry_price
    return drop <= -STOP_PCT   # continuous stop

def should_take_profit(entry_price: float, last: float) -> bool:
    if not entry_price or not last: return False
    gain = (last - entry_price) / entry_price
    return gain >= TP_PCT

def should_slow_exit(entry_ts: str, entry_price: float, last: float) -> bool:
    if not entry_ts or not entry_price or not last: return False
    mins = (dt.datetime.utcnow() - dt.datetime.fromisoformat(entry_ts.replace("Z","+00:00")[:19])).total_seconds()/60.0
    if mins < WINDOW: return False
    gain = (last - entry_price) / entry_price
    return gain >= SLOW_MIN

def open_positions_from_exchange() -> Dict[str,float]:
    # returns { "UAI/USD": amount, ... } for spot balances
    return K.open_positions()  # your adapter already used earlier to build the single-position guard

def sell_all_but_newest(pos: Dict[str,Any]) -> list[dict]:
    if len(pos) <= 1: return []
    # sort by entry_time; keep the newest, sell older ones
    sells = []
    newest = max(pos.values(), key=lambda r: r.get("entry_time") or "")
    for sym, rec in list(pos.items()):
        if rec is newest: 
            continue
        sells.append({"symbol": sym, "reason": "AUTO_TRIM_TO_SINGLE"})
    return sells

def place_buy(symbol: str, usd: float) -> dict:
    # market buy by USD
    return K.market_buy(symbol, usd)

def place_sell(symbol: str, amount: float) -> dict:
    # try market; if price-protection blocks, adapter should retry with a tight limit
    return K.smart_sell(symbol, amount)

def run() -> None:
    summary = {
        "when": now_utc(),
        "live": LIVE,
        "buy_usd": BUY_USD,
        "reserve_cash_pct": RESERVE,
        "universe_pick": PICK,
        "thresholds": {"tp_pct": TP_PCT*100, "stop_pct": STOP_PCT*100, "slow_min_pct": SLOW_MIN*100, "window_min": WINDOW},
        "status": "",
        "note": "",
        "symbol": "",
        "pick_source": "",
        "order": {},
    }

    # sync & single-position guard
    ex_pos = open_positions_from_exchange()              # live view
    state_pos = load_positions()                         # remembered entries
    # add unknown exchange coins into state if missing
    for sym, amt in ex_pos.items():
        if amt > 0 and sym not in state_pos:
            p = price_usd(sym)
            state_pos[sym] = {"symbol": sym, "entry_time": now_utc(), "entry_price": p, "buy_usd": None, "amount": amt, "source": "DISCOVERED"}
    save_positions(state_pos)

    # enforce single-position
    for sell in sell_all_but_newest(state_pos):
        sym = sell["symbol"]; amt = ex_pos.get(sym, 0)
        if amt > 0:
            place_sell(sym, amt)

    # evaluate exits (continuous STOP, TP, SLOW)
    for sym, rec in list(state_pos.items()):
        amt = ex_pos.get(sym, 0.0)
        if amt <= 0:
            state_pos.pop(sym, None)
            continue
        last = price_usd(sym)
        if last is None: 
            continue
        if should_sell_continuous(rec.get("entry_price") or last, last) or \
           should_take_profit(rec.get("entry_price") or last, last) or \
           should_slow_exit(rec.get("entry_time"), rec.get("entry_price") or last, last):
            place_sell(sym, amt)
            state_pos.pop(sym, None)

    save_positions(state_pos)

    # entry (only if we hold nothing after the trimming)
    ex_pos = open_positions_from_exchange()
    holding = sum(1 for a in ex_pos.values() if a > 0)
    if holding == 0 and BUY_USD >= DUST_MIN_USD:
        if PICK:
            target = sym_norm(PICK)
            summary["pick_source"]="UNIVERSE_PICK"
            summary["symbol"]=target
        else:
            # your existing candidate loader
            from tools.momentum_candidates import pick_best  # your module
            target = pick_best()  # returns e.g. "UAI/USD" or None
            if not target:
                summary.update(status="RISK_OFF", note="No candidate")
                write_summary(summary); return
            summary["pick_source"]="CANDIDATES_CSV"
            summary["symbol"]=target

        od = place_buy(target, BUY_USD)
        # record entry for continuous stop
        p = price_usd(target)
        state_pos = load_positions()
        state_pos[target] = {
          "symbol": target, "entry_time": now_utc(),
          "entry_price": p, "buy_usd": BUY_USD,
          "amount": od.get("filled") or od.get("amount"), "source": summary["pick_source"]
        }
        save_positions(state_pos)
        summary.update(status="LIVE_BUY_OK", order=od, note=f"buy {BUY_USD} {summary['symbol']} @ market")
        write_summary(summary); return

    summary.update(status="HOLD_OR_EXIT_CHECKED", note=f"positions={list(ex_pos.keys())}")
    write_summary(summary)

if __name__ == "__main__":
    run()
