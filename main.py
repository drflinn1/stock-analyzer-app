#!/usr/bin/env python3
"""
Crypto — Hourly 1-Coin Rotation
Hard gate for 1 open position + auto-sell rules + robust .state persistence.

Rules:
- MAX_POSITIONS=1 is enforced by the presence of .state/position.json.
- Exits: SL_PCT (default 1%), TP_PCT (default 5%), ROTATE_MINUTES timer with
  ROTATE_MIN_GAIN_PCT (default 3%).
- Always writes .state/run_summary.json/.md each run so you have artifacts.
- After a LIVE buy, writes .state/position.json. After a LIVE sell, deletes it.
"""

import json, os, time, datetime as dt
from pathlib import Path

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
POS_FILE = STATE / "position.json"
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD   = STATE / "run_summary.md"

def env_str(k, default=""):
    v = os.getenv(k, default)
    return "" if v is None else str(v)

def env_int(k, default=0):
    try: return int(env_str(k, str(default)).strip())
    except: return default

def env_float(k, default=0.0):
    try: return float(env_str(k, str(default)).strip())
    except: return default

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text())
    return None

def write_summary(status: str, details: dict):
    payload = {
        "when": now_iso(),
        "mode": "LIVE" if env_str("DRY_RUN","OFF").upper()=="OFF" else "DRY_RUN",
        "status": status,
        **details
    }
    save_json(SUMMARY_JSON, payload)
    # minimal markdown for artifact streams
    lines = [
        f"**When:** {payload['when']}",
        f"**Mode:** {payload['mode']}",
        f"**Status:** {status}",
        "",
        "```json",
        json.dumps(details, indent=2),
        "```",
    ]
    SUMMARY_MD.write_text("\n".join(lines))

# --- Broker shims (use your existing buy/sell helpers behind these) ----
# IMPORTANT: Keep the guts you already have for Kraken. If your repo already
# has real functions like live_buy(), live_sell(), get_last_price(), swap
# these 3 functions to call into them. The rotation gate logic below will work
# either way.

def live_buy(symbol: str, usd_amount: float) -> dict:
    """
    Do your real Kraken market buy here.
    Return a dict incl. fields: {'symbol','qty','avg_price','txid'}
    """
    # >>> call your existing live buy function <<<
    # For safety, keep keys consistent:
    raise NotImplementedError("Wire this to your existing Kraken BUY call.")

def live_sell(symbol: str, qty: float) -> dict:
    """
    Do your real Kraken market sell here.
    Return a dict incl. fields: {'symbol','qty','avg_price','txid'}
    """
    # >>> call your existing live sell function <<<
    raise NotImplementedError("Wire this to your existing Kraken SELL call.")

def get_last_price(symbol: str) -> float:
    """
    Return latest trade/mark price for `symbol` (e.g., ANON/USD).
    """
    # >>> call your existing price fetcher <<<
    raise NotImplementedError("Wire this to your price/ticker fetcher.")

# --- Rotation helpers ----------------------------------------------------

def have_open_position() -> bool:
    return POS_FILE.exists()

def pct_change(from_price: float, to_price: float) -> float:
    if from_price <= 0: return 0.0
    return (to_price - from_price) / from_price * 100.0

def load_position():
    return load_json(POS_FILE)

def save_position(symbol: str, qty: float, entry_price: float):
    save_json(POS_FILE, {
        "symbol": symbol,
        "qty": qty,
        "entry_price": entry_price,
        "opened_at": time.time()
    })

def clear_position():
    if POS_FILE.exists():
        POS_FILE.unlink()

def select_top_candidate() -> str:
    """
    Your ranker already picks a top symbol. If you write it to
    .state/momentum_candidates.csv or run_summary, read it here.
    For now we fallback to env UNIVERSE_PICK for testing.
    """
    pick = env_str("UNIVERSE_PICK", "").strip()
    if not pick:
        raise RuntimeError("No candidate found. Set UNIVERSE_PICK temporarily or wire ranker output here.")
    return pick

# --- Main ---------------------------------------------------------------

def main():
    dry_run = env_str("DRY_RUN","OFF").upper() != "OFF"
    usd_per_buy   = env_float("BUY_USD", 25.0)
    max_positions = env_int("MAX_POSITIONS", 1)
    tp_pct        = env_float("TP_PCT", 5.0)
    sl_pct        = env_float("SL_PCT", 1.0)
    rotate_min    = env_int("ROTATE_MINUTES", 60)
    rotate_gain   = env_float("ROTATE_MIN_GAIN_PCT", 3.0)

    # Always write a start summary stub
    write_summary("start", {
        "DRY_RUN": not not dry_run,
        "max_positions": max_positions,
        "tp_pct": tp_pct, "sl_pct": sl_pct,
        "rotate_minutes": rotate_min, "rotate_gain_pct": rotate_gain
    })

    # ---- If a position exists, enforce exit rules and skip new buys ----
    if have_open_position():
        pos = load_position()
        sym = pos["symbol"]; qty = float(pos["qty"])
        entry = float(pos["entry_price"]); opened = float(pos["opened_at"])
        price = get_last_price(sym)
        change = pct_change(entry, price)
        minutes = (time.time() - opened) / 60.0

        decision = "hold"
        reason   = ""

        if change <= -abs(sl_pct):
            decision, reason = "sell", f"SL hit ({change:.2f}%)"
        elif change >= abs(tp_pct):
            decision, reason = "sell", f"TP hit (+{change:.2f}%)"
        elif minutes >= rotate_min and change < abs(rotate_gain):
            decision, reason = "sell", f"Timer {rotate_min}m & gain {change:.2f}% < {rotate_gain}%"

        if decision == "sell":
            if dry_run:
                write_summary("DRY SELL", {"symbol": sym, "qty": qty, "price": price, "reason": reason})
                clear_position()
            else:
                res = live_sell(sym, qty)
                write_summary("LIVE SELL OK", {"symbol": sym, "qty": qty, "avg_price": res.get("avg_price"), "txid": res.get("txid"), "reason": reason})
                clear_position()
        else:
            write_summary("HOLD", {"symbol": sym, "qty": qty, "price": price, "change_pct": change, "minutes_open": minutes})
        return

    # ---- No open position: enforce MAX_POSITIONS before BUY ------------
    # This extra guard ensures we NEVER buy if a stale run forgot to clear state.
    if max_positions <= 0:
        write_summary("SKIP", {"reason": "MAX_POSITIONS<=0"})
        return
    if max_positions == 1 and POS_FILE.exists():
        write_summary("SKIP", {"reason": "position.json present; MAX_POSITIONS=1"})
        return

    # Choose leader & BUY
    try:
        pick = select_top_candidate()
    except Exception as e:
        write_summary("SKIP", {"reason": f"No candidate: {e}"})
        return

    if dry_run:
        # Mock execution:
        entry = get_last_price(pick)
        qty = (usd_per_buy / entry) if entry > 0 else 0.0
        save_position(pick, qty, entry)
        write_summary("DRY BUY", {"symbol": pick, "qty": qty, "entry_price": entry})
    else:
        res = live_buy(pick, usd_per_buy)   # must return symbol, qty, avg_price, txid
        save_position(res["symbol"], float(res["qty"]), float(res["avg_price"]))
        write_summary("LIVE BUY OK", {"symbol": res["symbol"], "qty": res["qty"], "avg_price": res["avg_price"], "txid": res.get("txid")})

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        # Helpful failure until you wire broker/price functions above.
        write_summary("ERROR", {"need_to_wire": str(e)})
        raise
    except Exception as e:
        write_summary("ERROR", {"exception": repr(e)})
        raise
