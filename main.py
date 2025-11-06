#!/usr/bin/env python3
"""
main.py — Rotation Edition
- Honors UNIVERSE_PICK if set (e.g., SOLUSD), otherwise scans "Gainers".
- Gainers priority: .state/momentum_candidates.csv (if present) → ccxt fetchTickers() USD pairs 24h % change.
- Implements rotation rules:
    * If within 60 min and P&L <= -1% → SELL (rotate)
    * If P&L >= +5% at any time → SELL (rotate)
    * If >= 60 min and P&L < +3% → SELL (rotate)
- Keeps exactly one open position at a time.
- Always writes .state/run_summary.{json,md}, positions.json, last_ok.txt
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# ---------- Paths ----------
STATE = Path(".state")
STATE.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD = STATE / "run_summary.md"
POSITIONS_JSON = STATE / "positions.json"
LAST_OK = STATE / "last_ok.txt"
CANDIDATES_CSV = STATE / "momentum_candidates.csv"

# ---------- Env helpers ----------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return "" if v is None else str(v)

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)).strip())
    except Exception:
        return default

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_iso(dt: Optional[datetime]=None) -> str:
    return (dt or utc_now()).isoformat(timespec="seconds")

# ---------- IO helpers ----------
def write_summary(data: Dict[str, Any]) -> None:
    try:
        SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[WARN] Failed writing {SUMMARY_JSON}: {e}", file=sys.stderr)

    try:
        lines = [
            f"**When:** {data.get('when','')}",
            f"**Live (DRY_RUN=OFF):** {data.get('live', False)}",
            f"**Pick Source:** {data.get('pick_source','')}",
            f"**Pick (symbol):** {data.get('symbol','')}",
            f"**BUY_USD:** {data.get('buy_usd','')}",
            f"**Status:** {data.get('status','')}",
            f"**Note:** {data.get('note','')}",
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

# ---------- Symbol mapping ----------
def to_ccxt_symbol(universe_pick: str) -> Optional[str]:
    up = (universe_pick or "").strip().upper()
    if not up.endswith("USD") or len(up) <= 3:
        return None
    base = up[:-3]
    return f"{base}/USD"

def from_ccxt_symbol(symbol: str) -> str:
    # "SOL/USD" -> "SOLUSD"
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return f"{base}{quote}"
    return symbol.replace("/", "")

# ---------- Candidate reading ----------
def read_candidates_csv() -> List[str]:
    """Return list of ccxt symbols like 'SOL/USD' from .state/momentum_candidates.csv."""
    if not CANDIDATES_CSV.exists():
        return []
    out: List[str] = []
    try:
        with CANDIDATES_CSV.open("r", newline="") as f:
            reader = csv.DictReader(f)
            # accept columns: symbol or pair, with/without slash
            for row in reader:
                raw = (row.get("symbol") or row.get("pair") or "").strip()
                if not raw:
                    continue
                s = raw.upper().replace(" ", "")
                if "/" not in s and s.endswith("USD") and len(s) > 3:
                    s = f"{s[:-3]}/USD"
                if s.endswith("/USD"):
                    out.append(s)
    except Exception as e:
        print(f"[WARN] Failed reading candidates CSV: {e}", file=sys.stderr)
    return out

# ---------- Gainers from ccxt ----------
def pick_top_gainer_ccxt(exchange, limit: int = 10) -> Optional[str]:
    """
    Fetch tickers, filter to */USD spot markets, rank by 24h % change.
    Returns the top symbol like 'SOL/USD', or None.
    """
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        print(f"[WARN] fetch_tickers failed: {e}", file=sys.stderr)
        return None

    ranked: List[Tuple[str, float]] = []
    for sym, t in tickers.items():
        if not sym.endswith("/USD"):
            continue
        # Try to get 24h percentage change
        pct = None
        # ccxt standard field
        if isinstance(t, dict) and t.get("percentage") is not None:
            try:
                pct = float(t["percentage"])
            except Exception:
                pct = None
        # Fallback (last/open) if available
        if pct is None:
            try:
                last = float(t.get("last") or 0.0)
                open_ = float(t.get("open") or 0.0)
                if last > 0 and open_ > 0:
                    pct = ((last / open_) - 1.0) * 100.0
            except Exception:
                pct = None
        if pct is None:
            continue
        ranked.append((sym, pct))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[0][0]
    print(f"[SCAN] Top gainer via ccxt: {top}")
    return top

# ---------- Trading ----------
def connect_exchange(kraken_key: str, kraken_secret: str):
    import ccxt
    ex = ccxt.kraken({
        "apiKey": kraken_key,
        "secret": kraken_secret,
        "enableRateLimit": True,
    })
    ex.load_markets()
    return ex

def fetch_last_price(exchange, symbol: str) -> Optional[float]:
    try:
        t = exchange.fetch_ticker(symbol)
        p = t.get("last")
        return float(p) if p is not None else None
    except Exception as e:
        print(f"[WARN] fetch_ticker({symbol}) failed: {e}", file=sys.stderr)
        return None

def spend_usd_market_buy(exchange, symbol: str, usd: float) -> Dict[str, Any]:
    print(f"[LIVE] BUY {symbol} — cost ${usd} (market)")
    order = exchange.create_order(symbol, "market", "buy", None, None, {"cost": usd})
    print("[LIVE] BUY order response:", order)
    return order

def market_sell_all(exchange, symbol: str, amount: float) -> Dict[str, Any]:
    print(f"[LIVE] SELL {symbol} — amount {amount} (market)")
    order = exchange.create_order(symbol, "market", "sell", amount)
    print("[LIVE] SELL order response:", order)
    return order

def base_free_balance(exchange, symbol: str) -> float:
    """Return free balance of the base asset for the symbol."""
    base = symbol.split("/")[0]
    bal = exchange.fetch_balance()
    amt = float(bal.get(base, {}).get("free") or 0.0)
    return amt

# ---------- Rotation decision ----------
def should_sell(entry_price: float, entry_time_iso: str,
                now_price: float,
                tp_pct: float, stop_pct: float, slow_min: float, window_min: int) -> Tuple[bool, str, float]:
    """
    Returns: (do_sell, reason, pnl_pct)
    """
    try:
        entry_time = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
    except Exception:
        entry_time = utc_now()

    age_min = (utc_now() - entry_time).total_seconds() / 60.0
    pnl_pct = ((now_price / entry_price) - 1.0) * 100.0

    # 1) TP any time
    if pnl_pct >= tp_pct:
        return True, f"TP hit (>= {tp_pct:.2f}%)", pnl_pct
    # 2) Tight stop inside first hour
    if age_min <= window_min and pnl_pct <= -abs(stop_pct):
        return True, f"Early stop (<= -{abs(stop_pct):.2f}%) within {window_min}m", pnl_pct
    # 3) Slow rotate after window if not meeting min gain
    if age_min >= window_min and pnl_pct < slow_min:
        return True, f"Slow rotate after {window_min}m (< {slow_min:.2f}%)", pnl_pct

    return False, "Hold", pnl_pct

# ---------- Main ----------
def main() -> int:
    when = utc_iso()

    # Env / thresholds
    dry_run = env_str("DRY_RUN", "ON").upper()
    live = (dry_run == "OFF")
    buy_usd = env_float("BUY_USD", 25.0)
    reserve_cash_pct = env_float("RESERVE_CASH_PCT", 0.0)
    universe_pick = env_str("UNIVERSE_PICK", "").upper().strip()

    # Rotation thresholds (customizable)
    TP_PCT = env_float("TP_PCT", 5.0)
    STOP_PCT = env_float("STOP_PCT", 1.0)         # early stop within first hour
    SLOW_MIN_PCT = env_float("SLOW_MIN_PCT", 3.0) # required gain after 60m
    WINDOW_MIN = int(env_float("WINDOW_MIN", 60)) # minutes

    # Secrets
    K = env_str("KRAKEN_API_KEY", "")
    S = env_str("KRAKEN_API_SECRET", "")

    result: Dict[str, Any] = {
        "when": when,
        "live": live,
        "buy_usd": buy_usd,
        "reserve_cash_pct": reserve_cash_pct,
        "universe_pick": universe_pick,
        "thresholds": {
            "tp_pct": TP_PCT,
            "stop_pct": STOP_PCT,
            "slow_min_pct": SLOW_MIN_PCT,
            "window_min": WINDOW_MIN,
        },
        "status": "START",
        "note": "",
        "symbol": "",
        "pick_source": "",
    }

    print("[ENV] DRY_RUN =", dry_run)
    print("[ENV] BUY_USD =", buy_usd)
    print("[ENV] UNIVERSE_PICK =", universe_pick)
    print("[ENV] TP/STOP/SLOW/WINDOW =", TP_PCT, STOP_PCT, SLOW_MIN_PCT, WINDOW_MIN)

    # Always load positions (one position max)
    positions = load_positions()
    open_symbols = list(positions.keys())
    open_symbol = open_symbols[0] if open_symbols else None

    if not live:
        result["status"] = "RISK_OFF"
        result["note"] = "DRY_RUN is ON; evaluation only."
        result["open_position"] = positions
        write_summary(result)
        return 0

    # Live mode needs secrets
    if not K or not S:
        result["status"] = "ERROR_NO_SECRETS"
        result["note"] = "Missing Kraken API secrets."
        write_summary(result)
        return 0

    # Connect to Kraken
    try:
        exchange = connect_exchange(K, S)
    except Exception as e:
        result["status"] = "ERROR_CONNECT"
        result["note"] = f"Connect failed: {e}"
        write_summary(result)
        return 0

    # ---------- If we have an open position, evaluate rotate/sell ----------
    if open_symbol:
        sym = open_symbol
        pos = positions[sym]
        entry_price = float(pos.get("entry_price") or 0.0)
        entry_time = pos.get("entry_time") or when

        last = fetch_last_price(exchange, sym)
        if not last or last <= 0:
            result["status"] = "ERROR_PRICE"
            result["symbol"] = sym
            result["note"] = f"No price for {sym}"
            result["open_position"] = positions
            write_summary(result)
            return 0

        do_sell, reason, pnl_pct = should_sell(entry_price, entry_time, last,
                                               TP_PCT, STOP_PCT, SLOW_MIN_PCT, WINDOW_MIN)
        result["symbol"] = sym
        result["pnl_pct"] = pnl_pct
        result["decision"] = reason

        if do_sell:
            if live:
                try:
                    amt = pos.get("amount")
                    if not amt or float(amt) <= 0:
                        amt = base_free_balance(exchange, sym)
                    amt = float(amt)
                    sell_order = market_sell_all(exchange, sym, amt)
                    result["status"] = "LIVE_SELL_OK"
                    result["sell_order"] = sell_order
                    # clear position
                    positions.pop(sym, None)
                    save_positions(positions)
                except Exception as e:
                    result["status"] = "LIVE_SELL_ERROR"
                    result["note"] = f"Sell error: {e}"
                    result["open_position"] = positions
                    write_summary(result)
                    return 0
            # After selling, we will attempt to buy a new pick (rotate)
        else:
            result["status"] = "HOLD"
            result["open_position"] = positions
            write_summary(result)
            try:
                LAST_OK.write_text(utc_iso()+"\n")
            except Exception:
                pass
            return 0

    # ---------- If no open position (or we just sold), pick next coin ----------
    pick_symbol: Optional[str] = None
    pick_source = ""

    # 0) If override is set, use it
    if universe_pick:
        pick_symbol = to_ccxt_symbol(universe_pick)
        pick_source = "UNIVERSE_PICK"
    # 1) Try candidates CSV
    if not pick_symbol:
        cands = read_candidates_csv()
        if cands:
            pick_symbol = cands[0]
            pick_source = "CANDIDATES_CSV"
    # 2) Fallback: scan ccxt gainers
    if not pick_symbol:
        ps = pick_top_gainer_ccxt(exchange)
        if ps:
            pick_symbol = ps
            pick_source = "CCXT_TOP_GAINER"

    if not pick_symbol:
        result["status"] = "NO_PICK"
        result["note"] = "No gainer found (scanner and fallback both empty)."
        write_summary(result)
        return 0

    result["symbol"] = pick_symbol
    result["pick_source"] = pick_source

    # Place the BUY
    try:
        order = spend_usd_market_buy(exchange, pick_symbol, buy_usd)
        # best-effort price and amount
        entry_price = None
        amount = None
        try:
            if order and isinstance(order, dict):
                entry_price = float(order.get("price") or 0) or None
                amount = float(order.get("amount") or 0) or None
        except Exception:
            pass
        if amount is None:
            # fallback to balance
            amount = base_free_balance(exchange, pick_symbol)

        positions = load_positions()
        positions[pick_symbol] = {
            "symbol": pick_symbol,
            "entry_time": utc_iso(),
            "entry_price": entry_price,
            "buy_usd": buy_usd,
            "amount": amount,
            "source": pick_source,
        }
        save_positions(positions)

        result["status"] = "LIVE_BUY_OK"
        result["order"] = order
        result["entry_price"] = entry_price
        result["amount"] = amount
    except Exception as e:
        msg = str(e)
        result["status"] = "LIVE_BUY_ERROR"
        if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
            result["status"] = "ERROR_FUNDS"
        elif "Minimum order" in msg:
            result["status"] = "ERROR_MIN_ORDER"
        result["note"] = msg

    # Heartbeat
    try:
        LAST_OK.write_text(utc_iso()+"\n")
    except Exception:
        pass

    write_summary(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
