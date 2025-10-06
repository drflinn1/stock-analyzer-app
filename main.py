# main.py — Kraken Live Bot (Min-lot dust sweep + Restricted auto-blacklist + TP/SL/Trail)
from __future__ import annotations
import os, sys, json, csv, math, time, traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

# ---------- small utils ----------
def getenv_any(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

def as_bool(v: Optional[str], default: bool=False) -> bool:
    if v is None: return default
    s = str(v).strip().lower()
    if s in ("1","true","yes","on"): return True
    if s in ("0","false","no","off"): return False
    return default

def as_float(v: Optional[str], default: float) -> float:
    try: return float(v)
    except: return default

def as_int(v: Optional[str], default: int) -> int:
    try: return int(float(v))
    except: return default

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def log(tag: str, msg: str) -> None:
    color = {"OK":"\033[92m","WARN":"\033[93m","ERR":"\033[91m","KPI":"\033[96m","SUM":"\033[95m"}.get(tag,"")
    end = "\033[0m" if color else ""
    print(f"{color}[{tag}]{end} {msg}", flush=True)

# ---------- ENV ----------
DRY_RUN  = as_bool(os.getenv("DRY_RUN"), True)
RUN_SWITCH = getenv_any("RUN_SWITCH", default="on").lower()=="on"

EXCHANGE   = getenv_any("EXCHANGE","kraken").lower()
BASE_QUOTE = getenv_any("BASE_QUOTE","USD").upper()

MAX_POSITIONS = as_int(os.getenv("MAX_POSITIONS"), 12)
MAX_BUYS_PER_RUN = as_int(os.getenv("MAX_BUYS_PER_RUN"), 1)
ROTATE_WHEN_FULL = as_bool(os.getenv("ROTATE_WHEN_FULL"), True)
ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)

MIN_NOTIONAL_USD = as_float(os.getenv("MIN_NOTIONAL_USD"), 5.0)
DUST_MIN_USD     = as_float(os.getenv("DUST_MIN_USD"), 2.0)
RESERVE_CASH_PCT = as_float(os.getenv("RESERVE_CASH_PCT"), 5.0)

TP_PCT     = as_float(os.getenv("TP_PCT"), 2.0)
SL_PCT     = as_float(os.getenv("SL_PCT"), 3.5)
TRAIL_PCT  = as_float(os.getenv("TRAIL_PCT"), 1.2)

DAILY_LOSS_CAP = as_float(os.getenv("DAILY_LOSS_CAP"), 3.0)
AUTO_PAUSE_ON_ERROR = as_bool(os.getenv("AUTO_PAUSE_ON_ERROR"), True)

LOG_LEVEL    = getenv_any("LOG_LEVEL","INFO")
STATE_DIR    = getenv_any("STATE_DIR",".state")
KPI_CSV_PATH = getenv_any("KPI_CSV_PATH",".state/kpi_history.csv")
POS_STATE    = os.path.join(STATE_DIR,"pos_state.json")
REJECT_LIST  = os.path.join(STATE_DIR,"reject_list.json")  # for restricted symbols
REJECT_TTL_DAYS = as_int(os.getenv("REJECT_TTL_DAYS"), 14)

# optional tight universe: "BTC/USD,ETH/USD,SOL/USD,DOGE/USD"
UNIVERSE_WHITELIST = [s.strip().upper() for s in getenv_any("UNIVERSE_WHITELIST","").split(",") if s.strip()]

SHOW_BANNER  = as_bool(os.getenv("SHOW_BANNER"), True)
TOPK         = as_int(os.getenv("TOPK"), 6)

# ---------- exchange init ----------
def make_exchange() -> ccxt.Exchange:
    if EXCHANGE!="kraken":
        raise SystemExit("Only Kraken supported in this build.")
    api_key = getenv_any("KRAKEN_API_KEY","KRAKEN_KEY","KRAKEN_API")
    api_sec = getenv_any("KRAKEN_API_SECRET","KRAKEN_SECRET","KRAKEN_APISECRET")
    api_otp = getenv_any("KRAKEN_API_OTP","KRAKEN_OTP","KRAKEN_TOTP")
    if not api_key or not api_sec:
        raise SystemExit("Kraken credentials missing (KRAKEN_API_KEY / KRAKEN_API_SECRET).")
    kwargs = {"apiKey": api_key, "secret": api_sec}
    if api_otp: kwargs["password"] = api_otp
    return ccxt.kraken(kwargs)

# ---------- files ----------
def ensure_state(): os.makedirs(STATE_DIR, exist_ok=True)

def load_json(path: str, default: Any) -> Any:
    ensure_state()
    if not os.path.exists(path): return default
    try:
        with open(path,"r",encoding="utf-8") as f: return json.load(f)
    except: return default

def save_json(path: str, data: Any) -> None:
    ensure_state()
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)

def append_kpi(row: Dict[str,Any]) -> None:
    ensure_state()
    new = not os.path.exists(KPI_CSV_PATH)
    with open(KPI_CSV_PATH,"a",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts","dry_run","positions","cash_usd","equity_usd","pnl_day_pct","buys","sells","rotations"])
        if new: w.writeheader()
        w.writerow(row)

def write_summary(lines: List[str]) -> None:
    ensure_state()
    with open(os.path.join(STATE_DIR,"summary.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------- universe filters ----------
STABLES = {"USDT","USDC"}
BLOCKLIST = set(["SPX/USD","EUR/USD","GBP/USD","USD/USD"])

def is_valid_symbol(s: str) -> bool:
    s = s.upper()
    if not s.endswith(f"/{BASE_QUOTE}"): return False
    if s in BLOCKLIST: return False
    base = s.split("/")[0]
    if base in STABLES: return False
    if UNIVERSE_WHITELIST and s not in UNIVERSE_WHITELIST: return False
    return True

# ---------- portfolio ----------
def fetch_cash_positions(ex: ccxt.Exchange) -> Tuple[float, Dict[str,Dict[str,float]]]:
    bal = ex.fetch_balance()
    cash = float(bal.get("total",{}).get(BASE_QUOTE,0.0))
    pos: Dict[str,Dict[str,float]] = {}
    for coin, amt in bal.get("total",{}).items():
        u = coin.upper()
        if u in (BASE_QUOTE, "USDT","USDC"): continue
        try: qty = float(amt)
        except: qty = 0.0
        if qty <= 0: continue
        sym = f"{u}/{BASE_QUOTE}"
        if not is_valid_symbol(sym): continue
        try:
            price = float(ex.fetch_ticker(sym).get("last") or 0.0)
        except: price = 0.0
        pos[sym] = {"amount": qty, "price": price, "value": qty*price}
    return cash, pos

def fetch_topk(ex: ccxt.Exchange, k: int) -> List[Tuple[str,float]]:
    markets = ex.load_markets()
    cands = [s for s in markets.keys() if is_valid_symbol(s)]
    scored: List[Tuple[str,float]] = []
    for s in cands:
        try:
            chg = float(ex.fetch_ticker(s).get("percentage") or 0.0)
        except: chg = 0.0
        scored.append((s, chg))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max(0,k)]

# ---------- state for TP/SL/Trail ----------
def seed_or_update(st: Dict[str,Dict[str,float]], sym: str, price: float, qty: float) -> None:
    ps = st.get(sym)
    if not ps:
        st[sym] = {"avg_price": price, "trail_max": price, "last_qty": qty}
        return
    prev_qty = float(ps.get("last_qty",0.0))
    prev_avg = float(ps.get("avg_price", price))
    if qty > prev_qty:
        add = qty - prev_qty
        ps["avg_price"] = (prev_avg*prev_qty + price*add)/max(qty,1e-12)
    if price > float(ps.get("trail_max", price)): ps["trail_max"] = price
    ps["last_qty"] = qty

def tp_hit(price: float, avg: float) -> bool:
    return avg>0 and (price-avg)/avg*100.0 >= TP_PCT

def sl_hit(price: float, avg: float) -> bool:
    return avg>0 and (price-avg)/avg*100.0 <= -SL_PCT

def trail_hit(price: float, trail_max: float, avg: float) -> bool:
    if trail_max<=0 or price<=0 or price<=avg: return False
    return (trail_max-price)/trail_max*100.0 >= TRAIL_PCT

# ---------- orders ----------
def market_buy(ex: ccxt.Exchange, sym: str, usd_amount: float, dry: bool, reject: Dict[str,float]) -> Optional[Dict[str,Any]]:
    if usd_amount < MIN_NOTIONAL_USD:
        log("WARN", f"Buy skipped < MIN_NOTIONAL_USD ${MIN_NOTIONAL_USD:.2f}")
        return None
    try:
        price = float(ex.fetch_ticker(sym)["last"])
        qty = usd_amount/max(price,1e-12)
        if dry:
            log("OK", f"DRY BUY {sym} ${usd_amount:.2f} (~{qty:.6f}) @ {price:.4f}")
            return {"id":"dry","side":"buy","symbol":sym,"amount":qty,"price":price}
        o = ex.create_order(sym,"market","buy",qty)
        log("OK", f"BUY {sym} ${usd_amount:.2f}")
        return o
    except ccxt.AuthenticationError as e:
        raise
    except Exception as e:
        msg = str(e)
        if "Invalid permissions" in msg or "trading restricted" in msg:
            # auto-blacklist this symbol for TTL
            until = time.time() + REJECT_TTL_DAYS*86400
            reject[sym] = until
            log("WARN", f"Auto-blacklist {sym} for {REJECT_TTL_DAYS}d (exchange says restricted)")
        else:
            log("ERR", f"Buy failed {sym}: {e}")
        return None

def market_sell(ex: ccxt.Exchange, sym: str, qty: float, dry: bool, reason: str="") -> Optional[Dict[str,Any]]:
    if qty <= 0: return None
    try:
        if dry:
            log("OK", f"DRY SELL {sym} qty {qty:.6f} {'— '+reason if reason else ''}")
            return {"id":"dry","side":"sell","symbol":sym,"amount":qty}
        o = ex.create_order(sym,"market","sell",qty)
        log("OK", f"SELL {sym} qty {qty:.6f} {'— '+reason if reason else ''}")
        return o
    except Exception as e:
        log("ERR", f"Sell failed {sym}: {e}")
        return None

# ---------- dust sweeper (min-lot aware) ----------
def apply_dust_sweeper(ex: ccxt.Exchange, positions: Dict[str,Dict[str,float]], dry: bool) -> float:
    freed = 0.0
    markets = ex.load_markets()
    for sym, pos in list(positions.items()):
        val = float(pos["value"])
        if val <= 0 or val > DUST_MIN_USD: continue
        qty = float(pos["amount"])
        # check Kraken min lot
        amt_min = 0.0
        try:
            limits = markets.get(sym,{}).get("limits",{}) or {}
            amt_min = float((limits.get("amount",{}) or {}).get("min") or 0.0)
        except: pass
        if amt_min and qty < amt_min:
            log("WARN", f"🧹 Dust unsweepable (below min lot): {sym} ${val:.2f}, qty {qty:.8f} < min {amt_min}")
            continue
        log("WARN", f"🧹 Sweeping dust: {sym} (${val:.2f})")
        market_sell(ex, sym, qty, dry, reason="dust")
        freed += val
        del positions[sym]
    return freed

# ---------- sell rules ----------
def apply_sell_rules(ex: ccxt.Exchange, positions: Dict[str,Dict[str,float]], st: Dict[str,Dict[str,float]], dry: bool) -> int:
    sells = 0
    for sym, pos in list(positions.items()):
        qty = float(pos["amount"])
        if qty <= 0: continue
        try: price = float(ex.fetch_ticker(sym)["last"])
        except: price = float(pos.get("price",0.0))
        ps = st.get(sym)
        if not ps:
            st[sym] = {"avg_price": price, "trail_max": price, "last_qty": qty}
            ps = st[sym]
        avg = float(ps.get("avg_price", price))
        tmax = float(ps.get("trail_max", price))
        if price > tmax:
            ps["trail_max"] = price; tmax = price

        if tp_hit(price, avg):
            log("OK", f"TAKE_PROFIT {sym} price {price:.6f} ≥ avg {avg:.6f} + {TP_PCT:.2f}%")
            market_sell(ex, sym, qty, dry, reason="TAKE_PROFIT")
            sells += 1; del positions[sym]; st.pop(sym, None); continue

        if sl_hit(price, avg):
            log("WARN", f"STOP_LOSS {sym} price {price:.6f} ≤ avg {avg:.6f} - {SL_PCT:.2f}%")
            market_sell(ex, sym, qty, dry, reason="STOP_LOSS")
            sells += 1; del positions[sym]; st.pop(sym, None); continue

        if trail_hit(price, tmax, avg):
            log("WARN", f"TRAILING_STOP {sym} drawdown from {tmax:.6f} ≥ {TRAIL_PCT:.2f}% (price {price:.6f})")
            market_sell(ex, sym, qty, dry, reason="TRAILING_STOP")
            sells += 1; del positions[sym]; st.pop(sym, None); continue

        # hold → keep state fresh
        seed_or_update(st, sym, price, qty)
    return sells

# ---------- rotation helpers ----------
def weakest_symbol(positions: Dict[str,Dict[str,float]], ex: ccxt.Exchange) -> Optional[Tuple[str,float]]:
    weakest, wchg = None, 1e9
    for sym in positions.keys():
        try: chg = float(ex.fetch_ticker(sym).get("percentage") or 0.0)
        except: chg = -999.0
        if chg < wchg: weakest, wchg = sym, chg
    if weakest is None: return None
    return weakest, wchg

# ---------- main ----------
def main() -> int:
    ensure_state()
    if SHOW_BANNER:
        log("SUM","===================================================")
        log("SUM", f"{now_iso()}  DRY_RUN={'ON' if DRY_RUN else 'OFF'}  RUN_SWITCH={'ON' if RUN_SWITCH else 'OFF'}")
        log("SUM", f"MaxPos {MAX_POSITIONS}, MaxBuys/run {MAX_BUYS_PER_RUN}, Dust <= ${DUST_MIN_USD:.2f}, MinNotional ${MIN_NOTIONAL_USD:.2f}")
        log("SUM","===================================================")

    if not RUN_SWITCH:
        log("WARN","RUN_SWITCH=off — skipping trading loop.")
        write_summary(["RUN_SWITCH=off — no trading performed."])
        return 0

    try:
        ex = make_exchange()
        _ = ex.fetch_ticker(f"BTC/{BASE_QUOTE}")
    except Exception as e:
        log("ERR", f"Exchange init/ticker failed: {e}")
        write_summary([f"Hard error: {e}"]);  return 1 if AUTO_PAUSE_ON_ERROR else 0

    try:
        # load state
        pos_state: Dict[str,Dict[str,float]] = load_json(POS_STATE, {})
        reject_map: Dict[str,float] = load_json(REJECT_LIST, {})  # sym -> unix expiry
        # purge expired rejects
        now = time.time()
        reject_map = {s:exp for s,exp in reject_map.items() if exp>now}

        cash, positions = fetch_cash_positions(ex)
        equity = cash + sum(p["value"] for p in positions.values())
        log("OK", f"Start — Cash ${cash:.2f}, Equity ${equity:.2f}, Positions {len(positions)}")

        # sells first
        sells1 = apply_sell_rules(ex, positions, pos_state, DRY_RUN)

        # dust sweep
        freed = apply_dust_sweeper(ex, positions, DRY_RUN)
        if freed>0:
            cash += freed
            log("OK", f"Dust sweep freed ~${freed:.2f}; Cash now ${cash:.2f}")

        # targets
        top = fetch_topk(ex, TOPK)
        top = [(s,c) for (s,c) in top if s not in reject_map or reject_map[s] < now]
        if UNIVERSE_WHITELIST:
            log("OK", "Whitelist active → " + ", ".join(UNIVERSE_WHITELIST))
        log("OK", "Top-K by 24h%: " + ", ".join([f"{s}({c:+.1f}%)" for s,c in top]))

        have = set(positions.keys())
        buys_done = 0; rotations = 0; sells_rot = 0

        # buy loop
        for s,_chg in top:
            if buys_done >= MAX_BUYS_PER_RUN: break
            if s in have: continue
            reserve = equity*(RESERVE_CASH_PCT/100.0)
            spendable = max(0.0, cash - reserve)

            if spendable < MIN_NOTIONAL_USD and ROTATE_WHEN_CASH_SHORT and len(positions)>0:
                w = weakest_symbol(positions, ex)
                if w:
                    wsym, wchg = w
                    qty = positions[wsym]["amount"]
                    log("WARN", f"ROTATE_WHEN_CASH_SHORT → selling weakest {wsym} ({wchg:+.1f}%) to fund {s}")
                    if market_sell(ex, wsym, qty, DRY_RUN, reason="cash_short"):
                        sells_rot += 1; rotations += 1
                        cash, positions = fetch_cash_positions(ex); have = set(positions.keys())
                        reserve = equity*(RESERVE_CASH_PCT/100.0)
                        spendable = max(0.0, cash - reserve)

            if spendable >= MIN_NOTIONAL_USD and buys_done < MAX_BUYS_PER_RUN and len(positions) < MAX_POSITIONS:
                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1,(MAX_POSITIONS-len(positions))))
                before_rejects = dict(reject_map)
                o = market_buy(ex, s, buy_amt, DRY_RUN, reject_map)
                if o:
                    buys_done += 1
                    cash, positions = fetch_cash_positions(ex); have = set(positions.keys())
                    try: price = float(ex.fetch_ticker(s)["last"])
                    except: price = 0.0
                    qty = positions.get(s,{}).get("amount",0.0)
                    seed_or_update(pos_state, s, price, float(qty))
                else:
                    # if restricted we already updated reject_map; persist shortly
                    if reject_map != before_rejects:
                        save_json(REJECT_LIST, reject_map)

        # if full → one swap
        if ROTATE_WHEN_FULL and len(positions)>=MAX_POSITIONS and buys_done<MAX_BUYS_PER_RUN:
            candidate = next((sym for sym,_ in top if sym not in have), None)
            if candidate and candidate not in reject_map:
                w = weakest_symbol(positions, ex)
                if w:
                    wsym, _wchg = w
                    if wsym != candidate:
                        qty = positions[wsym]["amount"]
                        log("WARN", f"ROTATE_WHEN_FULL → {wsym} -> {candidate}")
                        if market_sell(ex, wsym, qty, DRY_RUN, reason="rotate_full"):
                            sells_rot += 1; rotations += 1
                            cash, positions = fetch_cash_positions(ex); have=set(positions.keys())
                            reserve = equity*(RESERVE_CASH_PCT/100.0)
                            spendable = max(0.0, cash - reserve)
                            if spendable >= MIN_NOTIONAL_USD and buys_done<MAX_BUYS_PER_RUN:
                                buy_amt = max(MIN_NOTIONAL_USD, spendable / max(1,(MAX_POSITIONS-len(positions))))
                                before_rejects = dict(reject_map)
                                o = market_buy(ex, candidate, buy_amt, DRY_RUN, reject_map)
                                if o:
                                    buys_done += 1
                                    cash, positions = fetch_cash_positions(ex); have=set(positions.keys())
                                    try: price = float(ex.fetch_ticker(candidate)["last"])
                                    except: price = 0.0
                                    qty2 = positions.get(candidate,{}).get("amount",0.0)
                                    seed_or_update(pos_state, candidate, price, float(qty2))
                                else:
                                    if reject_map != before_rejects:
                                        save_json(REJECT_LIST, reject_map)

        # end state
        save_json(POS_STATE, pos_state)
        save_json(REJECT_LIST, reject_map)

        cash_end, positions_end = fetch_cash_positions(ex)
        equity_end = cash_end + sum(p["value"] for p in positions_end.values())
        log("KPI", f"End — Cash ${cash_end:.2f}, Equity ${equity_end:.2f}, Positions {len(positions_end)}")
        append_kpi({
            "ts": now_iso(),
            "dry_run": DRY_RUN,
            "positions": len(positions_end),
            "cash_usd": round(cash_end,2),
            "equity_usd": round(equity_end,2),
            "pnl_day_pct": 0.0,
            "buys": buys_done,
            "sells": int(sells1 + sells_rot),
            "rotations": rotations,
        })
        write_summary([
            f"Mode: {'DRY' if DRY_RUN else 'LIVE'}",
            f"Buys: {buys_done}, Sells: {int(sells1+sells_rot)}, Rotations: {rotations}",
            f"End Equity: ${equity_end:.2f} (Cash ${cash_end:.2f}, Positions {len(positions_end)})",
        ])
        return 0

    except ccxt.AuthenticationError as e:
        log("ERR", f"Authentication error: {e}")
        write_summary([f"Authentication error: {e}"])
        return 1 if AUTO_PAUSE_ON_ERROR else 0
    except Exception as e:
        log("ERR", f"Unhandled error: {e}")
        traceback.print_exc()
        write_summary([f"Unhandled error: {e}"])
        return 1 if AUTO_PAUSE_ON_ERROR else 0

if __name__ == "__main__":
    sys.exit(main())
