# main.py — Rotation-first, Losers-first exits, Why-Not logs
#
# Changes in this version
# - Rotation happens BEFORE exits (by default): dump weakest, buy strongest.
# - Exits are scoped by EXIT_SCOPE (default "weakest") so we don't take profit on winners
#   unless you ask for it.
# - Knobs: EXIT_SCOPE = weakest | bottom | all ; EXIT_BEFORE_ROTATE = true|false
#
# Core features kept
# - Auto-rank by 24h % move proxy
# - Rotate weakest→strongest with guards (min-cost, spread, cooldown, reserve)
# - Exits: TAKE_PROFIT / STOP_LOSS / TRAIL (with CI-friendly labels)
# - WHY-NOT logs for every rejection
# - Strict weakest option + partial-assist for cash
#
# ---------------- ENV ----------------
# Exchange & auth
#   EXCHANGE="kraken"
#   KRAKEN_API_KEY / KRAKEN_API_SECRET    # required for live
#   DRY_RUN="true"|"false" (default true)
#
# Universe & ranking
#   UNIVERSE="BTC,ETH,SOL,DOGE,ZEC,SUI,XLM,ADA,AVAX" (optional)
#   IGNORE_TICKERS="USDT,USDC,USD,EUR,GBP,XRP"
#   TOP_K="6"
#   EDGE_THRESHOLD="0.004"
#
# Trading guards
#   RESERVE_CASH_USD="25"
#   USD_PER_TRADE="10"
#   MIN_COST_PER_ORDER="5.0"
#   MAX_SPREAD_BPS="75"
#   COOLDOWN_MINUTES="10"
#
# Rotation behavior
#   ROTATE_WEAKEST_STRICT="false"
#   PARTIAL_ASSIST="true"
#   WHY_NOT_LOGS="true"
#
# Exit behavior
#   TAKE_PROFIT_PCT="0.05"      # TAKE_PROFIT
#   STOP_LOSS_PCT="0.03"        # STOP_LOSS
#   TRAILING_STOP_PCT="0.02"    # TRAIL
#   EXIT_SCOPE="weakest"        # weakest | bottom | all
#   EXIT_BEFORE_ROTATE="false"
#
from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

# ---------- utils ----------
def as_bool(v: Optional[str], default: bool=False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def as_float(v: Optional[str], default: float=0.0) -> float:
    try: return float(v) if v is not None else default
    except Exception: return default

def env_csv(name: str) -> List[str]:
    v = os.getenv(name) or ""
    return [s.strip().upper() for s in v.split(",") if s.strip()]

def now_utc() -> datetime: return datetime.now(timezone.utc)
def ts() -> str: return now_utc().strftime("%Y-%m-%d %H:%M:%S %Z")

# ---------- ENV ----------
EXCHANGE_NAME  = (os.getenv("EXCHANGE") or "kraken").lower()
DRY_RUN        = as_bool(os.getenv("DRY_RUN"), True)

UNIVERSE       = env_csv("UNIVERSE")
IGNORE_TICKERS = set(env_csv("IGNORE_TICKERS") or ["USDT","USDC","USD","EUR","GBP","XRP"])

TOP_K          = int(os.getenv("TOP_K") or "6")
EDGE_THRESHOLD = as_float(os.getenv("EDGE_THRESHOLD"), 0.004)

RESERVE_CASH_USD   = as_float(os.getenv("RESERVE_CASH_USD"), 25.0)
USD_PER_TRADE      = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MIN_COST_PER_ORDER = as_float(os.getenv("MIN_COST_PER_ORDER"), 5.0)
MAX_SPREAD_BPS     = as_float(os.getenv("MAX_SPREAD_BPS"), 75.0)
COOLDOWN_MINUTES   = int(os.getenv("COOLDOWN_MINUTES") or "10")

ROTATE_WEAKEST_STRICT = as_bool(os.getenv("ROTATE_WEAKEST_STRICT"), False)
PARTIAL_ASSIST        = as_bool(os.getenv("PARTIAL_ASSIST"), True)
WHY_NOT_LOGS          = as_bool(os.getenv("WHY_NOT_LOGS"), True)

TAKE_PROFIT_PCT    = as_float(os.getenv("TAKE_PROFIT_PCT"), 0.05)   # TAKE_PROFIT
STOP_LOSS_PCT      = as_float(os.getenv("STOP_LOSS_PCT"),   0.03)   # STOP_LOSS
TRAILING_STOP_PCT  = as_float(os.getenv("TRAILING_STOP_PCT"),0.02)  # TRAIL
EXIT_SCOPE         = (os.getenv("EXIT_SCOPE") or "weakest").strip().lower()  # weakest|bottom|all
EXIT_BEFORE_ROTATE = as_bool(os.getenv("EXIT_BEFORE_ROTATE"), False)

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
LAST_TRADES_FILE = STATE_DIR / "last_trades.json"
KPI_CSV          = STATE_DIR / "kpi_history.csv"

print(f"[BOOT] {ts()} EXCHANGE={EXCHANGE_NAME} DRY_RUN={DRY_RUN} TOP_K={TOP_K} EDGE={EDGE_THRESHOLD} "
      f"RESERVE=${RESERVE_CASH_USD} USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} "
      f"SPREAD_BPS<={MAX_SPREAD_BPS} COOLDOWN_MIN={COOLDOWN_MINUTES} STRICT={ROTATE_WEAKEST_STRICT} "
      f"PARTIAL_ASSIST={PARTIAL_ASSIST} EXIT_SCOPE={EXIT_SCOPE} EXIT_BEFORE_ROTATE={EXIT_BEFORE_ROTATE} "
      f"TP={TAKE_PROFIT_PCT:.3f} SL={STOP_LOSS_PCT:.3f} TRAIL={TRAILING_STOP_PCT:.3f}")

# ---------- ccxt ----------
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERROR] ccxt import failed: {e}")

def build_exchange():
    if EXCHANGE_NAME != "kraken":
        raise SystemExit("[ERROR] Only kraken is wired in this script right now.")
    return ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })

# ---------- helpers ----------
def load_last_trades() -> Dict[str, str]:
    if LAST_TRADES_FILE.exists():
        try: return json.loads(LAST_TRADES_FILE.read_text())
        except Exception: pass
    return {}

def save_last_trade(sym: str):
    d = load_last_trades(); d[sym.upper()] = now_utc().isoformat()
    LAST_TRADES_FILE.write_text(json.dumps(d, indent=2))

def cooldown_active(sym: str) -> bool:
    d = load_last_trades()
    if sym.upper() not in d: return False
    t = datetime.fromisoformat(d[sym.upper()])
    return (now_utc() - t) < timedelta(minutes=COOLDOWN_MINUTES)

def direct_market(ex, base: str, quote: str="USD") -> Optional[str]:
    s = f"{base}/{quote}"; return s if s in ex.markets else None

def spread_bps(ticker: dict) -> float:
    bid = ticker.get("bid") or 0.0; ask = ticker.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    mid = (bid+ask)/2.0; return (ask-bid)/mid*1e4

def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def record_kpi(summary: str):
    row = [now_utc().isoformat(), summary]
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["ts","summary"])
        w.writerow(row)

# ---------- exit checks (with labels for CI) ----------
def take_profit_check(sym: str, tkr: dict) -> bool:
    ch = tkr.get("percentage")
    try: pct = float(ch)/100.0 if ch is not None else 0.0
    except Exception: pct = 0.0
    if pct >= TAKE_PROFIT_PCT:
        print(f"[TAKE_PROFIT] {sym} 24h +{pct:.3%} >= {TAKE_PROFIT_PCT:.3%}")
        return True
    return False

def stop_loss_check(sym: str, tkr: dict) -> bool:
    ch = tkr.get("percentage")
    try: pct = float(ch)/100.0 if ch is not None else 0.0
    except Exception: pct = 0.0
    if pct <= -STOP_LOSS_PCT:
        print(f"[STOP_LOSS] {sym} 24h {pct:.3%} <= -{STOP_LOSS_PCT:.3%}")
        return True
    return False

def trailing_stop_check(sym: str, tkr: dict) -> bool:
    hi = tkr.get("high"); last = tkr.get("last") or tkr.get("close") or 0.0
    if not hi or not last or hi <= 0: return False
    dd = 1.0 - (last/hi)
    if dd >= TRAILING_STOP_PCT:
        print(f"[TRAIL] {sym} intraday drawdown {dd:.3%} >= {TRAILING_STOP_PCT:.3%}")
        return True
    return False

def exit_eligible(ex, sym: str, amt: float, tkr: dict, why: List[str]) -> bool:
    mkt = direct_market(ex, sym, "USD")
    if not mkt:            why.append("no USD market"); return False
    bps = spread_bps(tkr)
    if bps > MAX_SPREAD_BPS: why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    px = price_bid_or_last(tkr)
    if px <= 0:            why.append("no price"); return False
    m  = ex.market(mkt)
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if amt * px < floor:   why.append(f"notional ${amt*px:.2f} < min_cost ${floor:.2f}"); return False
    if cooldown_active(sym): why.append("cooldown active"); return False
    return True

def eligible_reason_buy(ex, base: str, usd_budget: float, tkr: dict, why: List[str]) -> bool:
    if base.upper() in IGNORE_TICKERS:
        why.append(f"{base}: in IGNORE_TICKERS"); return False
    sym = direct_market(ex, base, "USD")
    if not sym:            why.append(f"{base}: no {base}/USD market"); return False
    bps = spread_bps(tkr)
    if bps > MAX_SPREAD_BPS: why.append(f"{base}: spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    m  = ex.market(sym)
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if usd_budget < floor: why.append(f"{base}: budget ${usd_budget:.2f} < min_cost ${floor:.2f}"); return False
    return True

# ---------- main ----------
def main() -> int:
    print(f"[START] {ts()} — run")
    ex = build_exchange(); ex.load_markets()
    balances = ex.fetch_balance()
    total: Dict[str, float] = (balances.get("total") or {})
    usd = float(total.get("USD", 0.0))
    holdings = {a.upper(): amt for a, amt in total.items()
                if amt and a.upper() not in {"USD","USDT","USDC","EUR","GBP"}}
    print(f"[BAL] USD: ${usd:.2f}")
    print(f"[HOLD] {[(k, round(v,8)) for k,v in holdings.items() if v>0]}")

    # Prices & scoring
    tickers = ex.fetch_tickers()
    def score(sym: str) -> float:
        mkt = direct_market(ex, sym, "USD")
        if not mkt: return -1e9
        t = tickers.get(mkt) or {}
        ch = t.get("percentage")
        try: return float(ch)/100.0 if ch is not None else 0.0
        except Exception: return 0.0

    # Universe
    if UNIVERSE:
        universe = [s for s in UNIVERSE if s not in IGNORE_TICKERS]
    else:
        universe = []
        for mkt, m in ex.markets.items():
            if m.get("quote") == "USD":
                b = (m.get("base") or "").upper()
                if b and b not in IGNORE_TICKERS: universe.append(b)
        universe = sorted(set(universe))

    ranked = sorted(universe, key=lambda s: score(s), reverse=True)
    top = ranked[:max(TOP_K,1)]
    print(f"[RANK] Top {TOP_K}: {top}")

    holding_list = [h for h in holdings.keys() if h not in IGNORE_TICKERS]
    holding_sorted = sorted(holding_list, key=lambda s: score(s))  # weakest first
    weakest = holding_sorted[0] if holding_sorted else None
    best_buy = next((c for c in top if c not in IGNORE_TICKERS), None)
    edge = (score(best_buy) - score(weakest)) if (weakest and best_buy) else 0.0
    print(f"[EDGE] weakest={weakest} best_buy={best_buy} edge={edge:.4f} thr={EDGE_THRESHOLD:.4f}")

    did_anything = False

    # -------- Order of operations --------
    if EXIT_BEFORE_ROTATE:
        did_anything |= do_exits(ex, holdings, tickers, top, weakest)
        did_anything |= do_rotation(ex, usd, holdings, tickers, weakest, holding_sorted, best_buy, edge)
    else:
        did_anything |= do_rotation(ex, usd, holdings, tickers, weakest, holding_sorted, best_buy, edge)
        did_anything |= do_exits(ex, holdings, tickers, top, weakest)

    summary = f"did_anything={did_anything} usd=${usd:.2f} weakest={weakest} best={best_buy} edge={edge:.4f}"
    print(f"[SUMMARY] {summary}"); record_kpi(summary)
    print(f"[END] {ts()}"); return 0

# ---------- rotation then exits ----------
def do_rotation(ex, usd, holdings, tickers, weakest, holding_sorted, best_buy, edge) -> bool:
    if not (weakest and best_buy and edge >= EDGE_THRESHOLD):
        print("[ROTATE] No valid pair or edge below threshold; skip rotation.")
        return False

    # SELL check for weakest
    sell_sym = direct_market(ex, weakest, "USD")
    wtkr = tickers.get(sell_sym) if sell_sym else None
    if wtkr is None and sell_sym: wtkr = ex.fetch_ticker(sell_sym)
    w_amt = holdings.get(weakest, 0.0)
    why_sell: List[str] = []
    ok_sell = exit_eligible(ex, weakest, w_amt, wtkr or {}, why_sell)

    if not ok_sell and ROTATE_WEAKEST_STRICT:
        if WHY_NOT_LOGS:
            for r in why_sell: print(f"[WHY-NOT][SELL {weakest}] {r}")
        print("[ROTATE] STRICT on; weakest blocked → skip rotation.")
        return False

    # BUY check
    usd_to_use = max(USD_PER_TRADE, MIN_COST_PER_ORDER)
    btkr = tickers.get(f"{best_buy}/USD") or {}
    why_buy: List[str] = []
    ok_buy = eligible_reason_buy(ex, best_buy, usd_to_use, btkr, why_buy)
    if not ok_buy and WHY_NOT_LOGS:
        for r in why_buy: print(f"[WHY-NOT][BUY {best_buy}] {r}")

    # Partial assist if needed (weakest blocked but STRICT off)
    if not ok_sell and not ROTATE_WEAKEST_STRICT and PARTIAL_ASSIST:
        for cand in holding_sorted[1:]:
            sym = direct_market(ex, cand, "USD")
            tkr = tickers.get(sym) if sym else None
            if tkr is None and sym: tkr = ex.fetch_ticker(sym)
            amt = holdings.get(cand, 0.0)
            reasons: List[str] = []
            if exit_eligible(ex, cand, amt, tkr or {}, reasons):
                px = price_bid_or_last(tkr or {})
                if px > 0:
                    sell_amt = max((MIN_COST_PER_ORDER + 0.50)/px, 0.0)
                    print(f"[ASSIST] SELL slice {cand} to free ~${MIN_COST_PER_ORDER:.2f}")
                    if DRY_RUN: print(f"[DRY-RUN] create_order SELL {cand}/USD amount={sell_amt}")
                    else:
                        try: ex.create_order(symbol=f"{cand}/USD", type="market", side="sell", amount=sell_amt); save_last_trade(cand)
                        except Exception as e: print(f"[ERROR] assist sell failed: {e}")
                break
            elif WHY_NOT_LOGS:
                for r in reasons: print(f"[WHY-NOT][ASSIST SELL {cand}] {r}")

    # Perform rotation
    did = False
    if ok_sell:
        print(f"[ROTATE] SELL weakest {weakest} amount={w_amt}")
        if DRY_RUN: print(f"[DRY-RUN] create_order SELL {weakest}/USD amount={w_amt}")
        else:
            try: ex.create_order(symbol=f"{weakest}/USD", type="market", side="sell", amount=w_amt); save_last_trade(weakest); did = True
            except Exception as e: print(f"[ERROR] sell failed: {e}")

    if ok_buy:
        print(f"[BUY ] BUY {best_buy} ~${usd_to_use:.2f}")
        if DRY_RUN: print(f"[DRY-RUN] create_order BUY {best_buy}/USD cost~${usd_to_use:.2f}")
        else:
            try:
                t = tickers.get(f"{best_buy}/USD") or ex.fetch_ticker(f"{best_buy}/USD")
                px = price_bid_or_last(t); amt = (usd_to_use/px) if px>0 else 0.0
                ex.create_order(symbol=f"{best_buy}/USD", type="market", side="buy", amount=amt); save_last_trade(best_buy); did = True
            except Exception as e: print(f"[ERROR] buy failed: {e}")
    return did

def do_exits(ex, holdings, tickers, top, weakest) -> bool:
    # Determine which symbols are in scope for exits
    symbols = []
    if EXIT_SCOPE == "weakest" and weakest:
        symbols = [weakest]
    elif EXIT_SCOPE == "bottom":
        # any holding that is NOT in today's Top-K
        topset = set(top)
        symbols = [s for s in holdings.keys() if s not in topset]
    else:  # "all"
        symbols = list(holdings.keys())

    did = False
    for sym in sorted(symbols):
        if sym in IGNORE_TICKERS: continue
        mkt = direct_market(ex, sym, "USD")
        if not mkt: continue
        tkr = tickers.get(mkt) or ex.fetch_ticker(mkt)
        amt = holdings.get(sym, 0.0)
        do_tp = take_profit_check(sym, tkr)
        do_sl = stop_loss_check(sym, tkr)
        do_tr = trailing_stop_check(sym, tkr)
        if not (do_tp or do_sl or do_tr): 
            continue
        why: List[str] = []
        if not exit_eligible(ex, sym, amt, tkr, why):
            if WHY_NOT_LOGS: print(f"[WHY-NOT][EXIT {sym}] " + "; ".join(why))
            continue
        label = "TAKE_PROFIT" if do_tp else ("STOP_LOSS" if do_sl else "TRAIL")
        print(f"[EXIT] {label} → SELL {sym} amount={amt}")
        if DRY_RUN:
            print(f"[DRY-RUN] create_order SELL {sym}/USD amount={amt}")
        else:
            try: ex.create_order(symbol=f"{sym}/USD", type="market", side="sell", amount=amt); save_last_trade(sym); did = True
            except Exception as e: print(f"[ERROR] exit sell failed for {sym}: {e}")
    return did

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
