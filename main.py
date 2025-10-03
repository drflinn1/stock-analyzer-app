# main.py — Atomic rotation (sell only if buy is eligible), clean universe,
# weakest-ELIGIBLE first, exits after rotation, WHY-NOT logs.
#
# New knobs
#   ATOMIC_ROTATION="true"      # require both sell+buy to be eligible (default true)
#   MIN_QUOTE_VOL_USD="50000"   # filter universe by 24h quote volume
#
# Other knobs retained (see prior versions): DRY_RUN, TOP_K, EDGE_THRESHOLD,
# RESERVE_CASH_USD, USD_PER_TRADE, MIN_COST_PER_ORDER, MAX_SPREAD_BPS,
# COOLDOWN_MINUTES, ROTATE_WEAKEST_STRICT, PARTIAL_ASSIST, WHY_NOT_LOGS,
# TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_STOP_PCT, EXIT_SCOPE, EXIT_BEFORE_ROTATE,
# SELL_EPS, BUY_EPS, UNIVERSE, IGNORE_TICKERS.

from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

def as_bool(v: Optional[str], default=False): 
    return str(v).strip().lower() in {"1","true","yes","y","on"} if v is not None else default
def as_float(v: Optional[str], default=0.0):
    try: return float(v) if v is not None else default
    except Exception: return default
def env_csv(name: str) -> List[str]:
    v = os.getenv(name) or ""; return [s.strip().upper() for s in v.split(",") if s.strip()]
def now_utc(): return datetime.now(timezone.utc)
def ts(): return now_utc().strftime("%Y-%m-%d %H:%M:%S %Z")

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
EXIT_SCOPE         = (os.getenv("EXIT_SCOPE") or "weakest").strip().lower()
EXIT_BEFORE_ROTATE = as_bool(os.getenv("EXIT_BEFORE_ROTATE"), False)

# NEW
ATOMIC_ROTATION     = as_bool(os.getenv("ATOMIC_ROTATION"), True)
MIN_QUOTE_VOL_USD   = as_float(os.getenv("MIN_QUOTE_VOL_USD"), 50000.0)

SELL_EPS = as_float(os.getenv("SELL_EPS"), 0.995)
BUY_EPS  = as_float(os.getenv("BUY_EPS"),  0.995)

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
LAST_TRADES_FILE = STATE_DIR / "last_trades.json"
KPI_CSV          = STATE_DIR / "kpi_history.csv"

print(f"[BOOT] {ts()} EXCHANGE={EXCHANGE_NAME} DRY_RUN={DRY_RUN} TOP_K={TOP_K} EDGE={EDGE_THRESHOLD} "
      f"RESERVE=${RESERVE_CASH_USD} USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} "
      f"SPREAD_BPS<={MAX_SPREAD_BPS} COOLDOWN_MIN={COOLDOWN_MINUTES} STRICT={ROTATE_WEAKEST_STRICT} "
      f"PARTIAL_ASSIST={PARTIAL_ASSIST} EXIT_SCOPE={EXIT_SCOPE} EXIT_BEFORE_ROTATE={EXIT_BEFORE_ROTATE} "
      f"ATOMIC_ROTATION={ATOMIC_ROTATION} MIN_QUOTE_VOL_USD={MIN_QUOTE_VOL_USD} "
      f"TP={TAKE_PROFIT_PCT:.3f} SL={STOP_LOSS_PCT:.3f} TRAIL={TRAILING_STOP_PCT:.3f} "
      f"SELL_EPS={SELL_EPS} BUY_EPS={BUY_EPS}")

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERROR] ccxt import failed: {e}")

def build_exchange():
    if EXCHANGE_NAME != "kraken":
        raise SystemExit("[ERROR] Only kraken is wired.")
    return ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })

def load_last_trades():
    if LAST_TRADES_FILE.exists():
        try: return json.loads(LAST_TRADES_FILE.read_text())
        except Exception: pass
    return {}
def save_last_trade(sym: str):
    d = load_last_trades(); d[sym.upper()] = now_utc().isoformat()
    LAST_TRADES_FILE.write_text(json.dumps(d, indent=2))
def cooldown_active(sym: str) -> bool:
    d = load_last_trades(); s = sym.upper()
    if s not in d: return False
    return (now_utc() - datetime.fromisoformat(d[s])) < timedelta(minutes=COOLDOWN_MINUTES)

def direct_market(ex, base: str, quote="USD"):
    s = f"{base}/{quote}"; return s if s in ex.markets else None

def spread_bps(t: dict) -> float:
    bid = t.get("bid") or 0.0; ask = t.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    return (ask - bid) / ((bid + ask)/2.0) * 1e4
def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def amount_precision(ex, symbol: str) -> int:
    m = ex.market(symbol)
    p = (m.get("precision") or {}).get("amount", None)
    return p if isinstance(p, int) and p >= 0 else 8
def clamp(amount: float, decimals: int) -> float:
    q = 10 ** decimals; return math.floor(amount * q) / q

def record_kpi(summary: str):
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["ts","summary"])
        w.writerow([now_utc().isoformat(), summary])

# ---- eligibility helpers
def exit_eligible(ex, sym: str, amt: float, t: dict, why: List[str]) -> bool:
    mkt = direct_market(ex, sym, "USD")
    if not mkt: why.append("no USD market"); return False
    if cooldown_active(sym): why.append("cooldown active"); return False
    bps = spread_bps(t)
    if bps > MAX_SPREAD_BPS: why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    px = price_bid_or_last(t)
    if px <= 0: why.append("no price"); return False
    m = ex.market(mkt)
    min_cost = float((m.get("limits") or {}).get("cost", {}).get("min") or 0)
    floor = max(MIN_COST_PER_ORDER, min_cost)
    if amt * px < floor: why.append(f"notional ${amt*px:.2f} < min_cost ${floor:.2f}"); return False
    return True

def buy_eligible(ex, base: str, budget_usd: float, t: dict, why: List[str]) -> bool:
    if base.upper() in IGNORE_TICKERS: why.append("ignored"); return False
    mkt = direct_market(ex, base, "USD")
    if not mkt: why.append("no USD market"); return False
    bps = spread_bps(t)
    if bps > MAX_SPREAD_BPS: why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    m = ex.market(mkt)
    min_cost = float((m.get("limits") or {}).get("cost", {}).get("min") or 0)
    floor = max(MIN_COST_PER_ORDER, min_cost)
    if budget_usd < floor: why.append(f"budget ${budget_usd:.2f} < min_cost ${floor:.2f}"); return False
    return True

# ---- scoring
def score_from_ticker(t: dict) -> float:
    ch = t.get("percentage")
    try: return float(ch)/100.0 if ch is not None else 0.0
    except Exception: return 0.0

# ---- exit checks (labels for CI)
def take_profit_check(sym: str, t: dict) -> bool:
    s = score_from_ticker(t); 
    if s >= TAKE_PROFIT_PCT:
        print(f"[TAKE_PROFIT] {sym} 24h +{s:.3%} >= {TAKE_PROFIT_PCT:.3%}"); 
        return True
    return False
def stop_loss_check(sym: str, t: dict) -> bool:
    s = score_from_ticker(t);
    if s <= -STOP_LOSS_PCT:
        print(f"[STOP_LOSS] {sym} 24h {s:.3%} <= -{STOP_LOSS_PCT:.3%}");
        return True
    return False
def trailing_stop_check(sym: str, t: dict) -> bool:
    hi = t.get("high"); last = t.get("last") or t.get("close") or 0.0
    if not hi or not last or hi <= 0: return False
    dd = 1.0 - (last/hi)
    if dd >= TRAILING_STOP_PCT:
        print(f"[TRAIL] {sym} intraday drawdown {dd:.3%} >= {TRAILING_STOP_PCT:.3%}")
        return True
    return False

# ---- main
def main() -> int:
    print(f"[START] {ts()} — run")
    ex = build_exchange(); ex.load_markets()
    balances = ex.fetch_balance()
    total: Dict[str, float] = (balances.get("total") or {})
    free:  Dict[str, float] = (balances.get("free")  or {})
    usd = float(total.get("USD", 0.0))
    holdings = {a.upper(): amt for a, amt in total.items()
                if amt and a.upper() not in {"USD","USDT","USDC","EUR","GBP"}}
    print(f"[BAL] USD: ${usd:.2f}")
    print(f"[HOLD] {[(k, round(v,8)) for k,v in holdings.items() if v>0]}")

    tickers = ex.fetch_tickers()

    # ---- universe: spot, active, USD, volume+spread filters
    def market_ok(base: str) -> bool:
        if base in IGNORE_TICKERS: return False
        sym = direct_market(ex, base, "USD")
        if not sym: return False
        m = ex.market(sym)
        if not (m.get("active", True) and (m.get("spot", True) or m.get("type") == "spot")): 
            return False
        t = tickers.get(sym) or {}
        vol_q = t.get("quoteVolume")
        if vol_q is None:
            # Fallback: baseVolume*last
            bv = t.get("baseVolume"); px = price_bid_or_last(t)
            vol_q = (bv or 0) * (px or 0)
        if (vol_q or 0) < MIN_QUOTE_VOL_USD: 
            return False
        if spread_bps(t) > MAX_SPREAD_BPS:
            return False
        return True

    if UNIVERSE:
        universe = [s for s in UNIVERSE if market_ok(s)]
    else:
        universe = sorted({ (m.get("base") or "").upper()
                            for _, m in ex.markets.items()
                            if m.get("quote") == "USD" and (m.get("spot", True) or m.get("type")=="spot") })
        universe = [s for s in universe if market_ok(s)]

    # score map
    scores: Dict[str, float] = {}
    for b in universe:
        t = tickers.get(f"{b}/USD") or {}
        scores[b] = score_from_ticker(t)

    ranked = sorted(universe, key=lambda s: scores.get(s, -1e9), reverse=True)
    top = ranked[:max(TOP_K,1)]
    print(f"[RANK] Top {TOP_K}: {top}")

    # ---- weakest = first *eligible* to sell among holdings (worst->best)
    holding_list = [h for h in holdings.keys() if h not in IGNORE_TICKERS]
    holding_sorted = sorted(holding_list, key=lambda s: scores.get(s, -1e9))  # weakest first

    weakest = None; w_amt=0.0; wtkr=None
    for cand in holding_sorted:
        sym = direct_market(ex, cand, "USD")
        if not sym: 
            if WHY_NOT_LOGS: print(f"[WHY-NOT][WEAKEST {cand}] no USD market")
            continue
        wtkr = tickers.get(sym) or ex.fetch_ticker(sym)
        total_amt = float(holdings.get(cand, 0.0))
        free_amt  = free.get(cand, None)
        amt = (float(free_amt) if free_amt is not None else total_amt) * SELL_EPS
        why=[]; 
        if exit_eligible(ex, cand, amt, wtkr, why):
            weakest, w_amt = cand, amt
            break
        elif WHY_NOT_LOGS:
            for r in why: print(f"[WHY-NOT][WEAKEST {cand}] {r}")
    if not weakest:
        print("[ROTATE] No sell-eligible holding found. Skipping rotation.")
        return finish(usd, None, None, 0.0)

    # ---- choose first BUY candidate in Top-K that is eligible
    usd_to_use = max(USD_PER_TRADE, MIN_COST_PER_ORDER) * BUY_EPS
    buy = None; btkr=None
    for c in top:
        t = tickers.get(f"{c}/USD") or ex.fetch_ticker(f"{c}/USD")
        why=[]
        if buy_eligible(ex, c, usd_to_use, t, why):
            buy, btkr = c, t
            break
        elif WHY_NOT_LOGS:
            for r in why: print(f"[WHY-NOT][BUY {c}] {r}")
    if not buy:
        print("[ROTATE] No eligible buy in Top-K → not selling weakest (atomic rotation).")
        return finish(usd, weakest, None, 0.0)

    edge = (scores.get(buy, 0.0) - scores.get(weakest, 0.0))
    print(f"[EDGE] weakest={weakest} best_buy={buy} edge={edge:.4f} thr={EDGE_THRESHOLD:.4f}")

    did_anything = False

    # ---- rotation only if edge ok
    if edge >= EDGE_THRESHOLD:
        # SELL
        dec = amount_precision(ex, f"{weakest}/USD")
        w_amt = clamp(max(w_amt, 0.0), dec)
        print(f"[ROTATE] SELL weakest {weakest} amount={w_amt}")
        if DRY_RUN:
            print(f"[DRY-RUN] create_order SELL {weakest}/USD amount={w_amt}")
        else:
            try:
                ex.create_order(symbol=f"{weakest}/USD", type="market", side="sell", amount=w_amt)
                save_last_trade(weakest)
            except Exception as e:
                print(f"[ERROR] sell failed ({weakest}): {e}")
                # one smaller retry
                w_amt2 = clamp(w_amt*0.97, dec)
                if w_amt2>0 and w_amt2!=w_amt:
                    print(f"[RETRY] SELL {weakest} amount={w_amt2}")
                    try:
                        ex.create_order(symbol=f"{weakest}/USD", type="market", side="sell", amount=w_amt2)
                        save_last_trade(weakest)
                    except Exception as e2:
                        print(f"[ERROR] retry sell failed: {e2}")
                else:
                    print("[ROTATE] abort buy due to failed sell.")
                    return finish(usd, weakest, buy, edge)
        # BUY
        print(f"[BUY ] BUY {buy} budget~${usd_to_use:.2f}")
        if DRY_RUN:
            print(f"[DRY-RUN] create_order BUY {buy}/USD cost~${usd_to_use:.2f}")
            did_anything = True
        else:
            try:
                px = price_bid_or_last(btkr); decb = amount_precision(ex, f"{buy}/USD")
                amt_buy = clamp((usd_to_use/px) if px>0 else 0.0, decb)
                ex.create_order(symbol=f"{buy}/USD", type="market", side="buy", amount=amt_buy)
                save_last_trade(buy); did_anything = True
            except Exception as e:
                print(f"[ERROR] buy failed: {e}")
    else:
        print("[ROTATE] Edge below threshold; skip rotation.")

    # ---- exits AFTER rotation, scoped
    do_exits(ex, holdings, free, tickers, top, weakest)

    return finish(usd, weakest, buy, edge, did_anything)

def do_exits(ex, holdings, free, tickers, top, weakest):
    if EXIT_SCOPE == "weakest" and weakest:
        symbols = [weakest]
    elif EXIT_SCOPE == "bottom":
        symbols = [s for s in holdings.keys() if s not in set(top)]
    else:
        symbols = list(holdings.keys())

    for sym in sorted(symbols):
        if sym in IGNORE_TICKERS: continue
        mkt = direct_market(ex, sym, "USD")
        if not mkt: continue
        t = tickers.get(mkt) or ex.fetch_ticker(mkt)
        total_amt = float(holdings.get(sym, 0.0))
        free_amt  = free.get(sym, None)
        amt = (float(free_amt) if free_amt is not None else total_amt) * SELL_EPS
        tp = take_profit_check(sym, t); sl = stop_loss_check(sym, t); tr = trailing_stop_check(sym, t)
        if not (tp or sl or tr): continue
        why=[]
        if not exit_eligible(ex, sym, amt, t, why):
            if WHY_NOT_LOGS: print(f"[WHY-NOT][EXIT {sym}] " + "; ".join(why))
            continue
        dec = amount_precision(ex, f"{sym}/USD"); amt = clamp(amt, dec)
        label = "TAKE_PROFIT" if tp else ("STOP_LOSS" if sl else "TRAIL")
        print(f"[EXIT] {label} → SELL {sym} amount={amt}")
        if DRY_RUN:
            print(f"[DRY-RUN] create_order SELL {sym}/USD amount={amt}")
        else:
            try:
                ex.create_order(symbol=f"{sym}/USD", type="market", side="sell", amount=amt)
                save_last_trade(sym)
            except Exception as e:
                print(f"[ERROR] exit sell failed for {sym}: {e}")

def finish(usd, weakest, best, edge, did=False):
    summary = f"did_anything={did} usd=${usd:.2f} weakest={weakest} best={best} edge={edge:.4f}"
    print(f"[SUMMARY] {summary}")
    record_kpi(summary)
    print(f"[END] {ts()}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
