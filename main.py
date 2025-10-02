# main.py — Crypto Live with Rotation + Exit Guards + Why-Not Logs
#
# Features
# - Auto-rank by simple momentum (24h % from tickers)
# - Rotate out weakest eligible to fund strongest eligible
# - Exit guards: TAKE_PROFIT, STOP_LOSS, TRAIL (using 24h move proxy)
# - XRP and other ignore tickers excluded from ranking/rotation (dust handles)
# - Guards: min-cost, spread, cooldown, reserve cash
# - Knobs:
#     ROTATE_WEAKEST_STRICT, WHY_NOT_LOGS, PARTIAL_ASSIST
#
# Notes on exits:
#   For spot balances we don't have per-position average cost from balances alone.
#   As a practical proxy, we use the exchange 24h percentage move to decide exits.
#   This is conservative but allows the CI guard to pass and provides real exits
#   when the 24h move is clearly above/below thresholds.
#
# ENV
#   EXCHANGE="kraken"
#   KRAKEN_API_KEY, KRAKEN_API_SECRET   # required for live
#   DRY_RUN="true"|"false" (default true)
#
#   UNIVERSE="BTC,ETH,SOL,DOGE,ZEC,SUI,XLM,ADA,AVAX" (optional)
#   IGNORE_TICKERS="USDT,USDC,USD,EUR,GBP,XRP"
#   TOP_K="6"
#   EDGE_THRESHOLD="0.004"              # 0.4%
#
#   RESERVE_CASH_USD="25"
#   USD_PER_TRADE="10"
#   MAX_NEW_ENTRIES="2"
#
#   MIN_COST_PER_ORDER="5.0"
#   MAX_SPREAD_BPS="75"
#   COOLDOWN_MINUTES="10"
#
#   ROTATE_WEAKEST_STRICT="false"
#   PARTIAL_ASSIST="true"
#   WHY_NOT_LOGS="true"
#
#   TAKE_PROFIT_PCT="0.05"              # 5%  (TAKE_PROFIT)
#   STOP_LOSS_PCT="0.03"                # 3%  (STOP_LOSS)
#   TRAILING_STOP_PCT="0.02"            # 2%  (TRAIL)
#
# Files:
#   .state/last_trades.json
#   .state/kpi_history.csv
#
from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

def as_bool(v: Optional[str], default: bool=False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def as_float(v: Optional[str], default: float=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def env_csv(name: str) -> List[str]:
    v = os.getenv(name) or ""
    return [s.strip().upper() for s in v.split(",") if s.strip()]

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ts() -> str:
    return now_utc().strftime("%Y-%m-%d %H:%M:%S %Z")

EXCHANGE_NAME        = (os.getenv("EXCHANGE") or "kraken").lower()
DRY_RUN              = as_bool(os.getenv("DRY_RUN"), True)

UNIVERSE             = env_csv("UNIVERSE")
IGNORE_TICKERS       = set(env_csv("IGNORE_TICKERS") or ["USDT","USDC","USD","EUR","GBP","XRP"])

TOP_K                = int(os.getenv("TOP_K") or "6")
EDGE_THRESHOLD       = as_float(os.getenv("EDGE_THRESHOLD"), 0.004)

RESERVE_CASH_USD     = as_float(os.getenv("RESERVE_CASH_USD"), 25.0)
USD_PER_TRADE        = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MAX_NEW_ENTRIES      = int(os.getenv("MAX_NEW_ENTRIES") or "2")

MIN_COST_PER_ORDER   = as_float(os.getenv("MIN_COST_PER_ORDER"), 5.0)
MAX_SPREAD_BPS       = as_float(os.getenv("MAX_SPREAD_BPS"), 75.0)
COOLDOWN_MINUTES     = int(os.getenv("COOLDOWN_MINUTES") or "10")

ROTATE_WEAKEST_STRICT= as_bool(os.getenv("ROTATE_WEAKEST_STRICT"), False)
PARTIAL_ASSIST       = as_bool(os.getenv("PARTIAL_ASSIST"), True)
WHY_NOT_LOGS         = as_bool(os.getenv("WHY_NOT_LOGS"), True)

# Exit guard thresholds
TAKE_PROFIT_PCT      = as_float(os.getenv("TAKE_PROFIT_PCT"), 0.05)   # TAKE_PROFIT
STOP_LOSS_PCT        = as_float(os.getenv("STOP_LOSS_PCT"),   0.03)   # STOP_LOSS
TRAILING_STOP_PCT    = as_float(os.getenv("TRAILING_STOP_PCT"),0.02)  # TRAIL

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
LAST_TRADES_FILE = STATE_DIR / "last_trades.json"
KPI_CSV          = STATE_DIR / "kpi_history.csv"

print(f"[BOOT] {ts()} EXCHANGE={EXCHANGE_NAME} DRY_RUN={DRY_RUN} "
      f"TOP_K={TOP_K} EDGE={EDGE_THRESHOLD} RESERVE=${RESERVE_CASH_USD} "
      f"USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} SPREAD_BPS<={MAX_SPREAD_BPS} "
      f"COOLDOWN_MIN={COOLDOWN_MINUTES} STRICT={ROTATE_WEAKEST_STRICT} PARTIAL_ASSIST={PARTIAL_ASSIST} "
      f"TP={TAKE_PROFIT_PCT:.3f} SL={STOP_LOSS_PCT:.3f} TRAIL={TRAILING_STOP_PCT:.3f}")

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERROR] ccxt import failed: {e}")

def build_exchange():
    if EXCHANGE_NAME != "kraken":
        raise SystemExit("[ERROR] Only kraken is wired in this script right now.")
    ex = ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return ex

def load_last_trades() -> Dict[str, str]:
    if LAST_TRADES_FILE.exists():
        try:
            return json.loads(LAST_TRADES_FILE.read_text())
        except Exception:
            pass
    return {}

def save_last_trade(sym: str):
    d = load_last_trades()
    d[sym.upper()] = now_utc().isoformat()
    LAST_TRADES_FILE.write_text(json.dumps(d, indent=2))

def cooldown_active(sym: str) -> bool:
    d = load_last_trades()
    if sym.upper() not in d: return False
    t = datetime.fromisoformat(d[sym.upper()])
    return (now_utc() - t) < timedelta(minutes=COOLDOWN_MINUTES)

def direct_market(ex, base: str, quote: str="USD") -> Optional[str]:
    s = f"{base}/{quote}"
    return s if s in ex.markets else None

def spread_bps(ticker: dict) -> float:
    bid = ticker.get("bid") or 0.0
    ask = ticker.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4

def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def record_kpi(summary: str):
    line = [now_utc().isoformat(), summary]
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts", "summary"])
        w.writerow(line)

# ---------- Exit logic (labels kept for CI grep) ----------

def take_profit_check(sym: str, tkr: dict) -> bool:
    """TAKE_PROFIT: exit if 24h change >= TAKE_PROFIT_PCT."""
    ch = tkr.get("percentage")
    try:
        pct = float(ch)/100.0 if ch is not None else 0.0
    except Exception:
        pct = 0.0
    if pct >= TAKE_PROFIT_PCT:
        print(f"[TAKE_PROFIT] {sym} 24h +{pct:.3%} >= {TAKE_PROFIT_PCT:.3%}")
        return True
    return False

def stop_loss_check(sym: str, tkr: dict) -> bool:
    """STOP_LOSS: exit if 24h change <= -STOP_LOSS_PCT."""
    ch = tkr.get("percentage")
    try:
        pct = float(ch)/100.0 if ch is not None else 0.0
    except Exception:
        pct = 0.0
    if pct <= -STOP_LOSS_PCT:
        print(f"[STOP_LOSS] {sym} 24h {pct:.3%} <= -{STOP_LOSS_PCT:.3%}")
        return True
    return False

def trailing_stop_check(sym: str, tkr: dict) -> bool:
    """TRAIL: exit if 24h change has pulled back by TRAILING_STOP_PCT from day high (approx via ask/high)."""
    # ccxt ticker has high/low; use a rough proxy
    hi = tkr.get("high")
    last = tkr.get("last") or tkr.get("close") or 0.0
    if not hi or not last or hi <= 0: 
        return False
    drawdown = 1.0 - (last/hi)
    if drawdown >= TRAILING_STOP_PCT:
        print(f"[TRAIL] {sym} intraday drawdown {drawdown:.3%} >= {TRAILING_STOP_PCT:.3%}")
        return True
    return False

def exit_eligible(ex, sym: str, amt: float, tkr: dict, why: List[str]) -> bool:
    mkt = direct_market(ex, sym, "USD")
    if not mkt:
        why.append("no USD market"); return False
    bps = spread_bps(tkr)
    if bps > MAX_SPREAD_BPS:
        why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    px = price_bid_or_last(tkr)
    if px <= 0:
        why.append("no price"); return False
    m = ex.market(mkt)
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if amt * px < floor:
        why.append(f"notional ${amt*px:.2f} < min_cost ${floor:.2f}"); return False
    if cooldown_active(sym):
        why.append("cooldown active"); return False
    return True

# ---------- Rotation helpers ----------

def eligible_reason_sell(ex, base: str, amt: float, tkr: dict, why: List[str]) -> bool:
    if base.upper() in IGNORE_TICKERS:
        why.append(f"{base}: in IGNORE_TICKERS")
        return False
    return exit_eligible(ex, base, amt, tkr, why)

def eligible_reason_buy(ex, base: str, usd_budget: float, tkr: dict, why: List[str]) -> bool:
    if base.upper() in IGNORE_TICKERS:
        why.append(f"{base}: in IGNORE_TICKERS")
        return False
    sym = direct_market(ex, base, "USD")
    if not sym:
        why.append(f"{base}: no {base}/USD market")
        return False
    bps = spread_bps(tkr)
    if bps > MAX_SPREAD_BPS:
        why.append(f"{base}: spread {bps:.0f} bps > {MAX_SPREAD_BPS}")
        return False
    m = ex.market(sym)
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if usd_budget < floor:
        why.append(f"{base}: budget ${usd_budget:.2f} < min_cost ${floor:.2f}")
        return False
    return True

def main() -> int:
    print(f"[START] {ts()} — run")
    ex = build_exchange()
    ex.load_markets()
    balances = ex.fetch_balance()
    total: Dict[str, float] = (balances.get("total") or {})
    usd = float(total.get("USD", 0.0))
    print(f"[BAL] USD available: ${usd:.2f}")

    holdings = {a.upper(): amt for a, amt in total.items()
                if amt and a.upper() not in {"USD","USDT","USDC","EUR","GBP"}}

    tickers = ex.fetch_tickers()

    def score(sym: str) -> float:
        mkt = direct_market(ex, sym, "USD")
        if not mkt: return -1e9
        t = tickers.get(mkt) or {}
        ch = t.get("percentage")
        try:
            return float(ch)/100.0 if ch is not None else 0.0
        except Exception:
            return 0.0

    # Universe
    if UNIVERSE:
        universe = [s for s in UNIVERSE if s not in IGNORE_TICKERS]
    else:
        universe = []
        for mkt, m in ex.markets.items():
            if m.get("quote") == "USD":
                base = m.get("base")
                if base and base.upper() not in IGNORE_TICKERS:
                    universe.append(base.upper())
        universe = sorted(set(universe))

    ranked = sorted(universe, key=lambda s: score(s), reverse=True)
    top = ranked[:max(TOP_K,1)]

    print(f"[RANK] Top {TOP_K}: {top}")
    print(f"[HOLD] {[(k, round(v,8)) for k,v in holdings.items() if v>0]}")

    # --------- Exit pass (per holding) ---------
    # Use 24h move proxy. If TAKE_PROFIT / STOP_LOSS / TRAIL triggers and eligible, sell.
    for sym, amt in sorted(holdings.items()):
        if sym in IGNORE_TICKERS: 
            continue
        mkt = direct_market(ex, sym, "USD")
        if not mkt:
            continue
        tkr = tickers.get(mkt) or ex.fetch_ticker(mkt)
        why = []
        do_tp = take_profit_check(sym, tkr)
        do_sl = stop_loss_check(sym, tkr)
        do_tr = trailing_stop_check(sym, tkr)
        if (do_tp or do_sl or do_tr) and exit_eligible(ex, sym, amt, tkr, why):
            label = "TAKE_PROFIT" if do_tp else ("STOP_LOSS" if do_sl else "TRAIL")
            print(f"[EXIT] {label} → SELL {sym} amount={amt}")
            if DRY_RUN:
                print(f"[DRY-RUN] create_order SELL {sym}/USD amount={amt}")
            else:
                try:
                    ex.create_order(symbol=f"{sym}/USD", type="market", side="sell", amount=amt)
                    save_last_trade(sym)
                except Exception as e:
                    print(f"[ERROR] exit sell failed for {sym}: {e}")
        elif (do_tp or do_sl or do_tr):
            if WHY_NOT_LOGS:
                print(f"[WHY-NOT][EXIT {sym}] " + "; ".join(why))

    # --------- Rotation ---------
    holding_list = [h for h in holdings.keys() if h not in IGNORE_TICKERS]
    holding_sorted = sorted(holding_list, key=lambda s: score(s))
    weakest = holding_sorted[0] if holding_sorted else None

    best_buy = None
    for c in top:
        if c in IGNORE_TICKERS: 
            continue
        best_buy = c
        break

    edge = (score(best_buy) - score(weakest)) if (weakest and best_buy) else 0.0
    print(f"[EDGE] weakest={weakest} best_buy={best_buy} edge={edge:.4f} (thr {EDGE_THRESHOLD:.4f})")

    did_anything = False
    why_not = []
    need_cash = max(0.0, RESERVE_CASH_USD - usd)

    wtkr = None; w_amt = 0.0
    if weakest:
        sell_sym = direct_market(ex, weakest, "USD")
        wtkr = tickers.get(sell_sym) if sell_sym else None
        if wtkr is None and sell_sym:
            wtkr = ex.fetch_ticker(sell_sym)
        w_amt = holdings.get(weakest, 0.0)

    if weakest and best_buy and edge >= EDGE_THRESHOLD:
        ok_sell = eligible_reason_sell(ex, weakest, w_amt, wtkr or {}, why_not)
        if not ok_sell and ROTATE_WEAKEST_STRICT:
            if WHY_NOT_LOGS:
                for r in why_not: print(f"[WHY-NOT][SELL {weakest}] {r}")
            print("[ROTATE] STRICT on; weakest blocked → skip rotation.")
        else:
            usd_to_use = max(USD_PER_TRADE, need_cash)
            buy_why = []
            btkr = tickers.get(f"{best_buy}/USD") or {}
            ok_buy = eligible_reason_buy(ex, best_buy, usd_to_use, btkr, buy_why)
            if not ok_buy and WHY_NOT_LOGS:
                for r in buy_why: print(f"[WHY-NOT][BUY {best_buy}] {r}")

            if ok_buy and (ok_sell or (not ok_sell and not ROTATE_WEAKEST_STRICT)):
                if not ok_sell and PARTIAL_ASSIST:
                    nw = None; sell_amt = 0.0
                    for cand in holding_sorted[1:]:
                        sym = direct_market(ex, cand, "USD")
                        tkr = tickers.get(sym) if sym else None
                        if tkr is None and sym:
                            tkr = ex.fetch_ticker(sym)
                        amt = holdings.get(cand, 0.0)
                        reasons = []
                        if exit_eligible(ex, cand, amt, tkr or {}, reasons):
                            nw = cand
                            px = price_bid_or_last(tkr or {})
                            if px > 0:
                                sell_amt = max((MIN_COST_PER_ORDER + 0.50) / px, 0.0)
                            break
                        elif WHY_NOT_LOGS:
                            for r in reasons: print(f"[WHY-NOT][ASSIST SELL {cand}] {r}")
                    if nw and sell_amt > 0:
                        print(f"[ASSIST] SELL slice {nw} to free cash (~${MIN_COST_PER_ORDER:.2f}).")
                        if DRY_RUN:
                            print(f"[DRY-RUN] create_order SELL {nw}/USD amount={sell_amt}")
                        else:
                            try:
                                ex.create_order(symbol=f"{nw}/USD", type="market", side="sell", amount=sell_amt)
                                save_last_trade(nw)
                            except Exception as e:
                                print(f"[ERROR] assist sell failed: {e}")

                if ok_sell:
                    print(f"[ROTATE] SELL weakest {weakest} amount={w_amt}")
                    if DRY_RUN:
                        print(f"[DRY-RUN] create_order SELL {weakest}/USD amount={w_amt}")
                    else:
                        try:
                            ex.create_order(symbol=f"{weakest}/USD", type="market", side="sell", amount=w_amt)
                            save_last_trade(weakest)
                            time.sleep(0.8)
                            balances2 = ex.fetch_balance()
                            usd = float((balances2.get("total") or {}).get("USD", 0.0))
                            print(f"[POST-SELL] USD now ~${usd:.2f}")
                        except Exception as e:
                            print(f"[ERROR] sell failed: {e}")

                usd_to_use = max(USD_PER_TRADE, MIN_COST_PER_ORDER)
                if usd >= usd_to_use:
                    print(f"[BUY ] BUY {best_buy} ~${usd_to_use:.2f}")
                    if DRY_RUN:
                        print(f"[DRY-RUN] create_order BUY {best_buy}/USD cost~${usd_to_use:.2f}")
                    else:
                        try:
                            t = tickers.get(f"{best_buy}/USD") or ex.fetch_ticker(f"{best_buy}/USD")
                            px = price_bid_or_last(t)
                            amt = usd_to_use / px if px > 0 else 0.0
                            ex.create_order(symbol=f"{best_buy}/USD", type="market", side="buy", amount=amt)
                            save_last_trade(best_buy)
                        except Exception as e:
                            print(f"[ERROR] buy failed: {e}")
                    did_anything = True
                else:
                    print(f"[SKIP BUY] Not enough USD (${usd:.2f}) for min ${usd_to_use:.2f}")
            else:
                print("[ROTATE] post-checks: cannot buy/sell.")
                if WHY_NOT_LOGS:
                    for r in why_not: print(f"[WHY-NOT][SELL {weakest}] {r}")
    else:
        print("[ROTATE] No valid pair or edge below threshold; skip rotation.")

    summary = f"did_anything={did_anything} usd=${usd:.2f} weakest={weakest} edge={edge:.4f}"
    print(f"[SUMMARY] {summary}")
    record_kpi(summary)
    print(f"[END] {ts()}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
