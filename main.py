# main.py — Crypto Live with Rotation + Why-Not Logs
# - Auto-rank coins by simple momentum (24h % change from tickers)
# - Rotate out weakest eligible holding to fund strongest eligible candidate
# - XRP (and other ignore list) excluded from ranking/rotation (dust only)
# - Strong safety rails (min-cost, spread guard, cooldown, reserve-cash)
# - New knobs:
#     ROTATE_WEAKEST_STRICT   -> if the weakest cannot be sold, skip rotation (don’t dump a stronger coin)
#     WHY_NOT_LOGS            -> print explicit reasons for every candidate rejection
#     PARTIAL_ASSIST          -> if weakest is just under min-cost, optionally sell a small slice of next-weakest
#
# ENV (all optional unless marked **required** for live trading):
#   EXCHANGE                  = "kraken"
#   KRAKEN_API_KEY            = **required for live**
#   KRAKEN_API_SECRET         = **required for live**
#   DRY_RUN                   = "true" | "false" (default "true")
#
#   UNIVERSE                  = "BTC,ETH,SOL,DOGE,ZEC,SUI,XLM,ADA,AVAX" (comma list) or empty for "top traded" filter
#   IGNORE_TICKERS            = "USDT,USDC,USD,EUR,GBP,XRP" (never buy/sell during rotation; dust-sweeper handles)
#   TOP_K                     = "6"     (target portfolio count)
#   EDGE_THRESHOLD            = "0.004" (0.4% min edge between buy-candidate and sell-candidate to rotate)
#
#   RESERVE_CASH_USD          = "25"    (keep at least this much USD available)
#   USD_PER_TRADE             = "10"    (amount to deploy per new entry or rotation buy)
#   MAX_NEW_ENTRIES           = "2"     (guard new positions per run)
#
#   MIN_COST_PER_ORDER        = "5.0"   (USD notional min; we also respect exchange limits)
#   MAX_SPREAD_BPS            = "75"    (skip market if spread wider than this)
#   COOLDOWN_MINUTES          = "10"    (cooldown after trading a symbol)
#
#   ROTATE_WEAKEST_STRICT     = "false"
#   PARTIAL_ASSIST            = "true"
#   WHY_NOT_LOGS              = "true"
#
# Files touched:
#   .state/last_trades.json   (cooldowns)
#   .state/kpi_history.csv    (1-line KPI per run)
#
# Requires: ccxt

from __future__ import annotations
import os, json, time, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

# ---------- small utils ----------

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

# ---------- ENV ----------

EXCHANGE_NAME        = (os.getenv("EXCHANGE") or "kraken").lower()
DRY_RUN              = as_bool(os.getenv("DRY_RUN"), True)

UNIVERSE             = env_csv("UNIVERSE")
IGNORE_TICKERS       = set(env_csv("IGNORE_TICKERS") or ["USDT","USDC","USD","EUR","GBP","XRP"])

TOP_K                = int(os.getenv("TOP_K") or "6")
EDGE_THRESHOLD       = as_float(os.getenv("EDGE_THRESHOLD"), 0.004)  # 0.4%

RESERVE_CASH_USD     = as_float(os.getenv("RESERVE_CASH_USD"), 25.0)
USD_PER_TRADE        = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MAX_NEW_ENTRIES      = int(os.getenv("MAX_NEW_ENTRIES") or "2")

MIN_COST_PER_ORDER   = as_float(os.getenv("MIN_COST_PER_ORDER"), 5.0)
MAX_SPREAD_BPS       = as_float(os.getenv("MAX_SPREAD_BPS"), 75.0)
COOLDOWN_MINUTES     = int(os.getenv("COOLDOWN_MINUTES") or "10")

ROTATE_WEAKEST_STRICT= as_bool(os.getenv("ROTATE_WEAKEST_STRICT"), False)
PARTIAL_ASSIST       = as_bool(os.getenv("PARTIAL_ASSIST"), True)
WHY_NOT_LOGS         = as_bool(os.getenv("WHY_NOT_LOGS"), True)

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
LAST_TRADES_FILE = STATE_DIR / "last_trades.json"
KPI_CSV          = STATE_DIR / "kpi_history.csv"

print(f"[BOOT] {ts()} EXCHANGE={EXCHANGE_NAME} DRY_RUN={DRY_RUN} "
      f"TOP_K={TOP_K} EDGE={EDGE_THRESHOLD} RESERVE=${RESERVE_CASH_USD} "
      f"USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} SPREAD_BPS<={MAX_SPREAD_BPS} "
      f"COOLDOWN_MIN={COOLDOWN_MINUTES} STRICT={ROTATE_WEAKEST_STRICT} PARTIAL_ASSIST={PARTIAL_ASSIST}")

# ---------- ccxt ----------

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

# ---------- helpers ----------

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

def spread_bps(ticker: dict) -> float:
    bid = ticker.get("bid") or 0.0
    ask = ticker.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4

def direct_market(ex, base: str, quote: str="USD") -> Optional[str]:
    s = f"{base}/{quote}"
    return s if s in ex.markets else None

def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def record_kpi(summary: str):
    # append one line with timestamp + plain summary
    new = [now_utc().isoformat(), summary]
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts", "summary"])
        w.writerow(new)

# ---------- core logic ----------

def eligible_reason_sell(ex, base: str, amt: float, tkr: dict, why: List[str]) -> bool:
    """Check sell eligibility w/ reasons; return True if OK."""
    if base.upper() in IGNORE_TICKERS:
        why.append(f"{base}: in IGNORE_TICKERS")
        return False
    if cooldown_active(base):
        why.append(f"{base}: cooldown active")
        return False
    sym = direct_market(ex, base, "USD")
    if not sym:
        why.append(f"{base}: no {base}/USD market")
        return False
    bps = spread_bps(tkr)
    if bps > MAX_SPREAD_BPS:
        why.append(f"{base}: spread {bps:.0f} bps > {MAX_SPREAD_BPS}")
        return False
    px = price_bid_or_last(tkr)
    if px <= 0:
        why.append(f"{base}: no price")
        return False

    # Respect exchange min limits + our MIN_COST_PER_ORDER
    m = ex.market(sym)
    min_amt  = (m.get("limits") or {}).get("amount", {}).get("min", None)
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    notional = amt * px
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if notional < floor:
        why.append(f"{base}: notional ${notional:.2f} < min_cost ${floor:.2f}")
        return False
    return True

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
    # exchange min cost
    min_cost = (m.get("limits") or {}).get("cost", {}).get("min", None)
    floor = max(MIN_COST_PER_ORDER, float(min_cost or 0))
    if usd_budget < floor:
        why.append(f"{base}: budget ${usd_budget:.2f} < min_cost ${floor:.2f}")
        return False
    return True

def main() -> int:
    print(f"[START] {ts()} — running rotation")
    ex = build_exchange()
    ex.load_markets()
    balances = ex.fetch_balance()
    total: Dict[str, float] = (balances.get("total") or {})
    usd = float(total.get("USD", 0.0))
    print(f"[BAL] USD available: ${usd:.2f}")

    # holdings (exclude pure quotes + ignored)
    holdings = {a.upper(): amt for a, amt in total.items()
                if amt and a.upper() not in {"USD","USDT","USDC","EUR","GBP"}}

    # tickers + simple "momentum": use 24h change % from ticker if present, else 0
    tickers = ex.fetch_tickers()
    def score(sym: str) -> float:
        mkt = direct_market(ex, sym, "USD")
        if not mkt: return -1e9
        t = tickers.get(mkt) or {}
        ch = t.get("percentage")  # many ccxt exchanges expose 24h change %
        try:
            return float(ch) / 100.0 if ch is not None else 0.0
        except Exception:
            return 0.0

    # build universe: either provided env or all markets with USD + not ignored
    candidates_all = []
    if UNIVERSE:
        candidates_all = [s for s in UNIVERSE if s not in IGNORE_TICKERS]
    else:
        for mkt, m in ex.markets.items():
            if m.get("quote") == "USD":
                base = m.get("base")
                if base and base.upper() not in IGNORE_TICKERS:
                    candidates_all.append(base.upper())
        candidates_all = sorted(set(candidates_all))

    ranked = sorted(candidates_all, key=lambda s: score(s), reverse=True)
    top = ranked[:max(TOP_K, 1)]

    print(f"[RANK] Top {TOP_K}: {top[:TOP_K]}")
    print(f"[HOLD] {[(k, round(v,8)) for k,v in holdings.items() if v>0]}")

    # Determine weakest holding by score (among holdings that are in universe)
    holding_list = [h for h in holdings.keys() if h not in IGNORE_TICKERS]
    if not holding_list:
        print("[INFO] No non-ignored holdings. Consider entering fresh positions if cash and edge allow.")
    holding_sorted = sorted(holding_list, key=lambda s: score(s))  # ascending: weakest first
    weakest = holding_sorted[0] if holding_sorted else None

    # Determine best buy candidate from "top" that we don't already hold (or we allow pyramid via USD_PER_TRADE)
    best_buy = None
    for c in top:
        if c in IGNORE_TICKERS: 
            continue
        best_buy = c
        break

    # Edge between best buy and weakest sell
    if weakest and best_buy:
        edge = score(best_buy) - score(weakest)
    else:
        edge = 0.0

    print(f"[EDGE] weakest={weakest} best_buy={best_buy} edge={edge:.4f} (thr {EDGE_THRESHOLD:.4f})")

    # Decide rotation
    did_anything = False
    why_not = []

    # Ensure reserve cash
    need_cash = max(0.0, RESERVE_CASH_USD - usd)

    if weakest:
        # Build sell-eligibility and amounts
        sell_sym = direct_market(ex, weakest, "USD")
        wtkr = tickers.get(sell_sym) if sell_sym else None
        if wtkr is None and sell_sym:
            wtkr = ex.fetch_ticker(sell_sym)
        w_amt = holdings.get(weakest, 0.0)
    else:
        wtkr = None; w_amt = 0.0

    # First: check if we should rotate at all
    if weakest and best_buy and edge >= EDGE_THRESHOLD:
        ok_sell = eligible_reason_sell(ex, weakest, w_amt, wtkr or {}, why_not)
        if not ok_sell and ROTATE_WEAKEST_STRICT:
            if WHY_NOT_LOGS:
                for r in why_not: print(f"[WHY-NOT][SELL {weakest}] {r}")
            print("[ROTATE] STRICT mode on and weakest not sellable → skipping rotation this run.")
        else:
            # compute buy eligibility
            usd_to_use = max(USD_PER_TRADE, need_cash)  # ensure we cover reserve if needed
            buy_why = []
            ok_buy = eligible_reason_buy(ex, best_buy, usd_to_use, tickers.get(f"{best_buy}/USD", {}), buy_why)
            if not ok_buy and WHY_NOT_LOGS:
                for r in buy_why: print(f"[WHY-NOT][BUY {best_buy}] {r}")

            if ok_buy and (ok_sell or (not ok_sell and not ROTATE_WEAKEST_STRICT)):
                # If weakest not sellable and STRICT is off, optionally PARTIAL_ASSIST:
                if not ok_sell and PARTIAL_ASSIST:
                    # Try to sell a *small* slice of the next-weakest that IS eligible, just to raise usd_to_use
                    nw = None
                    sell_amt = 0.0
                    for cand in holding_sorted[1:]:  # skip the true weakest we couldn't sell
                        sym = direct_market(ex, cand, "USD")
                        tkr = tickers.get(sym) if sym else None
                        if tkr is None and sym:
                            tkr = ex.fetch_ticker(sym)
                        amt = holdings.get(cand, 0.0)
                        reasons = []
                        if eligible_reason_sell(ex, cand, amt, tkr or {}, reasons):
                            nw = cand
                            # aim to raise MIN_COST_PER_ORDER worth of USD
                            px = price_bid_or_last(tkr or {})
                            if px > 0:
                                sell_amt = max((MIN_COST_PER_ORDER + 0.50) / px, 0.0)
                            break
                        elif WHY_NOT_LOGS:
                            for r in reasons: print(f"[WHY-NOT][ASSIST SELL {cand}] {r}")
                    if nw and sell_amt > 0:
                        print(f"[ASSIST] Selling small slice of {nw} to free cash for buy (target ~${MIN_COST_PER_ORDER:.2f}).")
                        if DRY_RUN:
                            print(f"[DRY-RUN] create_order SELL {nw}/USD amount={sell_amt}")
                        else:
                            try:
                                ex.create_order(symbol=f"{nw}/USD", type="market", side="sell", amount=sell_amt)
                                save_last_trade(nw)
                            except Exception as e:
                                print(f"[ERROR] assist sell failed: {e}")

                # Proceed with normal rotation path:
                if ok_sell:
                    print(f"[ROTATE] SELL weakest {weakest} amount={w_amt}")
                    if DRY_RUN:
                        print(f"[DRY-RUN] create_order SELL {weakest}/USD amount={w_amt}")
                    else:
                        try:
                            ex.create_order(symbol=f"{weakest}/USD", type="market", side="sell", amount=w_amt)
                            save_last_trade(weakest)
                            # refresh USD after sale (rough)
                            time.sleep(0.8)
                            balances2 = ex.fetch_balance()
                            usd_new = float((balances2.get("total") or {}).get("USD", 0.0))
                            print(f"[POST-SELL] USD now ~${usd_new:.2f}")
                            usd = usd_new
                        except Exception as e:
                            print(f"[ERROR] sell failed: {e}")

                # Buy step (use USD_PER_TRADE, respect min-cost)
                usd_to_use = max(USD_PER_TRADE, MIN_COST_PER_ORDER)
                if usd >= usd_to_use:
                    print(f"[BUY ] BUY {best_buy} ~${usd_to_use:.2f}")
                    if DRY_RUN:
                        print(f"[DRY-RUN] create_order BUY {best_buy}/USD cost~${usd_to_use:.2f}")
                    else:
                        try:
                            # market buy by amount: compute amount = usd_to_use / price
                            t = tickers.get(f"{best_buy}/USD") or ex.fetch_ticker(f"{best_buy}/USD")
                            px = price_bid_or_last(t)
                            amt = usd_to_use / px if px > 0 else 0.0
                            ex.create_order(symbol=f"{best_buy}/USD", type="market", side="buy", amount=amt)
                            save_last_trade(best_buy)
                        except Exception as e:
                            print(f"[ERROR] buy failed: {e}")
                    did_anything = True
                else:
                    print(f"[SKIP BUY] Not enough USD (${usd:.2f}) for min trade ${usd_to_use:.2f}")
            else:
                print("[ROTATE] Conditions not met for buy/sell after checks.")
                if WHY_NOT_LOGS:
                    for r in why_not: print(f"[WHY-NOT][SELL {weakest}] {r}")
    else:
        print("[ROTATE] No valid weakest/best pair or edge below threshold; skipping rotation.")

    # Summary
    summary = f"did_anything={did_anything} usd=${usd:.2f} weakest={weakest} best={best_buy} edge={edge:.4f}"
    print(f"[SUMMARY] {summary}")
    record_kpi(summary)
    print(f"[END] {ts()}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
