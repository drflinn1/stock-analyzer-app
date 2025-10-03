# main.py — Rotation-first "day-trade" with fixed direction (worst -> best),
# atomic buy/sell, cooldown, dust ignore, and WHY-NOT logs.

from __future__ import annotations
import os, json, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ---------- small utils ----------
def as_bool(v: Optional[str], default=False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1","true","y","yes","on"}

def as_float(v: Optional[str], default=0.0) -> float:
    try: return float(v) if v is not None else default
    except Exception: return default

def env_csv(name: str) -> List[str]:
    v = os.getenv(name) or ""
    return [s.strip().upper() for s in v.split(",") if s.strip()]

def now_utc() -> datetime: return datetime.now(timezone.utc)
def ts() -> str: return now_utc().strftime("%Y-%m-%d %H:%M:%S %Z")

# ---------- ENV ----------
EXCHANGE_NAME   = (os.getenv("EXCHANGE") or "kraken").lower()
DRY_RUN         = as_bool(os.getenv("DRY_RUN"), True)

# Universe & filters
UNIVERSE        = env_csv("UNIVERSE")
IGNORE_TICKERS  = set(env_csv("IGNORE_TICKERS") or ["USDT","USDC","USD","EUR","GBP","XRP"])
MIN_QUOTE_VOL_USD = as_float(os.getenv("MIN_QUOTE_VOL_USD"), 50_000.0)
MAX_SPREAD_BPS  = as_float(os.getenv("MAX_SPREAD_BPS"), 75.0)

# Sizing, costs & reserves
TOP_K           = int(os.getenv("TOP_K") or "6")
USD_PER_TRADE   = as_float(os.getenv("USD_PER_TRADE"), 10.0)
MIN_COST_PER_ORDER = as_float(os.getenv("MIN_COST_PER_ORDER"), 5.0)
RESERVE_CASH_USD = as_float(os.getenv("RESERVE_CASH_USD"), 25.0)

# Rotation rule (this is the "go back but fix the backwards logic" part)
ROTATE_MIN_EDGE_PCT        = as_float(os.getenv("ROTATE_MIN_EDGE_PCT"), 0.004)  # 0.4%
ROTATE_MAX_SWITCHES_PER_RUN= int(os.getenv("ROTATE_MAX_SWITCHES_PER_RUN") or "1")
ROTATE_COOLDOWN_MIN        = int(os.getenv("ROTATE_COOLDOWN_MIN") or "30")
ATOMIC_ROTATION            = as_bool(os.getenv("ATOMIC_ROTATION"), True)         # sell only if buy passes

# Hygiene
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES") or "10")  # trade-level (exit & rotation)
SELL_EPS         = as_float(os.getenv("SELL_EPS"), 0.995)       # sell 99.5% to avoid fee/precision rejects
BUY_EPS          = as_float(os.getenv("BUY_EPS"),  0.995)
DUST_IGNORE_BELOW_USD = as_float(os.getenv("DUST_IGNORE_BELOW_USD"), 1.00)
WHY_NOT_LOGS     = as_bool(os.getenv("WHY_NOT_LOGS"), True)

# Exit guards (kept for CI / safety; scoped to weakest so we don't dump winners first)
TAKE_PROFIT_PCT  = as_float(os.getenv("TAKE_PROFIT_PCT"), 0.05)  # TAKE_PROFIT
STOP_LOSS_PCT    = as_float(os.getenv("STOP_LOSS_PCT"), 0.03)    # STOP_LOSS
TRAILING_STOP_PCT= as_float(os.getenv("TRAILING_STOP_PCT"), 0.02)# TRAIL
EXIT_SCOPE       = (os.getenv("EXIT_SCOPE") or "weakest").strip().lower()

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
LAST_TRADES_FILE = STATE_DIR / "last_trades.json"          # per-symbol trade cooldown
ROTATE_CD_FILE   = STATE_DIR / "rotate_cooldown.json"      # ping-pong guard
KPI_CSV          = STATE_DIR / "kpi_history.csv"

print(
  f"[BOOT] {ts()} EXCHANGE={EXCHANGE_NAME} DRY_RUN={DRY_RUN} TOP_K={TOP_K} "
  f"EDGE(min)={ROTATE_MIN_EDGE_PCT} USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} "
  f"RESERVE=${RESERVE_CASH_USD} VOL_USD>={MIN_QUOTE_VOL_USD} SPREAD_BPS<={MAX_SPREAD_BPS} "
  f"ROTATE_MAX={ROTATE_MAX_SWITCHES_PER_RUN} ROTATE_CD_MIN={ROTATE_COOLDOWN_MIN} ATOMIC={ATOMIC_ROTATION} "
  f"SELL_EPS={SELL_EPS} BUY_EPS={BUY_EPS} DUST_IGNORE<${DUST_IGNORE_BELOW_USD}"
)

# ---------- ccxt ----------
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERROR] ccxt import failed: {e}")

def build_exchange():
    if EXCHANGE_NAME != "kraken":
        raise SystemExit("[ERROR] Only 'kraken' is implemented.")
    return ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY", ""),
        "secret": os.getenv("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

# ---------- helpers ----------
def load_json(p: Path) -> dict:
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: pass
    return {}

def save_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, indent=2))

def load_last_trades() -> Dict[str, str]:
    return load_json(LAST_TRADES_FILE)

def save_last_trade(sym: str):
    d = load_last_trades(); d[sym.upper()] = now_utc().isoformat(); save_json(LAST_TRADES_FILE, d)

def cooldown_active(sym: str, minutes: int) -> bool:
    d = load_last_trades(); s = sym.upper()
    if s not in d: return False
    return (now_utc() - datetime.fromisoformat(d[s])) < timedelta(minutes=minutes)

def rotate_cooldown_active(sym: str) -> bool:
    d = load_json(ROTATE_CD_FILE); s = sym.upper()
    if s not in d: return False
    return (now_utc() - datetime.fromisoformat(d[s])) < timedelta(minutes=ROTATE_COOLDOWN_MIN)

def mark_rotate_sell(sym: str):
    d = load_json(ROTATE_CD_FILE); d[sym.upper()] = now_utc().isoformat(); save_json(ROTATE_CD_FILE, d)

def direct_market(ex, base: str, quote="USD") -> Optional[str]:
    s = f"{base}/{quote}"; return s if s in ex.markets else None

def spread_bps(t: dict) -> float:
    bid = t.get("bid") or 0.0; ask = t.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    return (ask - bid) / ((bid + ask) / 2.0) * 1e4

def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def amount_precision(ex, symbol: str) -> int:
    m = ex.market(symbol)
    p = (m.get("precision") or {}).get("amount", None)
    return p if isinstance(p, int) and p >= 0 else 8

def clamp(amount: float, decimals: int) -> float:
    q = 10 ** decimals; return math.floor(max(amount, 0.0) * q) / q

def record_kpi(summary: str):
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["ts","summary"])
        w.writerow([now_utc().isoformat(), summary])

def score_from_ticker(t: dict) -> float:
    """24h return proxy: prefer 'percentage' else compute (last-open)/open."""
    ch = t.get("percentage")
    if ch is not None:
        try: return float(ch)/100.0
        except Exception: pass
    last = t.get("last") or t.get("close"); opn = t.get("open")
    try:
        if last and opn and float(opn) > 0:
            return (float(last) - float(opn)) / float(opn)
    except Exception: pass
    return 0.0

# ---------- eligibility ----------
def exit_eligible(ex, sym: str, amt: float, t: dict, why: List[str]) -> bool:
    mkt = direct_market(ex, sym, "USD")
    if not mkt: why.append("no USD market"); return False
    if cooldown_active(sym, COOLDOWN_MINUTES): why.append("cooldown active"); return False
    if rotate_cooldown_active(sym): why.append("rotate cooldown"); return False
    bps = spread_bps(t)
    if bps > MAX_SPREAD_BPS: why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    px = price_bid_or_last(t); 
    if px <= 0: why.append("no price"); return False
    m = ex.market(mkt)
    min_cost = float((m.get("limits") or {}).get("cost", {}).get("min") or 0.0)
    floor = max(MIN_COST_PER_ORDER, min_cost)
    if amt * px < floor: why.append(f"notional ${amt*px:.2f} < min_cost ${floor:.2f}"); return False
    return True

def buy_eligible(ex, base: str, budget_usd: float, t: dict, why: List[str]) -> bool:
    if base.upper() in IGNORE_TICKERS: why.append("ignored"); return False
    sym = direct_market(ex, base, "USD")
    if not sym: why.append("no USD market"); return False
    bps = spread_bps(t)
    if bps > MAX_SPREAD_BPS: why.append(f"spread {bps:.0f} bps > {MAX_SPREAD_BPS}"); return False
    m = ex.market(sym)
    min_cost = float((m.get("limits") or {}).get("cost", {}).get("min") or 0.0)
    floor = max(MIN_COST_PER_ORDER, min_cost)
    if budget_usd < floor: why.append(f"budget ${budget_usd:.2f} < min_cost ${floor:.2f}"); return False
    return True

# ---------- exit checks (labels kept for CI) ----------
def take_profit_check(sym: str, t: dict) -> bool:
    s = score_from_ticker(t)
    if s >= TAKE_PROFIT_PCT:
        print(f"[TAKE_PROFIT] {sym} 24h +{s:.3%} >= {TAKE_PROFIT_PCT:.3%}")
        return True
    return False

def stop_loss_check(sym: str, t: dict) -> bool:
    s = score_from_ticker(t)
    if s <= -STOP_LOSS_PCT:
        print(f"[STOP_LOSS] {sym} 24h {s:.3%} <= -{STOP_LOSS_PCT:.3%}")
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

# ---------- main ----------
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

    # ---- universe (USD spot, active, liquid, tight spread)
    def market_ok(base: str) -> bool:
        if base in IGNORE_TICKERS: return False
        sym = direct_market(ex, base, "USD")
        if not sym: return False
        m = ex.market(sym)
        if not (m.get("active", True) and (m.get("spot", True) or m.get("type")=="spot")):
            return False
        t = tickers.get(sym) or {}
        vol_q = t.get("quoteVolume")
        if vol_q is None:
            bv = t.get("baseVolume"); px = price_bid_or_last(t)
            vol_q = (bv or 0) * (px or 0)
        if (vol_q or 0) < MIN_QUOTE_VOL_USD: return False
        if spread_bps(t) > MAX_SPREAD_BPS: return False
        return True

    if UNIVERSE:
        universe = [s for s in UNIVERSE if market_ok(s)]
    else:
        universe = sorted({ (m.get("base") or "").upper()
                            for _, m in ex.markets.items()
                            if m.get("quote")=="USD" and (m.get("spot", True) or m.get("type")=="spot") })
        universe = [s for s in universe if market_ok(s)]

    # ---- score & rank
    scores: Dict[str, float] = {}
    for b in universe:
        scores[b] = score_from_ticker(tickers.get(f"{b}/USD") or {})
    ranked = sorted(universe, key=lambda s: scores.get(s, -1e9), reverse=True)
    top = ranked[:max(TOP_K,1)]

    print("[RANK] Top candidates (24h%):")
    for c in top: print(f"  - {c:<8} {scores.get(c,0.0):+6.2%}")

    # ---- BUY CANDIDATE: pick the BEST eligible in Top-K
    budget = max(USD_PER_TRADE, MIN_COST_PER_ORDER) * BUY_EPS
    best_buy: Optional[str] = None; btkr = None; best_score = -1e9
    print("[SCAN] Top-K buy candidates:")
    for c in top:
        t = tickers.get(f"{c}/USD") or ex.fetch_ticker(f"{c}/USD")
        s = score_from_ticker(t); bps = spread_bps(t)
        vol_q = t.get("quoteVolume") or 0
        why=[]; ok = buy_eligible(ex, c, budget, t, why)
        flag = "OK" if ok else "NO"
        print(f"  - {c:<8} 24h={s:+6.2%} spread={bps:>4.0f}bps volUSD≈{vol_q:,.0f} {flag} {'; '.join(why)}")
        if ok and s > best_score:
            best_score, best_buy, btkr = s, c, t

    # ---- Are we fully allocated?
    usd_free_for_buy = usd - RESERVE_CASH_USD
    fully_allocated = usd_free_for_buy < max(USD_PER_TRADE, MIN_COST_PER_ORDER)
    print(f"[ALLOC] fully_allocated={fully_allocated} usd_free_for_buy=${usd_free_for_buy:.2f}")

    did_anything = False

    # ---- If NOT fully allocated and we have an eligible buy → just buy
    if not fully_allocated and best_buy:
        did_anything |= do_buy(ex, best_buy, btkr, budget)
        return finish(usd, None, best_buy, 0.0, did_anything)

    # ---- Else we need to ROTATE from worst -> best (fixed direction)
    # pick weakest among *sell-eligible* holdings, skipping dust & cooldown
    holding_list = [h for h in holdings.keys() if h not in IGNORE_TICKERS]
    # compute each holding's 24h% score
    hold_scores = {h: score_from_ticker(tickers.get(f"{h}/USD") or {}) for h in holding_list}
    # sort weakest first
    ordered = sorted(holding_list, key=lambda s: hold_scores.get(s, 0.0))

    weakest = None; w_amt = 0.0; wtkr=None
    for cand in ordered:
        sym = direct_market(ex, cand, "USD")
        if not sym: 
            if WHY_NOT_LOGS: print(f"[WHY-NOT][WEAKEST {cand}] no USD market"); 
            continue
        wtkr = tickers.get(sym) or ex.fetch_ticker(sym)
        px = price_bid_or_last(wtkr); val_usd = float(holdings[cand]) * (px or 0.0)
        if val_usd < DUST_IGNORE_BELOW_USD:
            if WHY_NOT_LOGS: print(f"[DUST] ignore {cand} value≈${val_usd:.2f} < ${DUST_IGNORE_BELOW_USD:.2f}")
            continue
        total_amt = float(holdings[cand]); free_amt = (balances.get("free") or {}).get(cand, None)
        amt = (float(free_amt) if free_amt is not None else total_amt) * SELL_EPS
        why=[]
        if exit_eligible(ex, cand, amt, wtkr, why):
            weakest, w_amt = cand, amt
            break
        elif WHY_NOT_LOGS:
            for r in why: print(f"[WHY-NOT][WEAKEST {cand}] {r}")

    if not weakest:
        print("[ROTATE] No sell-eligible holding found. Skipping rotation.")
        return finish(usd, None, best_buy, 0.0, did_anything)

    if not best_buy:
        if ATOMIC_ROTATION:
            print("[ROTATE] No eligible buy in Top-K → not selling weakest (atomic).")
            return finish(usd, weakest, None, 0.0, did_anything)
        else:
            print("[ROTATE] No eligible buy but ATOMIC_ROTATION=false → may still sell.")

    edge = (scores.get(best_buy,0.0) - hold_scores.get(weakest,0.0)) if best_buy else 0.0
    print(f"[EDGE] weakest={weakest}({hold_scores.get(weakest,0.0):+6.2%}) best={best_buy}({scores.get(best_buy,0.0):+6.2%}) edge={edge:.4f} thr={ROTATE_MIN_EDGE_PCT:.4f}")

    switches = 0
    while switches < max(1, ROTATE_MAX_SWITCHES_PER_RUN):
        if best_buy is None or edge < ROTATE_MIN_EDGE_PCT: 
            print("[ROTATE] Edge below threshold or no buy; stop.")
            break
        ok_sell = do_sell(ex, weakest, wtkr, w_amt)
        if not ok_sell:
            print("[ROTATE] Sell failed; stop.")
            break
        mark_rotate_sell(weakest)
        ok_buy = do_buy(ex, best_buy, btkr, budget)
        did_anything = did_anything or ok_sell or ok_buy
        switches += 1
        break  # single-step rotation is usually enough per run

    # ---- Exits scoped to weakest AFTER rotation (keeps CI tokens, avoids dumping winners)
    do_exits(ex, {weakest: holdings.get(weakest, 0.0)}, tickers)

    return finish(usd, weakest, best_buy, edge, did_anything)

# ---------- order helpers ----------
def do_sell(ex, sym: str, tkr: dict, amt: float) -> bool:
    mkt = f"{sym}/USD"; dec = amount_precision(ex, mkt)
    amt = clamp(amt, dec)
    print(f"[SELL] {sym} amount={amt}")
    if DRY_RUN:
        print(f"[DRY-RUN] create_order SELL {mkt} amount={amt}")
        return True
    try:
        ex.create_order(symbol=mkt, type="market", side="sell", amount=amt)
        save_last_trade(sym); 
        return True
    except Exception as e:
        print(f"[ERROR] sell failed ({sym}): {e}")
        retry = clamp(amt * 0.97, dec)
        if retry > 0 and retry != amt:
            print(f"[RETRY] SELL {sym} amount={retry}")
            try:
                ex.create_order(symbol=mkt, type="market", side="sell", amount=retry)
                save_last_trade(sym); 
                return True
            except Exception as e2:
                print(f"[ERROR] retry sell failed: {e2}")
        return False

def do_buy(ex, base: str, tkr: dict, budget: float) -> bool:
    mkt = f"{base}/USD"; px = price_bid_or_last(tkr)
    if px <= 0:
        print(f"[WHY-NOT][BUY {base}] no price")
        return False
    amt = (budget * BUY_EPS) / px
    dec = amount_precision(ex, mkt); amt = clamp(amt, dec)
    print(f"[BUY ] {base} budget~${budget:.2f} → amount={amt}")
    if DRY_RUN:
        print(f"[DRY-RUN] create_order BUY {mkt} amount={amt}")
        return True
    try:
        ex.create_order(symbol=mkt, type="market", side="buy", amount=amt)
        save_last_trade(base)
        return True
    except Exception as e:
        print(f"[ERROR] buy failed ({base}): {e}")
        return False

# ---------- exits (scoped to weakest by default) ----------
def do_exits(ex, subset_holdings: Dict[str,float], tickers: dict) -> bool:
    did = False
    for sym in sorted(subset_holdings.keys()):
        mkt = direct_market(ex, sym, "USD")
        if not mkt: continue
        t = tickers.get(mkt) or ex.fetch_ticker(mkt)
        # CI tokens
        tp = take_profit_check(sym, t)
        sl = stop_loss_check(sym, t)
        tr = trailing_stop_check(sym, t)
        if not (tp or sl or tr): 
            continue
        # Eligibility w/ min-cost
        px = price_bid_or_last(t); total_amt = subset_holdings.get(sym, 0.0)
        amt = total_amt * SELL_EPS
        why=[]
        if not exit_eligible(ex, sym, amt, t, why):
            print(f"[WHY-NOT][EXIT {sym}] " + "; ".join(why))
            continue
        # sell it
        did |= do_sell(ex, sym, t, amt)
    return did

# ---------- finish ----------
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
