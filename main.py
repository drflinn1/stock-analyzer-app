# main.py — Mean-revert day-trader (buy dips, quick TP/SL/trail)
# - Entries: ranks coins by 15m RSI (low) and % below VWAP (low) -> buys most oversold eligible
# - Exits (uniform): TAKE_PROFIT, STOP_LOSS, TRAILING stop, tracked vs our entry price
# - Atomic: won't sell to rotate unless a buy is eligible (and only when fully allocated)
# - Dust ignored; min-cost respected; WHY-NOT logs everywhere

from __future__ import annotations
import os, json, math, csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ---------- env helpers ----------
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

# ---------- CONFIG (env) ----------
EXCHANGE         = (os.getenv("EXCHANGE") or "kraken").lower()
DRY_RUN          = as_bool(os.getenv("DRY_RUN"), True)

# Universe / filters
UNIVERSE         = env_csv("UNIVERSE")
IGNORE_TICKERS   = set(env_csv("IGNORE_TICKERS") or ["USDT","USDC","USD","EUR","GBP"])
MIN_QUOTE_VOL_USD= as_float(os.getenv("MIN_QUOTE_VOL_USD"), 10000.0)
MAX_SPREAD_BPS   = as_float(os.getenv("MAX_SPREAD_BPS"), 150.0)

# Sizing, reserves
TOP_K            = int(os.getenv("TOP_K") or "6")
USD_PER_TRADE    = as_float(os.getenv("USD_PER_TRADE"), 15.0)
MIN_COST_PER_ORDER = as_float(os.getenv("MIN_COST_PER_ORDER"), 5.0)
RESERVE_CASH_USD = as_float(os.getenv("RESERVE_CASH_USD"), 25.0)

# Mean-revert signal params
TIMEFRAME        = os.getenv("TIMEFRAME") or "15m"
RSI_LEN          = int(os.getenv("RSI_LEN") or "14")
VWAP_BARS        = int(os.getenv("VWAP_BARS") or "20")  # ~5 hours on 15m
OVERSOLD_RSI     = as_float(os.getenv("OVERSOLD_RSI"), 35.0)

# Exits (uniform for all positions)
TAKE_PROFIT_PCT  = as_float(os.getenv("TAKE_PROFIT_PCT"), 0.02)   # +2.0%
STOP_LOSS_PCT    = as_float(os.getenv("STOP_LOSS_PCT"),   0.01)   # -1.0%
TRAILING_STOP_PCT= as_float(os.getenv("TRAILING_STOP_PCT"),0.007) # 0.7%

# Rotation behavior when fully allocated
ATOMIC_ROTATION  = as_bool(os.getenv("ATOMIC_ROTATION"), True)
ROTATE_WHEN_FULL = as_bool(os.getenv("ROTATE_WHEN_FULL"), True)   # can swap worst->best when no USD

# Hygiene
COOLDOWN_MINUTES         = int(os.getenv("COOLDOWN_MINUTES") or "10")
SELL_EPS                 = as_float(os.getenv("SELL_EPS"), 0.995)
BUY_EPS                  = as_float(os.getenv("BUY_EPS"),  0.995)
DUST_IGNORE_BELOW_USD    = as_float(os.getenv("DUST_IGNORE_BELOW_USD"), 1.00)
WHY_NOT_LOGS             = as_bool(os.getenv("WHY_NOT_LOGS"), True)

STATE_DIR = Path(".state"); STATE_DIR.mkdir(exist_ok=True)
TRADES_FILE   = STATE_DIR / "last_trades.json"     # per-symbol cooldown
POSITIONS_FILE= STATE_DIR / "positions.json"       # our entries/peaks for exits
KPI_CSV       = STATE_DIR / "kpi_history.csv"

print(
  f"[BOOT] {ts()} EXCHANGE={EXCHANGE} DRY_RUN={DRY_RUN} TOP_K={TOP_K} TF={TIMEFRAME} "
  f"USD_PER_TRADE=${USD_PER_TRADE} MIN_COST=${MIN_COST_PER_ORDER} RESERVE=${RESERVE_CASH_USD} "
  f"RSI_LEN={RSI_LEN} VWAP_BARS={VWAP_BARS} OVERSOLD_RSI={OVERSOLD_RSI} "
  f"TP={TAKE_PROFIT_PCT:.3f} SL={STOP_LOSS_PCT:.3f} TRAIL={TRAILING_STOP_PCT:.3f} "
  f"VOL_USD>={MIN_QUOTE_VOL_USD} SPREAD_BPS<={MAX_SPREAD_BPS} ATOMIC={ATOMIC_ROTATION} "
  f"DUST_IGNORE<${DUST_IGNORE_BELOW_USD}"
)

# ---------- ccxt ----------
try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERROR] ccxt import failed: {e}")

def build_exchange():
    if EXCHANGE != "kraken":
        raise SystemExit("[ERROR] Only Kraken wired for now.")
    return ccxt.kraken({
        "apiKey": os.getenv("KRAKEN_API_KEY",""),
        "secret": os.getenv("KRAKEN_API_SECRET",""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

# ---------- state ----------
def load_json(p: Path) -> dict:
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: pass
    return {}

def save_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, indent=2))

def load_trades(): return load_json(TRADES_FILE)
def save_trade(sym: str):
    d = load_trades(); d[sym.upper()] = now_utc().isoformat(); save_json(TRADES_FILE, d)
def cooldown_active(sym: str) -> bool:
    d = load_trades(); s=sym.upper()
    if s not in d: return False
    return (now_utc() - datetime.fromisoformat(d[s])) < timedelta(minutes=COOLDOWN_MINUTES)

def load_positions(): return load_json(POSITIONS_FILE)
def set_position(sym: str, entry_price: float):
    d = load_positions()
    d[sym] = {"entry": entry_price, "peak": entry_price, "ts": now_utc().isoformat()}
    save_json(POSITIONS_FILE, d)
def clear_position(sym: str):
    d = load_positions()
    if sym in d: del d[sym]
    save_json(POSITIONS_FILE, d)
def update_peak(sym: str, price: float):
    d = load_positions()
    if sym in d:
        d[sym]["peak"] = max(d[sym].get("peak", price), price)
        save_json(POSITIONS_FILE, d)

# ---------- utils ----------
def direct_market(ex, base: str, quote="USD") -> Optional[str]:
    s = f"{base}/USD"; return s if s in ex.markets else None

def spread_bps(t: dict) -> float:
    bid = t.get("bid") or 0.0; ask = t.get("ask") or 0.0
    if bid <= 0 or ask <= 0: return 1e9
    return (ask - bid) / ((ask + bid)/2.0) * 1e4

def price_bid_or_last(t: dict) -> float:
    return t.get("bid") or t.get("last") or t.get("close") or 0.0

def amount_precision(ex, symbol: str) -> int:
    m = ex.market(symbol)
    p = (m.get("precision") or {}).get("amount", None)
    return p if isinstance(p, int) and p >= 0 else 8

def clamp(amount: float, decimals: int) -> float:
    q = 10 ** decimals; return math.floor(max(amount, 0.0)*q)/q

def record_kpi(summary: str):
    exists = KPI_CSV.exists()
    with KPI_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["ts","summary"])
        w.writerow([now_utc().isoformat(), summary])

# ---------- indicators ----------
def rsi(values: List[float], period: int=14) -> Optional[float]:
    if len(values) < period+1: return None
    gains, losses = 0.0, 0.0
    for i in range(1, period+1):
        ch = values[-i] - values[-i-1]
        gains += max(ch, 0.0); losses += max(-ch, 0.0)
    if (gains + losses) == 0: return 50.0
    avg_gain = gains / period; avg_loss = losses / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def vwap_from_bars(bars: List[list], last_n: int=20) -> Optional[float]:
    if len(bars) < max(1, last_n): return None
    sub = bars[-last_n:]
    num = sum((b[4]) * (b[5] or 0.0) for b in sub)  # close * volume
    den = sum((b[5] or 0.0) for b in sub)
    if den <= 0: return None
    return num / den

# ---------- eligibility ----------
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

# ---------- order ops ----------
def do_sell(ex, sym: str, amt: float) -> bool:
    mkt = f"{sym}/USD"; dec = amount_precision(ex, mkt)
    amt = clamp(amt, dec)
    print(f"[SELL] {sym} amount={amt}")
    if DRY_RUN:
        print(f"[DRY-RUN] create_order SELL {mkt} amount={amt}")
        return True
    try:
        ex.create_order(symbol=mkt, type="market", side="sell", amount=amt)
        save_trade(sym); clear_position(sym)
        return True
    except Exception as e:
        print(f"[ERROR] sell failed ({sym}): {e}")
        retry = clamp(amt*0.97, dec)
        if retry > 0 and retry != amt:
            print(f"[RETRY] SELL {sym} amount={retry}")
            try:
                ex.create_order(symbol=mkt, type="market", side="sell", amount=retry)
                save_trade(sym); clear_position(sym)
                return True
            except Exception as e2:
                print(f"[ERROR] retry sell failed: {e2}")
        return False

def do_buy(ex, base: str, tkr: dict, budget: float) -> bool:
    px = price_bid_or_last(tkr)
    if px <= 0: 
        print(f"[WHY-NOT][BUY {base}] no price")
        return False
    amt = (budget * BUY_EPS) / px
    mkt = f"{base}/USD"; dec = amount_precision(ex, mkt)
    amt = clamp(amt, dec)
    print(f"[BUY ] {base} budget~${budget:.2f} → amount={amt}")
    if DRY_RUN:
        print(f"[DRY-RUN] create_order BUY {mkt} amount={amt}")
        set_position(base, px)
        return True
    try:
        ex.create_order(symbol=mkt, type="market", side="buy", amount=amt)
        save_trade(base); set_position(base, px)
        return True
    except Exception as e:
        print(f"[ERROR] buy failed ({base}): {e}")
        return False

# ---------- strategy core ----------
def mean_revert_score(ex, symbol: str, bars: List[list]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (rsi15m, pct_to_vwap, last_price). Lower RSI and more negative pct_to_vwap => more oversold."""
    closes = [b[4] for b in bars]
    r = rsi(closes, RSI_LEN)
    v = vwap_from_bars(bars, VWAP_BARS)
    last = closes[-1] if closes else None
    pct_to_vwap = None
    if v and last:
        pct_to_vwap = (last / v) - 1.0
    return r, pct_to_vwap, last

def pick_best_buy_mean_revert(ex, top: List[str], tickers: dict) -> Tuple[Optional[str], Optional[dict], dict]:
    """Scan Top-K; compute RSI/VWAP; log; return best eligible by (RSI asc, pct_to_vwap asc)."""
    budget = max(USD_PER_TRADE, MIN_COST_PER_ORDER) * BUY_EPS
    rankings = []
    print("[SCAN_MEAN_REVERT] candidates (RSI, %toVWAP, spread, OK?)")
    for base in top:
        sym = f"{base}/USD"
        try:
            bars = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=max(RSI_LEN+VWAP_BARS+5, 60))
        except Exception as e:
            print(f"  - {base:<8} fetch_ohlcv error: {e}")
            continue
        r, pct_v, last = mean_revert_score(ex, sym, bars)
        t = tickers.get(sym) or ex.fetch_ticker(sym)
        why=[]; ok = buy_eligible(ex, base, budget, t, why)
        bps = spread_bps(t)
        r_s = f"{r:5.1f}" if r is not None else " None"
        pv_s= f"{(pct_v or 0.0)*100:6.2f}%" if pct_v is not None else "  None "
        flag = "OK" if ok else "NO"
        print(f"  - {base:<8} rsi={r_s}  %vwap={pv_s}  spread={bps:>4.0f}bps  {flag} {'; '.join(why)}")
        if ok and r is not None and pct_v is not None:
            rankings.append((r, pct_v, base, t))
    if not rankings: 
        return None, None, {}
    # Lower RSI first, then more negative pct_to_vwap
    rankings.sort(key=lambda x: (x[0], x[1]))
    rsi_val, pv, best, tkr = rankings[0]
    meta = {"rsi": rsi_val, "pct_to_vwap": pv}
    return best, tkr, meta

def fully_allocated(usd: float) -> bool:
    return (usd - RESERVE_CASH_USD) < max(USD_PER_TRADE, MIN_COST_PER_ORDER)

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

    # Universe: USD spot, active, vol/spread filters
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
        universe = sorted({
            (m.get("base") or "").upper()
            for _, m in ex.markets.items()
            if m.get("quote")=="USD" and (m.get("spot", True) or m.get("type")=="spot")
        })
        universe = [s for s in universe if market_ok(s)]

    # Rank top-K by "oversold": RSI asc then % to VWAP asc
    ranked = []
    for b in universe:
        sym = f"{b}/USD"
        try:
            bars = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=max(RSI_LEN+VWAP_BARS+5, 60))
            r, pv, _ = mean_revert_score(ex, sym, bars)
            if r is None or pv is None: continue
            ranked.append((r, pv, b))
        except Exception:
            continue
    ranked.sort(key=lambda x: (x[0], x[1]))
    top = [b for _,__,b in ranked[:max(1, TOP_K)]]
    print("[RANK] Top oversold (RSI, %vwap):")
    for r, pv, b in ranked[:max(1, TOP_K)]:
        print(f"  - {b:<8} rsi={r:5.1f}  %vwap={(pv*100):6.2f}%")
    print(f"[RANK] Top {TOP_K}: {top}")

    # Pick the best buy candidate
    best_buy, btkr, meta = pick_best_buy_mean_revert(ex, top, tickers)

    # Uniform exits on all positions
    did = False
    pos = load_positions()
    for sym, amt in holdings.items():
        if sym in IGNORE_TICKERS: continue
        mkt = direct_market(ex, sym, "USD")
        if not mkt: continue
        t = tickers.get(mkt) or ex.fetch_ticker(mkt)
        px = price_bid_or_last(t)
        if px <= 0: continue
        info = pos.get(sym)
        if not info: 
            # If we inherited a position w/o state, initialize and skip exits this run
            set_position(sym, px)
            print(f"[STATE] init position for {sym} @ {px}")
            continue
        entry = float(info.get("entry", px))
        peak  = float(info.get("peak", entry))
        update_peak(sym, px)
        pnl = (px / entry) - 1.0
        drawdown = 1.0 - (px / max(peak, 1e-9))
        do_tp = pnl >= TAKE_PROFIT_PCT
        do_sl = pnl <= -STOP_LOSS_PCT
        do_tr = (drawdown >= TRAILING_STOP_PCT) and (px > entry)  # trail only if in profit
        if not (do_tp or do_sl or do_tr):
            continue
        # sell eligibility (dust/min-cost guard)
        total_amt = float(amt)
        free_amt  = (free.get(sym) if free is not None else None)
        sell_amt  = (float(free_amt) if (free_amt is not None and free_amt > 0) else total_amt) * SELL_EPS
        why=[]
        if not exit_eligible(ex, sym, sell_amt, t, why):
            if WHY_NOT_LOGS: print(f"[WHY-NOT][EXIT {sym}] {'; '.join(why)}")
            continue
        label = "TP" if do_tp else ("SL" if do_sl else "TRAIL")
        print(f"[EXIT] {label} {sym}: pnl={pnl:+.2%} dd={drawdown:.2%} → sell")
        did |= do_sell(ex, sym, sell_amt)

    # Decide to buy and/or rotate
    usd_buyable = usd - RESERVE_CASH_USD
    can_buy = (best_buy is not None) and (usd_buyable >= max(USD_PER_TRADE, MIN_COST_PER_ORDER))
    if can_buy:
        print(f"[ENTRY] BUY {best_buy} (rsi={meta.get('rsi'):.1f}, %vwap={(meta.get('pct_to_vwap')*100):.2f}%)")
        did |= do_buy(ex, best_buy, btkr, max(USD_PER_TRADE, MIN_COST_PER_ORDER))
        return finish(usd, did, best_buy)

    if not can_buy and ROTATE_WHEN_FULL and best_buy:
        # Try to free USD by selling the MOST overbought eligible holding: highest RSI, above VWAP
        if not holdings:
            return finish(usd, did, None)
        candidates = []
        for h in holdings.keys():
            if h in IGNORE_TICKERS: continue
            sym = f"{h}/USD"
            try:
                bars = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=max(RSI_LEN+VWAP_BARS+5, 60))
            except Exception:
                continue
            r, pv, _ = mean_revert_score(ex, sym, bars)
            if r is None or pv is None: continue
            # Prefer overbought (RSI high, price above VWAP)
            candidates.append((-(pv), r, h))  # pv negative is below VWAP; we want ABOVE, so use -pv
        if candidates:
            candidates.sort(reverse=True)  # highest -pv (i.e., most above vwap), then higher RSI
            _,__,weakest = candidates[0]
            mkt = direct_market(ex, weakest, "USD")
            if mkt:
                t = tickers.get(mkt) or ex.fetch_ticker(mkt)
                total_amt = float(holdings.get(weakest, 0.0))
                free_amt  = (free.get(weakest) if free is not None else None)
                sell_amt  = (float(free_amt) if (free_amt is not None and free_amt > 0) else total_amt) * SELL_EPS
                why=[]
                if ATOMIC_ROTATION and not buy_eligible(ex, best_buy, max(USD_PER_TRADE, MIN_COST_PER_ORDER)*BUY_EPS, btkr, why):
                    print(f"[ROTATE] skip — buy not eligible: {'; '.join(why)}")
                else:
                    why=[] 
                    if exit_eligible(ex, weakest, sell_amt, t, why):
                        print(f"[ROTATE] SELL {weakest} → BUY {best_buy}")
                        did |= do_sell(ex, weakest, sell_amt)
                        did |= do_buy(ex, best_buy, btkr, max(USD_PER_TRADE, MIN_COST_PER_ORDER))
                    elif WHY_NOT_LOGS:
                        print(f"[WHY-NOT][ROTATE sell {weakest}] {'; '.join(why)}")

    return finish(usd, did, best_buy)

def finish(usd, did, best):
    summary = f"did_anything={did} usd=${usd:.2f} best={best}"
    print(f"[SUMMARY] {summary}")
    record_kpi(summary)
    print(f"[END] {ts()}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
