# trader/crypto_engine.py
# Full-loop crypto engine:
#  - Auto-pick Top-K USD pairs
#  - Entries (market) with position cap, reserve cash, min order guard
#  - Exits: TP, SL, Trailing stop (activate + step)
#  - Daily loss cap (blocks NEW entries after cap is hit; still allows sells)
#  - Persistence: .state/positions.json and .state/daily_loss.json
#
# Works as package (python -m trader.crypto_engine) or as script.

from __future__ import annotations
import os, sys, json, math, time
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Tuple, Optional

# -------- robust import of broker --------
try:
    from trader.broker_crypto_ccxt import CCXTCryptoBroker  # package import
except ModuleNotFoundError:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from broker_crypto_ccxt import CCXTCryptoBroker  # type: ignore

# ---------- tiny utils ----------
UTC = timezone.utc

def now_utc_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def as_bool(s: Optional[str], default: bool) -> bool:
    if s is None: return default
    return s.strip().lower() in ("1","true","yes","y","on")

def as_int(s: Optional[str], default: int) -> int:
    try: return int(s) if s is not None else default
    except Exception: return default

def as_float(s: Optional[str], default: float) -> float:
    try: return float(s) if s is not None else default
    except Exception: return default

def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

STATE_DIR = ".state"
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")
DAILY_LOSS_FILE = os.path.join(STATE_DIR, "daily_loss.json")

# ---------- ENV ----------
def read_env() -> Dict[str, Any]:
    return {
        "DRY_RUN": as_bool(os.getenv("DRY_RUN","true"), True),
        "EXCHANGE_ID": os.getenv("EXCHANGE_ID","kraken"),
        "BASE": os.getenv("BASE_CURRENCY","USD"),
        "UNIVERSE": os.getenv("UNIVERSE","auto"),
        "MAX_POSITIONS": as_int(os.getenv("MAX_POSITIONS","4"), 4),
        "PER_TRADE_USD": as_float(os.getenv("PER_TRADE_USD","25"), 25.0),
        "RESERVE_USD": as_float(os.getenv("RESERVE_USD","100"), 100.0),
        "DAILY_LOSS_CAP_USD": as_float(os.getenv("DAILY_LOSS_CAP_USD","40"), 40.0),

        # order sizing/guards
        "MIN_ORDER_USD": as_float(os.getenv("MIN_ORDER_USD","10"), 10.0),

        # exits
        "TP_PCT": as_float(os.getenv("TP_PCT","0.035"), 0.035),                 # +3.5%
        "SL_PCT": as_float(os.getenv("SL_PCT","0.020"), 0.020),                 # -2.0%
        "TRAIL_ACTIVATE_PCT": as_float(os.getenv("TRAIL_ACTIVATE_PCT","0.025"), 0.025),  # +2.5%
        "TRAIL_STEP_PCT": as_float(os.getenv("TRAIL_STEP_PCT","0.010"), 0.010),          # -1.0% below peak

        # auto-pick knobs (via dispatch + repo variables)
        "AUTO_TOP_K": as_int(os.getenv("AUTO_TOP_K", os.getenv("MAX_POSITIONS","4")), 4),
        "AUTO_MIN_USD_VOL": as_float(os.getenv("AUTO_MIN_USD_VOL","2000000"), 2_000_000.0),
        "AUTO_MIN_PRICE": as_float(os.getenv("AUTO_MIN_PRICE","0.05"), 0.05),
        "AUTO_EXCLUDE": os.getenv("AUTO_EXCLUDE","USDT/USD,USDC/USD,EUR/USD,GBP/USD,USD/USD,SPX/USD,PUMP/USD,BABY/USD,ALKIMI/USD"),
    }

# ---------- state I/O ----------
def load_json(path: str, default: Any) -> Any:
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def today_str() -> str:
    return date.today().isoformat()

# ---------- auto universe ----------
def build_auto_universe(ex, base_quote: str, top_k: int, min_usd_vol: float,
                        min_price: float, exclude_symbols: List[str]) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    try:
        ex.load_markets()
    except Exception as e:
        return [], [f"load_markets failed: {e}"]

    usd_syms = [s for s in ex.markets.keys() if isinstance(s,str) and s.endswith(f"/{base_quote}")]
    if not usd_syms:
        return [], [f"no /{base_quote} symbols found"]

    excl = set(x.strip().upper() for x in exclude_symbols if x.strip())
    usd_syms = [s for s in usd_syms if s.upper() not in excl and not s.upper().startswith(f"{base_quote}/")]
    if not usd_syms:
        return [], ["all USD symbols filtered by exclusions"]

    # fetch tickers (best effort)
    tickers: Dict[str,Any] = {}
    try:
        tickers = ex.fetch_tickers(usd_syms)
    except Exception as e:
        reasons.append(f"fetch_tickers bulk failed: {e}; trying singles up to 50")
        for s in usd_syms[:50]:
            try: tickers[s] = ex.fetch_ticker(s)
            except Exception: pass
    if not tickers:
        return [], ["no tickers returned"]

    scored: List[Tuple[str,float,float,float]] = []
    for sym, t in tickers.items():
        last = t.get("last") or t.get("close")
        pct = t.get("percentage")
        base_vol = t.get("baseVolume")
        quote_vol = t.get("quoteVolume")

        try: price = float(last) if last is not None else None
        except Exception: price = None
        if price is None or price < min_price: continue

        try:
            usd_vol = (float(quote_vol) if quote_vol is not None
                       else (float(base_vol)*price if base_vol is not None else None))
        except Exception:
            usd_vol = None
        if usd_vol is None or usd_vol < min_usd_vol: continue

        try: pct_val = float(pct) if pct is not None else 0.0
        except Exception: pct_val = 0.0

        score = pct_val * math.log(max(usd_vol,1.0) + 1.0)
        scored.append((sym, score, pct_val, usd_vol))

    if not scored:
        reasons.append(f"no symbols passed filters (min_price={min_price}, min_usd_vol={min_usd_vol})")

    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    picks = [s for s,_,_,_ in scored[:max(1, top_k)]]
    return picks, reasons

# ---------- core trading helpers ----------
def fetch_last(ex, symbol: str) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol)
        last = t.get("last") or t.get("close")
        return float(last) if last is not None else None
    except Exception:
        return None

def place_market_buy(broker: CCXTCryptoBroker, symbol: str, notional_usd: float, dry_run: bool) -> Tuple[float,float]:
    """
    Returns (qty, avg_price). In DRY_RUN, simulates using current price.
    """
    ex = broker.ex
    price = fetch_last(ex, symbol)
    if price is None or price <= 0:
        raise RuntimeError(f"{symbol}: no price for buy")

    qty = notional_usd / price
    if dry_run:
        print(f"{now_utc_str()} INFO: [DRY] BUY {symbol} ~{qty:.8f} @ ${price:.2f} (${notional_usd:.2f})")
        return qty, price

    # Live: create market order
    try:
        o = ex.create_order(symbol, type="market", side="buy", amount=qty)
        # Best-effort fill parse
        filled = float(o.get("filled") or qty)
        avg = float(o.get("average") or price)
        print(f"{now_utc_str()} INFO: BUY {symbol} {filled:.8f} @ ${avg:.2f} (${filled*avg:.2f})  orderId={o.get('id')}")
        return filled, avg
    except Exception as e:
        raise RuntimeError(f"{symbol}: buy failed: {e}")

def place_market_sell(broker: CCXTCryptoBroker, symbol: str, qty: float, dry_run: bool) -> Tuple[float,float]:
    """
    Returns (filled_qty, avg_price). In DRY_RUN, simulates using current price.
    """
    ex = broker.ex
    price = fetch_last(ex, symbol)
    if price is None or price <= 0:
        raise RuntimeError(f"{symbol}: no price for sell")

    if dry_run:
        print(f"{now_utc_str()} INFO: [DRY] SELL {symbol} {qty:.8f} @ ${price:.2f} (${qty*price:.2f})")
        return qty, price

    try:
        o = ex.create_order(symbol, type="market", side="sell", amount=qty)
        filled = float(o.get("filled") or qty)
        avg = float(o.get("average") or price)
        print(f"{now_utc_str()} INFO: SELL {symbol} {filled:.8f} @ ${avg:.2f} (${filled*avg:.2f})  orderId={o.get('id')}")
        return filled, avg
    except Exception as e:
        raise RuntimeError(f"{symbol}: sell failed: {e}")

# ---------- main loop ----------
def main() -> int:
    env = read_env()

    print("="*74)
    print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if env["DRY_RUN"] else "🟢 LIVE TRADING")
    print("="*74)
    print(f"{now_utc_str()} INFO: Starting trader in CRYPTO mode. Dry run={env['DRY_RUN']}. Broker=ccxt")

    ensure_dir(STATE_DIR)
    positions: Dict[str, Any] = load_json(POSITIONS_FILE, {})
    daily_loss: Dict[str, Any] = load_json(DAILY_LOSS_FILE, {"day": today_str(), "realized_usd": 0.0})

    # reset daily loss bucket if date rolled
    if daily_loss.get("day") != today_str():
        daily_loss = {"day": today_str(), "realized_usd": 0.0}

    # build broker + read cash
    broker = CCXTCryptoBroker(exchange_id=env["EXCHANGE_ID"], dry_run=env["DRY_RUN"])
    usd = 0.0
    try:
        broker.load_markets()
        usd = broker.usd_cash()
        print(f"{now_utc_str()} INFO: [ccxt] USD/ZUSD balance detected: ${usd:,.2f}")
    except Exception as e:
        print(f"{now_utc_str()} WARN: Could not fetch USD/ZUSD balance: {e}")

    # universe candidates
    uni_env = env["UNIVERSE"].strip().lower()
    if uni_env == "auto":
        ex = broker.ex
        exclude = [x for x in env["AUTO_EXCLUDE"].split(",") if x.strip()]
        candidates, reasons = build_auto_universe(
            ex=ex,
            base_quote=env["BASE"],
            top_k=env["AUTO_TOP_K"],
            min_usd_vol=env["AUTO_MIN_USD_VOL"],
            min_price=env["AUTO_MIN_PRICE"],
            exclude_symbols=exclude,
        )
        if candidates:
            print(f"{now_utc_str()} INFO: Universe (auto): top {len(candidates)} → {candidates}")
        else:
            print(f"{now_utc_str()} INFO: Universe (auto): none selected.")
            for r in reasons: print(f"{now_utc_str()} INFO: reason: {r}")
    else:
        candidates = [u.strip() for u in os.getenv("UNIVERSE","").split(",") if u.strip()]
        print(f"{now_utc_str()} INFO: Universe (manual): {candidates}")

    # -------- EXIT pass (always allowed) --------
    realizations = 0.0
    to_del: List[str] = []
    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0))
        if qty <= 0: 
            to_del.append(sym)
            continue

        price = fetch_last(broker.ex, sym)
        if price is None: 
            print(f"{now_utc_str()} WARN: {sym} no price for exit check")
            continue

        entry = float(p.get("entry", 0.0))
        peak  = float(p.get("peak", entry))
        change = (price - entry) / entry if entry > 0 else 0.0

        # trailing peak update
        if change >= env["TRAIL_ACTIVATE_PCT"]:
            peak = max(peak, price)
            p["peak"] = peak

        reason = None
        if change >= env["TP_PCT"]:
            reason = f"TP hit (+{env['TP_PCT']*100:.2f}%)"
        elif change <= -env["SL_PCT"]:
            reason = f"SL hit (-{env['SL_PCT']*100:.2f}%)"
        elif change >= env["TRAIL_ACTIVATE_PCT"]:
            drop = (peak - price) / peak if peak > 0 else 0.0
            if drop >= env["TRAIL_STEP_PCT"]:
                reason = f"TRAIL drop {drop*100:.2f}% from peak"

        if reason:
            try:
                filled, avg = place_market_sell(broker, sym, qty, env["DRY_RUN"])
                pnl = (avg - entry) * filled
                realizations += pnl
                to_del.append(sym)
                print(f"{now_utc_str()} INFO: EXIT {sym} ({reason}) | entry=${entry:.4f} -> exit=${avg:.4f} | qty={filled:.8f} | pnl=${pnl:.2f}")
            except Exception as e:
                print(f"{now_utc_str()} ERROR: sell {sym} failed: {e}")

    # cleanup closed
    for sym in to_del:
        positions.pop(sym, None)

    # update daily loss tally
    if abs(realizations) > 1e-9:
        daily_loss["realized_usd"] = float(daily_loss.get("realized_usd", 0.0)) + realizations
        print(f"{now_utc_str()} INFO: Realized this run: ${realizations:+.2f} | Day total: ${daily_loss['realized_usd']:+.2f}")
        save_json(DAILY_LOSS_FILE, daily_loss)

    # -------- ENTRY pass (blocked if daily cap reached) --------
    cap_hit = (-float(daily_loss.get("realized_usd",0.0))) >= env["DAILY_LOSS_CAP_USD"]
    if cap_hit:
        print(f"{now_utc_str()} WARN: Daily loss cap reached (${env['DAILY_LOSS_CAP_USD']:.2f}). **Blocking new buys** this run.")
    else:
        # cash available for new entries
        reserved = env["RESERVE_USD"]
        avail_cash = max(0.0, usd - reserved)
        open_positions = len(positions)
        cap_left = max(0, env["MAX_POSITIONS"] - open_positions)
        per_trade = env["PER_TRADE_USD"]

        if cap_left > 0 and per_trade >= env["MIN_ORDER_USD"] and avail_cash >= per_trade:
            # filter candidates to those we don't already hold
            buy_list = [s for s in candidates if s not in positions][:cap_left]
            for sym in buy_list:
                if avail_cash < per_trade:
                    print(f"{now_utc_str()} INFO: Out of allocatable cash (avail=${avail_cash:.2f}).")
                    break
                try:
                    qty, avg = place_market_buy(broker, sym, per_trade, env["DRY_RUN"])
                    positions[sym] = {
                        "qty": qty,
                        "entry": avg,
                        "peak": avg,
                        "ts": now_utc_str(),
                    }
                    avail_cash -= per_trade
                except Exception as e:
                    print(f"{now_utc_str()} ERROR: buy {sym} failed: {e}")
        else:
            print(f"{now_utc_str()} INFO: No entry (cap_left={max(0, env['MAX_POSITIONS'] - len(positions))}, per_trade=${env['PER_TRADE_USD']:.2f}, avail=${max(0.0, usd - env['RESERVE_USD']):.2f})")

    # persist positions
    save_json(POSITIONS_FILE, positions)

    # KPI summary
    open_positions = len(positions)
    cap_left = max(0, env["MAX_POSITIONS"] - open_positions)
    print(f"{now_utc_str()} INFO: KPI SUMMARY | dry_run={env['DRY_RUN']} | open={open_positions} | cap_left={cap_left} | usd=${usd:,.2f} | day_realized=${daily_loss.get('realized_usd',0.0):+.2f}")
    print(f"{now_utc_str()} INFO: Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
