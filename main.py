#!/usr/bin/env python3
import os, time, json, math, re, sys, random, datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Config via environment
# =========================
EXCHANGE_NAME   = os.getenv("EXCHANGE", "kraken").lower()
MODE            = os.getenv("MODE", "live").lower()              # "live" or "dry"
DRY_RUN         = MODE != "live" or os.getenv("DRY_RUN", "false").lower() == "true"

PER_TRADE_USD   = float(os.getenv("PER_TRADE_USD", "10"))
DAILY_CAP_USD   = float(os.getenv("DAILY_CAP_USD", "25"))
DROP_PCT        = float(os.getenv("DROP_PCT", "2.0"))

# Percent-based exits (fallback / legacy)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))
TRAIL_ACTIVATE  = float(os.getenv("TRAILING_ACTIVATE_PCT", "1.0"))
TRAIL_DELTA     = float(os.getenv("TRAILING_DELTA_PCT", "1.0"))
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT", "0"))

# --- Adaptive exits (ATR) ---
ATR_EXITS       = int(os.getenv("ATR_EXITS", "1"))               # 1 = use ATR exits; 0 = use % exits
ATR_PERIOD      = int(os.getenv("ATR_PERIOD", "14"))
ATR_TP_MULT     = float(os.getenv("ATR_TP_MULT", "1.5"))         # take-profit when move >= TP_MULT * ATR
ATR_TRAIL_MULT  = float(os.getenv("ATR_TRAIL_MULT", "1.0"))      # trailing pullback >= TRAIL_MULT * ATR
ATR_ACT_MULT    = float(os.getenv("ATR_ACT_MULT", "0.8"))        # start trailing after move >= ACT_MULT * ATR
ATR_STOP_MULT   = float(os.getenv("ATR_STOP_MULT", "1.2"))       # stop losers when drawdown >= STOP_MULT * ATR

# Entry filter (reduce random buys)
ENTRY_GUARD       = int(os.getenv("ENTRY_GUARD", "1"))           # 1 = on
ENTRY_EMA_SHORT   = int(os.getenv("ENTRY_EMA_SHORT", "9"))
ENTRY_EMA_LONG    = int(os.getenv("ENTRY_EMA_LONG", "21"))
ENTRY_RSI_PERIOD  = int(os.getenv("ENTRY_RSI_PERIOD", "14"))
ENTRY_RSI_MIN     = float(os.getenv("ENTRY_RSI_MIN", "50"))
ENTRY_RSI_MAX     = float(os.getenv("ENTRY_RSI_MAX", "75"))

# Candle timeframe for indicators (ccxt style: 1m, 5m, 15m, 1h, ...)
TIMEFRAME       = os.getenv("TIMEFRAME", "5m")

MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "50"))
PER_ASSET_CAP   = float(os.getenv("PER_ASSET_CAP_USD", "50"))
COOLDOWN_MIN    = int(os.getenv("COOLDOWN_MINUTES", "30"))

AUTO_UNIV_COUNT = int(os.getenv("AUTO_UNIVERSE_COUNT", "500"))
QUOTE           = os.getenv("QUOTE", "USD").upper()
MARKET          = os.getenv("MARKET", "crypto").lower()
SELL_DEBUG      = int(os.getenv("SELL_DEBUG", "0"))

STATE_DIR       = ".state"
POSITIONS_FILE  = os.path.join(STATE_DIR, "positions.json")
COOLDOWN_FILE   = os.path.join(STATE_DIR, "cooldown.json")

os.makedirs(STATE_DIR, exist_ok=True)

def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

positions: Dict[str, Any] = _load_json(POSITIONS_FILE, {})
cooldown: Dict[str, str]  = _load_json(COOLDOWN_FILE, {})

# ---------- Apply temporary Sell Tester overrides ----------
def apply_temp_overrides():
    """Map UI params from the Sell Tester workflow into runtime overrides.

    Supported envs (all optional):
      TEMP_TP_PCT                        -> TAKE_PROFIT_PCT
      TEMP_TRAILING_ACTIVATE_PCT         -> TRAIL_ACTIVATE
      TEMP_TRAILING_DELTA_PCT            -> TRAIL_DELTA
      TEMP_STOP_LOSS_PCT                 -> STOP_LOSS_PCT
      TEMP_ATR_STOP_MULT                 -> ATR_STOP_MULT
      SELL_DEBUG                         -> SELL_DEBUG (1/0)
    """
    global TAKE_PROFIT_PCT, TRAIL_ACTIVATE, TRAIL_DELTA, STOP_LOSS_PCT
    global ATR_STOP_MULT, SELL_DEBUG

    applied = {}

    def _maybe_float(name):
        v = os.getenv(name)
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    m = _maybe_float("TEMP_TP_PCT")
    if m is not None:
        TAKE_PROFIT_PCT = m; applied["TAKE_PROFIT_PCT"] = m

    m = _maybe_float("TEMP_TRAILING_ACTIVATE_PCT")
    if m is not None:
        TRAIL_ACTIVATE = m; applied["TRAIL_ACTIVATE"] = m

    m = _maybe_float("TEMP_TRAILING_DELTA_PCT")
    if m is not None:
        TRAIL_DELTA = m; applied["TRAIL_DELTA"] = m

    m = _maybe_float("TEMP_STOP_LOSS_PCT")
    if m is not None:
        STOP_LOSS_PCT = m; applied["STOP_LOSS_PCT"] = m

    m = _maybe_float("TEMP_ATR_STOP_MULT")
    if m is not None:
        ATR_STOP_MULT = m; applied["ATR_STOP_MULT"] = m

    sd = os.getenv("SELL_DEBUG")
    if sd is not None:
        try:
            SELL_DEBUG = int(sd)
            applied["SELL_DEBUG"] = SELL_DEBUG
        except Exception:
            pass

    if applied:
        # One concise line so you can see overrides at the top of the run.
        print("[sellcfg] applied temp overrides → " +
              ", ".join(f"{k}={v}" for k, v in applied.items()))

# ---------- Auto-heal ----------
def _normalize_lot(x: Any):
    if isinstance(x, dict) and "qty" in x and "cost" in x:
        try:
            return {
                "qty": float(x["qty"]),
                "cost": float(x["cost"]),
                "entry": x.get("entry") or datetime.datetime.utcnow().isoformat(),
                **{k: v for k, v in x.items() if k.startswith("_")}
            }
        except Exception:
            return None
    return None

def normalize_positions():
    changed = False
    if not isinstance(positions, dict):
        print("WARN: positions.json invalid; resetting")
        positions.clear()
        _save_json(POSITIONS_FILE, positions)
        return
    to_delete = []
    for sym, lots in list(positions.items()):
        new_lots: List[dict] = []
        if isinstance(lots, str):
            try:
                lots = json.loads(lots); changed = True
            except Exception:
                print(f"WARN: positions[{sym}] non-JSON string; dropping")
                to_delete.append(sym); continue
        if isinstance(lots, dict):
            lot = _normalize_lot(lots)
            if lot: new_lots = [lot]; changed = True
        elif isinstance(lots, list):
            for l in lots:
                lot = _normalize_lot(l)
                if lot: new_lots.append(lot)
                else: changed = True
        else:
            print(f"WARN: positions[{sym}] unexpected type {type(lots)}; dropping")
            to_delete.append(sym); continue
        if new_lots: positions[sym] = new_lots
        else: to_delete.append(sym)
    for sym in to_delete: positions.pop(sym, None)
    if changed or to_delete:
        print("NOTE: normalized positions.json")
        _save_json(POSITIONS_FILE, positions)

normalize_positions()

# =========================
# Exchange (CCXT)
# =========================
try:
    import ccxt  # type: ignore
except Exception:
    print("ccxt not available; add to requirements.txt", file=sys.stderr)
    ccxt = None

def mk_exchange():
    if EXCHANGE_NAME == "kraken":
        ex = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY", ""),
            "secret": os.getenv("KRAKEN_API_SECRET", ""),
            "enableRateLimit": True,
        })
    else:
        raise RuntimeError(f"Unsupported exchange: {EXCHANGE_NAME}")
    ex.load_markets()
    return ex

exchange = mk_exchange() if ccxt else None

def get_free_usd():
    if not exchange: return 0.0
    try:
        bal = exchange.fetch_free_balance()
        return float(bal.get(QUOTE, 0.0))
    except Exception:
        return 0.0

def market_price(symbol):
    if not exchange: return 0.0
    t = exchange.fetch_ticker(symbol)
    return float(t["last"] or t["close"] or 0.0)

def list_tradeable_pairs():
    if not exchange: return []
    return sorted(set(
        m["symbol"] for m in exchange.markets.values()
        if m.get("spot") and m.get("active") and m.get("quote") == QUOTE
    ))

# =========================
# Indicator helpers (EMA, RSI, ATR)
# =========================
def ema(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) == 0:
        return values[:]
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out

def rsi(values: List[float], period: int) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains, losses = [], []
    for i in range(1, len(values)):
        d = values[i] - values[i-1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out = [50.0]*(period)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else 999999.0
        out.append(100.0 - (100.0 / (1.0 + rs)))
    return out + [50.0]*(len(values)-len(out))

def atr(high: List[float], low: List[float], close: List[float], period: int) -> List[float]:
    if len(close) < period + 2:
        return [0.0] * len(close)
    tr_list = []
    prev_close = close[0]
    for i in range(1, len(close)):
        tr = max(high[i]-low[i], abs(high[i]-prev_close), abs(low[i]-prev_close))
        tr_list.append(tr)
        prev_close = close[i]
    atrs = []
    if len(tr_list) < period:
        return [0.0]*len(close)
    first = sum(tr_list[:period]) / period
    atrs.append(first)
    for i in range(period, len(tr_list)):
        atrs.append((atrs[-1]*(period-1) + tr_list[i]) / period)
    padded = [0.0]*(len(close)-len(atrs)-1) + atrs
    padded.append(atrs[-1] if atrs else 0.0)
    return padded

_ohlcv_cache: Dict[Tuple[str,str,int], Tuple[List[float],List[float],List[float]]] = {}

def fetch_candles(sym: str, timeframe: str, limit: int=200) -> Tuple[List[float],List[float],List[float]]:
    key = (sym, timeframe, limit)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
    if not exchange or not getattr(exchange, "has", {}).get("fetchOHLCV", False):
        return [], [], []
    try:
        raw = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        highs = [float(x[2]) for x in raw]
        lows  = [float(x[3]) for x in raw]
        closes= [float(x[4]) for x in raw]
        _ohlcv_cache[key] = (highs, lows, closes)
        return highs, lows, closes
    except Exception:
        return [], [], []

def indicators(sym: str) -> Dict[str, float]:
    highs, lows, closes = fetch_candles(sym, TIMEFRAME, limit=200)
    if not closes:
        return {"ema_s":0.0, "ema_l":0.0, "rsi":50.0, "atr_pct":0.0}
    ema_s = ema(closes, ENTRY_EMA_SHORT)[-1]
    ema_l = ema(closes, ENTRY_EMA_LONG)[-1]
    r = rsi(closes, ENTRY_RSI_PERIOD)[-1]
    a = atr(highs, lows, closes, ATR_PERIOD)[-1]
    last_close = closes[-1]
    atr_pct = (a/last_close*100.0) if last_close>0 else 0.0
    return {"ema_s":ema_s, "ema_l":ema_l, "rsi":r, "atr_pct":atr_pct}

# =========================
# Ledger helpers
# =========================
def _lots(sym) -> List[dict]:
    v = positions.get(sym, [])
    return v if isinstance(v, list) else []

def total_positions_count():
    return sum(1 for lots in positions.values() if isinstance(lots, list) for _ in lots)

def position_usd_exposure(sym, price):
    lots = _lots(sym)
    qty  = sum(float(l.get("qty", 0.0)) for l in lots if isinstance(l, dict))
    return qty * price

def add_lot(sym, qty, cost):
    lots = _lots(sym)
    lots.append({"qty": float(qty), "cost": float(cost), "entry": datetime.datetime.utcnow().isoformat()})
    positions[sym] = lots
    _save_json(POSITIONS_FILE, positions)

def clear_sym(sym):
    positions.pop(sym, None)
    _save_json(POSITIONS_FILE, positions)

def cooldown_active(sym):
    if sym not in cooldown: return False
    try:
        t = datetime.datetime.fromisoformat(cooldown[sym])
        return (datetime.datetime.utcnow() - t) < datetime.timedelta(minutes=COOLDOWN_MIN)
    except Exception:
        return False

def mark_sold(sym):
    cooldown[sym] = datetime.datetime.utcnow().isoformat()
    _save_json(COOLDOWN_FILE, cooldown)

# =========================
# Tax Ledger
# =========================
try:
    from tax_ledger import TaxLedger
    _ledger = TaxLedger()
    print(f"[boot] TaxLedger ready → {_ledger.ledger_path}")
except Exception as e:
    _ledger = None
    print(f"[warn] TaxLedger not available: {e}")

def _maybe_tax_log_per_lot(sym: str, lots: List[dict], sell_px: float, order_id: Optional[str] = None):
    if DRY_RUN or _ledger is None:
        if DRY_RUN and SELL_DEBUG:
            try:
                profit_preview = sum(max(0.0, (sell_px - float(l["cost"])) * float(l["qty"])) for l in lots)
                print(f"[tax][dry] preview (not recorded) potential profit ~${profit_preview:.2f}")
            except Exception:
                pass
        return
    reserved_total = 0.0
    profit_total = 0.0
    for l in lots:
        try:
            qty = float(l["qty"]); cost = float(l["cost"])
            proceeds = qty * sell_px
            cost_basis = qty * cost
            opened_at = l.get("entry")
            hp_days = None
            if isinstance(opened_at, str):
                try:
                    opened_dt = datetime.datetime.fromisoformat(opened_at.replace("Z",""))
                    hp_days = (datetime.datetime.utcnow() - opened_dt).days
                except Exception:
                    hp_days = None
            res = _ledger.record_sell(
                market=MARKET, symbol=sym, qty=qty, avg_price_usd=sell_px,
                proceeds_usd=proceeds, cost_basis_usd=cost_basis, fees_usd=0.0,
                holding_period_days=hp_days, run_id=os.getenv("GITHUB_RUN_ID"), trade_id=order_id
            )
            profit_total += float(res.get("profit_usd", 0.0))
            reserved_total += float(res.get("reserved_usd", 0.0))
        except Exception as e:
            print(f"[tax][error] lot-record failed for {sym}: {e}")
    print(f"[tax] profit=${profit_total:.2f} reserved=${reserved_total:.2f} "
          f"(rate={getattr(_ledger,'reserve_rate',0):.2f}+{getattr(_ledger,'state_rate',0):.2f}) "
          f"→ {_ledger.ledger_path}")

# =========================
# Market sizing helpers (wallet/min size/precision)
# =========================
def base_asset(sym: str) -> str:
    return sym.split("/")[0]

def free_base_qty(sym: str) -> float:
    if not exchange: return 0.0
    try:
        bal = exchange.fetch_free_balance()
        return float(bal.get(base_asset(sym), 0.0))
    except Exception:
        return 0.0

def market_limits(sym: str) -> dict:
    try:
        return exchange.market(sym).get("limits", {}) or {}
    except Exception:
        return {}

def market_precision(sym: str) -> dict:
    try:
        return exchange.market(sym).get("precision", {}) or {}
    except Exception:
        return {}

def min_trade_qty(sym: str, px: float) -> float:
    lim = market_limits(sym)
    min_amt  = float((lim.get("amount") or {}).get("min") or 0.0)
    min_cost = float((lim.get("cost")   or {}).get("min") or 0.0)
    if px > 0 and min_cost > 0:
        min_amt = max(min_amt, min_cost / px)
    return float(min_amt or 0.0)

def round_qty(sym: str, qty: float) -> float:
    try:
        prec = market_precision(sym).get("amount")
        if isinstance(prec, int):
            return float(f"{qty:.{prec}f}")
    except Exception:
        pass
    return qty

# =========================
# Order helpers
# =========================
def place_buy(sym, usd_amt):
    px = market_price(sym)
    if px <= 0: 
        print(f"[buyerr] {sym}: bad price")
        return False, 0.0, "bad_price"

    qty = usd_amt / px
    min_qty = min_trade_qty(sym, px)
    if min_qty > 0 and qty < min_qty:
        print(f"[buyskip] {sym}: qty {qty:.8f} < min {min_qty:.8f} (usd ${usd_amt:.2f})")
        return False, 0.0, "below_min"

    qty = round_qty(sym, qty)
    if qty <= 0:
        print(f"[buyskip] {sym}: rounded qty is 0")
        return False, 0.0, "rounded_zero"

    if DRY_RUN:
        print(f"BUY {sym}: DRY ~${usd_amt:.2f} @ ~{px:.8f} qty~{qty:.8f}")
        add_lot(sym, qty, px)
        return True, qty, "dry"
    try:
        order = exchange.create_market_buy_order(sym, qty)
        print(f"BUY {sym}: bought {qty:.6f} ~${usd_amt:.2f} (order id {order.get('id','?')})")
        add_lot(sym, qty, px)
        return True, qty, "ok"
    except Exception as e:
        print(f"BUY ERROR {sym}: {e}")
        return False, 0.0, "err"

def place_sell(sym, qty, reason):
    px = market_price(sym)
    if px <= 0:
        return False, 0.0, "bad_price"

    wallet_free = free_base_qty(sym)
    req_qty = float(qty)
    qty = min(req_qty, wallet_free)
    min_qty = min_trade_qty(sym, px)
    qty = round_qty(sym, qty)

    if qty <= 0 or (min_qty > 0 and qty < min_qty):
        print(f"[sellskip] {sym}: request {req_qty:.8f}, wallet {wallet_free:.8f}, min {min_qty:.8f} → skip")
        if wallet_free <= 0:
            clear_sym(sym); mark_sold(sym)
        return False, 0.0, "min_or_wallet"

    usd = qty * px
    lots_snapshot = [l for l in _lots(sym) if isinstance(l, dict) and "qty" in l and "cost" in l]

    if DRY_RUN:
        print(f"SELL {sym}: {reason} (DRY) qty={qty:.6f} ~${usd:.2f}")
        _maybe_tax_log_per_lot(sym, lots_snapshot, px, order_id=None)
        clear_sym(sym); mark_sold(sym)
        return True, usd, "dry"

    try:
        order = exchange.create_market_sell_order(sym, qty)
        oid = order.get('id', '?')
        print(f"SELL {sym}: {reason} qty={qty:.6f} ~${usd:.2f} (order id {oid})")
        _maybe_tax_log_per_lot(sym, lots_snapshot, px, order_id=str(oid))
        clear_sym(sym); mark_sold(sym)
        return True, usd, "ok"
    except Exception as e:
        print(f"SELL ERROR {sym}: {e}")
        return False, 0.0, "err"

# =========================
# Sell engine (ATR or %)
# =========================
def evaluate_sells():
    realized = 0.0
    for sym, lots in list(positions.items()):
        if not isinstance(lots, list) or not lots:
            continue
        valid = [l for l in lots if isinstance(l, dict) and "qty" in l and "cost" in l]
        if not valid:
            continue
        qty = sum(float(l["qty"]) for l in valid)
        if qty <= 0:
            continue
        px = market_price(sym)
        if px <= 0:
            continue

        avg_cost = sum(float(l["qty"])*float(l["cost"]) for l in valid)/qty
        chg_pct = (px - avg_cost) / avg_cost * 100.0

        hi_key = "_high"
        hi = max([float(l.get(hi_key, avg_cost)) for l in valid] + [avg_cost])
        if px > hi:
            for l in valid: l[hi_key] = px
            positions[sym] = valid; _save_json(POSITIONS_FILE, positions)
            hi = px
        pullback = (hi - px) / hi * 100.0 if hi > 0 else 0.0

        # --- Runtime market constraints for audit ---
        wallet_free = free_base_qty(sym)
        min_qty = min_trade_qty(sym, px)
        min_notional = (min_qty * px) if min_qty > 0 else 0.0
        notional = qty * px

        atrp = 0.0
        if ATR_EXITS:
            try:
                atrp = indicators(sym)["atr_pct"]
            except Exception:
                atrp = 0.0

        # SELL DEBUG: compact but rich audit
        if SELL_DEBUG:
            if ATR_EXITS and atrp > 0:
                x_info = (f"ATR%={atrp:.3f}  TP>={ATR_TP_MULT:.2f}*ATR  "
                          f"ACT>={ATR_ACT_MULT:.2f}*ATR  DELTA>={ATR_TRAIL_MULT:.2f}*ATR  "
                          f"STOP>={ATR_STOP_MULT:.2f}*ATR")
            else:
                x_info = (f"TP={TAKE_PROFIT_PCT:.2f}%  ACT={TRAIL_ACTIVATE:.2f}%  "
                          f"DELTA={TRAIL_DELTA:.2f}%  SL={STOP_LOSS_PCT:.2f}%")
            print(f"[AUDIT] {sym} qty={qty:.8f} notional=${notional:.2f} "
                  f"min_sell~${min_notional:.2f} wallet_base={wallet_free:.8f} "
                  f"avg_cost={avg_cost:.8f} px={px:.8f} chg={chg_pct:.2f}% "
                  f"hi={hi:.8f} pullback={pullback:.2f}%  {x_info}")

        # --- Decisions ---
        if ATR_EXITS and atrp > 0:
            tp_thresh    = ATR_TP_MULT    * atrp
            act_thresh   = ATR_ACT_MULT   * atrp
            trail_thresh = ATR_TRAIL_MULT * atrp
            stop_thresh  = ATR_STOP_MULT  * atrp

            if chg_pct >= tp_thresh:
                ok, usd, _ = place_sell(sym, qty, f"ATR_TP {ATR_TP_MULT:.2f}*ATR (~{tp_thresh:.2f}%)")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> ATR_TP fired for {sym}")
                continue

            if chg_pct >= act_thresh and pullback >= trail_thresh:
                ok, usd, _ = place_sell(sym, qty, f"ATR_TRAIL {ATR_TRAIL_MULT:.2f}*ATR (~{trail_thresh:.2f}%)")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> ATR_TRAIL fired for {sym}")
                continue

            if chg_pct <= -stop_thresh:
                ok, usd, _ = place_sell(sym, qty, f"ATR_STOP {ATR_STOP_MULT:.2f}*ATR (~{stop_thresh:.2f}%)")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> ATR_STOP fired for {sym}")
                continue

        else:
            if TAKE_PROFIT_PCT > 0 and chg_pct >= TAKE_PROFIT_PCT:
                ok, usd, _ = place_sell(sym, qty, f"TAKE_PROFIT {TAKE_PROFIT_PCT:.2f}%")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> TAKE_PROFIT fired for {sym}")
                continue

            if TRAIL_ACTIVATE > 0 and chg_pct >= TRAIL_ACTIVATE and pullback >= TRAIL_DELTA:
                ok, usd, _ = place_sell(sym, qty, f"TRAIL_STOP {TRAIL_DELTA:.2f}% from peak")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> TRAIL_STOP fired for {sym}")
                continue

            if STOP_LOSS_PCT > 0 and chg_pct <= -abs(STOP_LOSS_PCT):
                ok, usd, _ = place_sell(sym, qty, f"STOP_LOSS {STOP_LOSS_PCT:.2f}%")
                realized += usd if ok else 0.0
                if SELL_DEBUG: print(f"[sellchk] -> STOP_LOSS fired for {sym}")
                continue

    if realized > 0:
        print(f"REALIZED: freed ~${realized:.2f} from sells")
    return realized

# =========================
# Buy engine (with guards)
# =========================
def entry_ok(sym: str, price: float) -> Tuple[bool, str]:
    if not ENTRY_GUARD:
        return True, "guard_off"
    try:
        ind = indicators(sym)
        ema_ok = ind["ema_s"] > ind["ema_l"]
        r = ind["rsi"]
        r_ok = (ENTRY_RSI_MIN <= r <= ENTRY_RSI_MAX)
        return (ema_ok and r_ok, f"ema_ok={ema_ok} rsi={r:.1f}")
    except Exception as e:
        return False, f"ind_err:{e}"

def can_buy(sym, price, free_usd):
    if cooldown_active(sym): return False, "cooldown"
    if total_positions_count() >= MAX_POSITIONS: return False, "max_positions"
    exposure = position_usd_exposure(sym, price)
    if exposure + PER_TRADE_USD > PER_ASSET_CAP: return False, "per_asset_cap"
    if DAILY_CAP_USD <= 0: return False, "daily_cap_zero"
    ok, why = entry_ok(sym, price)
    if not ok: return False, f"entry:{why}"
    return True, "ok"

# =========================
# Auto-universe
# =========================
def autopick_universe(limit=AUTO_UNIV_COUNT):
    symbols = list_tradeable_pairs()
    def score(sym):
        try:
            t = exchange.fetch_ticker(sym)
            return float(t.get("baseVolume") or 0.0)
        except Exception:
            return 0.0
    ranked = sorted(symbols, key=score, reverse=True)
    if len(ranked) < limit:
        ranked += [s for s in symbols if s not in ranked]
    pick = ranked[:limit]
    print(f"auto_universe: picked {len(pick)} of {len(symbols)} candidates")
    return pick

# =========================
# Main cycle
# =========================
def run_cycle():
    print("=== START TRADING OUTPUT ===")
    print(f"Python {sys.version.split()[0]}")

    apply_temp_overrides()  # <- pick up Sell Tester inputs if present
    evaluate_sells()

    uni = autopick_universe(AUTO_UNIV_COUNT)

    free_usd = get_free_usd()
    print(f"FREE_USD: ${free_usd:.2f}")

    budget = min(DAILY_CAP_USD, free_usd)
    buys_this_run = 0.0

    for sym in uni:
        if budget < PER_TRADE_USD: break
        try:
            price = market_price(sym)
        except Exception:
            continue
        ok, why = can_buy(sym, price, free_usd)
        if not ok:
            if SELL_DEBUG: print(f"[buychk] skip {sym} → {why}")
            continue
        success, qty, _ = place_buy(sym, PER_TRADE_USD)
        if success:
            buys_this_run += PER_TRADE_USD
            budget -= PER_TRADE_USD

    print(f"Budget left after buys: ${budget:.2f}")
    print("=== END TRADING OUTPUT ===")
    return buys_this_run

if __name__ == "__main__":
    try:
        run_cycle()
    except KeyboardInterrupt:
        pass
