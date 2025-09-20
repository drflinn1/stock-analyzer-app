import os, sys, math, time, json, csv, pathlib, datetime as dt
from typing import Dict, List, Tuple
import ccxt
import pandas as pd
import numpy as np

# ---------- Helpers ----------
def env_str(k, d=""): return os.environ.get(k, d)
def env_f(k, d=0.0):  return float(os.environ.get(k, d))
def env_i(k, d=0):    return int(float(os.environ.get(k, d)))
def now_ts():         return int(time.time())
def utcnow():         return dt.datetime.utcnow()

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_CSV = DATA_DIR / "equity_history.csv"
TAX_LEDGER = DATA_DIR / "tax_ledger.csv"
POS_META    = DATA_DIR / "pos_meta.json"

# ---------- TAX ledger fields ----------
TAX_FIELDS = ["ts","event","side","symbol","qty","price","notional","fee","fee_ccy","order_id","note"]

# ---------- Read env ----------
MODE = env_str("MODE","live")
DRY  = env_str("DRY_RUN","true").lower()=="true"
EXCHANGE = env_str("EXCHANGE","kraken")
BASE = env_str("BASE_CCY","USD")

AUTO_UNIVERSE = env_str("AUTO_UNIVERSE","true").lower()=="true"
UNIVERSE_TOP_N = env_i("UNIVERSE_TOP_N",6)
MAX_CONCURRENT_POS = env_i("MAX_CONCURRENT_POS",5)
MIN_LIQ_USD_24H = env_f("MIN_LIQ_USD_24H",1_000_000)
MOMENTUM_LOOKBACK_H = env_i("MOMENTUM_LOOKBACK_H",24)
MIN_PRICE_USD = env_f("MIN_PRICE_USD",0.01)
MIN_NOTIONAL_USD = env_f("MIN_NOTIONAL_USD",5)

TF_FAST = env_str("TF_FAST","15m")
TF_SLOW = env_str("TF_SLOW","1h")
EMA_SHORT = env_i("EMA_SHORT",20)
EMA_LONG  = env_i("EMA_LONG",50)
RSI_LEN   = env_i("RSI_LEN",14)
RSI_BUY   = env_f("RSI_BUY",60)  # defensive for now
RSI_SELL  = env_f("RSI_SELL",45)

ATR_LEN   = env_i("ATR_LEN",14)
RISK_PER_TRADE_PCT = env_f("RISK_PER_TRADE_PCT",1.6)
PER_TRADE_USD_CAP  = env_f("PER_TRADE_USD_CAP",75)
MAX_TRADES_PER_RUN = env_i("MAX_TRADES_PER_RUN",6)
PORTFOLIO_MAX_EXPOSURE_PCT = env_f("PORTFOLIO_MAX_EXPOSURE_PCT",75)

ATR_MULT_TP = env_f("ATR_MULTIPLIER_TP",1.0)
ATR_MULT_TS = env_f("ATR_MULTIPLIER_TS",1.4)
ATR_MULT_SL = env_f("ATR_MULTIPLIER_SL",1.8)
TRAIL_ACTIVATE_AT_TP_FRAC = env_f("TRAIL_ACTIVATE_AT_TP_FRAC",0.5)

STALE_MAX_HOURS     = env_i("STALE_MAX_HOURS",24)
STALE_RSI_MAX_BARS  = env_i("STALE_RSI_MAX_BARS",6)
STALE_MIN_GAIN_PCT  = env_f("STALE_MIN_GAIN_PCT",-3)

PYRAMID_ENABLED         = env_str("PYRAMID_ENABLED","true").lower()=="true"
PYRAMID_MAX_ADDS        = env_i("PYRAMID_MAX_ADDS",2)
PYRAMID_ADD_AT_GAIN_PCT = env_f("PYRAMID_ADD_AT_GAIN_PCT",2.0)
PYRAMID_STEP_ATR        = env_f("PYRAMID_STEP_ATR",0.75)

DAILY_CAP_USD = env_f("DAILY_CAP_USD",300)
MIN_FREE_CASH_USD = env_f("MIN_FREE_CASH_USD",15)
SLIPPAGE_PCT = env_f("SLIPPAGE_PCT",0.10)

# --- NEW: buy-fee safety buffer (Option A)
FEE_BUY_BUFFER_PCT = env_f("FEE_BUY_BUFFER_PCT", 0.5)

DD_TRIP = env_f("DRAWDOWN_PCT_TRIP",15)
DD_COOLDOWN_RUNS = env_i("DRAWDOWN_COOLDOWN_RUNS",8)

WRITE_EQUITY_CSV = env_str("WRITE_EQUITY_CSV","true").lower()=="true"
PRINT_EQUITY_ONE_LINER = env_str("PRINT_EQUITY_ONE_LINER","true").lower()=="true"

# Guard tokens required by your guard job:
TAKE_PROFIT_TOKEN = "TAKE_PROFIT"
STOP_LOSS_TOKEN   = "STOP_LOSS"

# ---------- Exchange ----------
def make_exchange():
    if EXCHANGE.lower()=="kraken":
        ex = ccxt.kraken({
            "apiKey": env_str("KRAKEN_API_KEY",""),
            "secret": env_str("KRAKEN_API_SECRET",""),
            "enableRateLimit": True,
        })
    else:
        raise RuntimeError("Only Kraken wired here.")
    return ex

ex = make_exchange()
MARKETS = ex.load_markets()

# ---------- IO ----------
def append_equity_csv(ts:int, equity:float):
    write_header = not EQUITY_CSV.exists()
    with EQUITY_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["ts","equity"])
        w.writerow([ts, f"{equity:.2f}"])

def append_tax(row: Dict):
    write_header = not TAX_LEDGER.exists()
    with TAX_LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TAX_FIELDS)
        if write_header: w.writeheader()
        w.writerow(row)

def _fee_from_order(od):
    fee = 0.0; fee_ccy = None
    f = (od or {}).get("fee")
    if f:
        fee = float(f.get("cost") or 0)
        fee_ccy = f.get("currency")
    fl = (od or {}).get("fees") or []
    if fl and fee == 0 and isinstance(fl, list):
        fee = float(fl[0].get("cost") or 0)
        fee_ccy = fl[0].get("currency")
    return fee, fee_ccy

def load_pos_meta():
    if POS_META.exists():
        return json.loads(POS_META.read_text())
    return {}

def save_pos_meta(meta):
    POS_META.write_text(json.dumps(meta, indent=2, sort_keys=True))

# ---------- Market data ----------
def fetch_ohlcv(symbol, timeframe, limit=200):
    for _ in range(3):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            time.sleep(1)
    raise

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def indicators(df: pd.DataFrame):
    df["ema_s"] = df["c"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_l"] = df["c"].ewm(span=EMA_LONG, adjust=False).mean()
    delta = df["c"].diff()
    up = np.where(delta>0, delta, 0.0)
    down = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(RSI_LEN).mean()
    roll_down = pd.Series(down).rolling(RSI_LEN).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi"] = 100 - (100/(1+rs))
    tr = pd.concat([
        (df["h"]-df["l"]).abs(),
        (df["h"]-df["c"].shift(1)).abs(),
        (df["l"]-df["c"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    return df

# ---------- Universe & ranking ----------
def list_markets_usd():
    syms = []
    for s, m in MARKETS.items():
        if m.get("spot") and (m.get("quote") in ["USD","USDT"]) and (m.get("active") or True):
            syms.append(s)
    return syms

def score_symbol(sym)->Tuple[float,dict]:
    try:
        ticker = ex.fetch_ticker(sym)
        last = float(ticker["last"] or 0)
        quote_vol = float(ticker.get("quoteVolume") or 0)
        if last < MIN_PRICE_USD: return -1e9, {}
        if quote_vol < MIN_LIQ_USD_24H: return -1e9, {}
        change = float(ticker.get("percentage") or 0.0)
        score = change + 0.000001*quote_vol
        return score, {"last":last,"liq":quote_vol,"pct":change}
    except Exception:
        return -1e9, {}

def rank_universe(cands: List[str])->List[str]:
    scored = []
    for s in cands:
        sc, meta = score_symbol(s)
        if meta: scored.append((sc, s))
    scored.sort(reverse=True)
    return [s for _,s in scored[:UNIVERSE_TOP_N]]

# ---------- Portfolio / balance ----------
def portfolio_and_equity():
    bal = ex.fetch_balance()
    total = bal.get("total",{})
    free  = bal.get("free",{})
    usd_free = float(free.get(BASE, 0) or 0)
    usd_total = float(total.get(BASE, 0) or 0)
    equity = usd_total
    positions = {}
    for asset, qty in total.items():
        if asset in [BASE, "USDT", "USD.S"]:
            continue
        q = float(qty or 0)
        if q <= 0: continue
        for sym in (f"{asset}/{BASE}", f"{asset}/USDT"):
            try:
                price = float(ex.fetch_ticker(sym)["last"])
                positions[sym] = q
                equity += q * price
                break
            except Exception:
                continue
    return positions, usd_free, equity

def portfolio_exposure(positions: Dict[str,float])->float:
    _, _, equity = portfolio_and_equity()
    exp = 0.0
    for sym, qty in positions.items():
        try:
            px = float(ex.fetch_ticker(sym)["last"])
            exp += qty * px
        except Exception:
            pass
    return 0.0 if equity<=0 else 100.0 * exp / equity

# ---------- Orders ----------
def place_order(side, symbol, qty):
    if qty <= 0: return None
    if DRY: return {"id":"DRY", "side":side, "symbol":symbol, "qty":qty}
    try:
        return ex.create_market_buy_order(symbol, qty) if side=="buy" else ex.create_market_sell_order(symbol, qty)
    except Exception as e:
        print(f"[order] fail {side} {symbol} {qty}: {e}")
        return None

# ---------- Risk / sizing ----------
def size_by_atr(symbol, price, atr, equity, step_mult=1.0):
    if atr<=0 or price<=0: return 0.0, 0.0
    risk_dollars = equity * (RISK_PER_TRADE_PCT/100.0) * step_mult
    unit_risk = ATR_MULT_SL * atr
    if unit_risk<=0: return 0.0, 0.0
    qty = (risk_dollars / unit_risk)
    notional = qty * price
    notional = min(notional, PER_TRADE_USD_CAP)
    qty = notional / price
    return max(qty, 0.0), notional

# ---------- Exchange min guards ----------
def market_limits(symbol):
    m = MARKETS.get(symbol)
    if not m:
        ex.load_markets(True); m = ex.markets.get(symbol)
    amt_min = float(((m or {}).get("limits") or {}).get("amount",{}).get("min") or 0)
    cost_min = float(((m or {}).get("limits") or {}).get("cost",{}).get("min") or 0)
    return max(amt_min, 0.0), max(cost_min, 0.0)

def tradeable(symbol, qty, price):
    amt_min, exch_cost_min = market_limits(symbol)
    notional = qty * price
    min_cost = max(MIN_NOTIONAL_USD, exch_cost_min)
    ok = (qty >= max(amt_min, 0.0)) and (notional >= min_cost)
    return ok, amt_min, min_cost, notional

# ---------- Position meta ----------
pos_meta = load_pos_meta()
def get_meta(sym):
    return pos_meta.get(sym, {"entry_ts":None,"entry_px":None,"add_count":0,"last_add_px":None})
def set_meta(sym, **kwargs):
    m = get_meta(sym); m.update(kwargs); pos_meta[sym]=m; save_pos_meta(pos_meta)

# ---------- Stale / pyramid ----------
def is_stale(sym, df_fast: pd.DataFrame, meta)->bool:
    if meta.get("entry_ts"):
        age_h = (utcnow() - dt.datetime.fromtimestamp(meta["entry_ts"])).total_seconds()/3600.0
        if age_h >= STALE_MAX_HOURS: return True
    rsi_tail = df_fast["rsi"].tail(STALE_RSI_MAX_BARS)
    if len(rsi_tail)==STALE_RSI_MAX_BARS and (rsi_tail < RSI_SELL).all(): return True
    entry_px = meta.get("entry_px")
    if entry_px:
        pnl_pct = 100.0*(df_fast["c"].iloc[-1]/entry_px - 1.0)
        if pnl_pct <= STALE_MIN_GAIN_PCT: return True
    return False

def pyramid_should_add(sym, df_fast: pd.DataFrame, meta)->bool:
    if not PYRAMID_ENABLED: return False
    if meta.get("add_count",0) >= PYRAMID_MAX_ADDS: return False
    entry_px = meta.get("last_add_px") or meta.get("entry_px")
    if not entry_px: return False
    c = df_fast["c"].iloc[-1]
    gain_pct = 100.0*(c/entry_px - 1.0)
    return gain_pct >= PYRAMID_ADD_AT_GAIN_PCT

# ---------- Drawdown breaker ----------
def allow_new_entries_by_dd():
    if not EQUITY_CSV.exists(): return True
    df = pd.read_csv(EQUITY_CSV)
    if df.empty: return True
    e = float(df["equity"].iloc[-1]); hwm = float(df["equity"].max())
    dd = 0.0 if hwm<=0 else 100.0*(1.0 - e/hwm)
    return dd < DD_TRIP

def equity_one_liner(now_e):
    hwm = now_e
    if EQUITY_CSV.exists():
        try:
            df = pd.read_csv(EQUITY_CSV)
            if not df.empty: hwm = max(now_e, df["equity"].max())
        except Exception: pass
    dd = 0.0 if hwm<=0 else 100.0*(1.0 - now_e/hwm)
    return hwm, dd

# ---------- Main run ----------
def run():
    print("=== START TRADING OUTPUT ===")
    print(f"Python {sys.version.split()[0]}")
    positions, usd_free, equity = portfolio_and_equity()
    print(f"[audit] hydrated {len(positions)} symbols from wallet for evaluation")

    # SELL PASS
    for sym, qty in list(positions.items()):
        try:
            df_f = indicators(to_df(fetch_ohlcv(sym, TF_FAST, limit=ATR_LEN*4+60))).dropna()
            c = df_f["c"].iloc[-1]; atr = df_f["atr"].iloc[-1]
            m = get_meta(sym)

            if is_stale(sym, df_f, m):
                ok, amt_min, min_cost, notional = tradeable(sym, qty, c)
                if not ok:
                    print(f"[sellskip] {sym}: qty={qty:.8g} ${notional:.2f} < min (amount>={amt_min} cost>={min_cost}) -> hold as dust")
                else:
                    od = place_order("sell", sym, qty)
                    if od:
                        fee, fee_ccy = _fee_from_order(od)
                        print(f"SELL STALE_EXIT {sym}: qty={qty:.8f} ~${qty*c:.2f}")
                        append_tax({
                            "ts": now_ts(),"event":"TRADE","side":"sell","symbol":sym,
                            "qty":qty,"price":c,"notional":qty*c,"fee":fee,"fee_ccy":fee_ccy or BASE,
                            "order_id": od.get('id'),"note":"STALE_EXIT"
                        })
                        set_meta(sym, add_count=0, entry_ts=None, entry_px=None, last_add_px=None)
                continue

            entry_px = m.get("entry_px")
            if entry_px and atr>0:
                tp = entry_px + ATR_MULT_TP*atr
                sl = entry_px - ATR_MULT_SL*atr
                if c >= entry_px + TRAIL_ACTIVATE_AT_TP_FRAC*ATR_MULT_TP*atr:
                    trail = c - ATR_MULT_TS*atr
                    sl = max(sl, trail)
                if c >= tp or c <= sl:
                    ok, amt_min, min_cost, notional = tradeable(sym, qty, c)
                    if not ok:
                        print(f"[sellskip] {sym}: qty={qty:.8g} ${notional:.2f} < min (amount>={amt_min} cost>={min_cost}) -> hold as dust")
                    else:
                        tag = TAKE_PROFIT_TOKEN if c >= tp else STOP_LOSS_TOKEN
                        od = place_order("sell", sym, qty)
                        if od:
                            fee, fee_ccy = _fee_from_order(od)
                            print(f"SELL {tag} {sym}: px={c:.6f} qty={qty:.8f} ~${qty*c:.2f}")
                            append_tax({
                                "ts": now_ts(),"event":"TRADE","side":"sell","symbol":sym,
                                "qty":qty,"price":c,"notional":qty*c,"fee":fee,"fee_ccy":fee_ccy or BASE,
                                "order_id": od.get('id'),"note": tag
                            })
                            set_meta(sym, add_count=0, entry_ts=None, entry_px=None, last_add_px=None)
        except Exception as e:
            print(f"[sellpass] {sym} error: {e}")

    # refresh after sells
    positions, usd_free, equity = portfolio_and_equity()

    # PYRAMID adds
    if PYRAMID_ENABLED:
        for sym, qty in list(positions.items()):
            try:
                df_f = indicators(to_df(fetch_ohlcv(sym, TF_FAST, limit=ATR_LEN*4+60))).dropna()
                c = df_f["c"].iloc[-1]; atr = df_f["atr"].iloc[-1]
                m = get_meta(sym)
                if pyramid_should_add(sym, df_f, m):
                    exposure_pct = portfolio_exposure(positions)
                    if exposure_pct >= PORTFOLIO_MAX_EXPOSURE_PCT: continue
                    qty_add, notional = size_by_atr(sym, c, atr, equity, step_mult=PYRAMID_STEP_ATR)
                    ok, amt_min, min_cost, _ = tradeable(sym, qty_add, c)
                    # ---- FEES BUFFER APPLIED HERE ----
                    cash_needed = notional * (1.0 + FEE_BUY_BUFFER_PCT/100.0)
                    if (not ok) or (notional < MIN_NOTIONAL_USD) or ((usd_free - cash_needed) < MIN_FREE_CASH_USD):
                        continue
                    od = place_order("buy", sym, qty_add)
                    if od:
                        fee, fee_ccy = _fee_from_order(od)
                        print(f"PYRAMID + {sym}: +{qty_add:.8f} ~${notional:.2f} (cash_needed=${cash_needed:.2f})")
                        append_tax({
                            "ts": now_ts(),"event":"TRADE","side":"buy","symbol":sym,
                            "qty":qty_add,"price":c,"notional":qty_add*c,"fee":fee,"fee_ccy":fee_ccy or BASE,
                            "order_id": od.get('id'),"note":"PYRAMID"
                        })
                        set_meta(sym, add_count=m.get("add_count",0)+1, last_add_px=c)
                        usd_free -= cash_needed
                        positions[sym] = qty + qty_add
            except Exception as e:
                print(f"[pyramid] {sym} error: {e}")

    # ENTRY PASS
    if allow_new_entries_by_dd():
        cands = list_markets_usd() if AUTO_UNIVERSE else list(positions.keys())
        short_list = rank_universe(cands)
        entries_done = 0
        for sym in short_list:
            if entries_done >= MAX_TRADES_PER_RUN: break
            if len(positions) >= MAX_CONCURRENT_POS: break
            try:
                df_f = indicators(to_df(fetch_ohlcv(sym, TF_FAST, limit=ATR_LEN*4+60))).dropna()
                df_s = indicators(to_df(fetch_ohlcv(sym, TF_SLOW, limit=ATR_LEN*4+60))).dropna()
                if df_f.empty or df_s.empty: continue
                c = df_f["c"].iloc[-1]; atr = df_f["atr"].iloc[-1]
                if c < MIN_PRICE_USD or atr <= 0: continue
                ok_fast = (df_f["ema_s"].iloc[-1] > df_f["ema_l"].iloc[-1]) and (df_f["rsi"].iloc[-1] >= RSI_BUY)
                ok_slow = (df_s["ema_s"].iloc[-1] > df_s["ema_l"].iloc[-1])
                if not (ok_fast and ok_slow): continue
                exposure_pct = portfolio_exposure(positions)
                if exposure_pct >= PORTFOLIO_MAX_EXPOSURE_PCT: break
                qty, notional = size_by_atr(sym, c, atr, equity, step_mult=1.0)
                ok, amt_min, min_cost, _ = tradeable(sym, qty, c)
                # ---- FEES BUFFER APPLIED HERE ----
                cash_needed = notional * (1.0 + FEE_BUY_BUFFER_PCT/100.0)
                if (not ok) or (notional < MIN_NOTIONAL_USD) or ((usd_free - cash_needed) < MIN_FREE_CASH_USD):
                    continue
                od = place_order("buy", sym, qty)
                if od:
                    fee, fee_ccy = _fee_from_order(od)
                    append_tax({
                        "ts": now_ts(),"event":"TRADE","side":"buy","symbol":sym,
                        "qty":qty,"price":c,"notional":qty*c,"fee":fee,"fee_ccy":fee_ccy or BASE,
                        "order_id": od.get('id'),"note":"ENTRY"
                    })
                    set_meta(sym, entry_ts=now_ts(), entry_px=c, last_add_px=c, add_count=0)
                    positions[sym] = positions.get(sym,0.0) + qty
                    usd_free -= cash_needed
                    entries_done += 1
                    print(f"BUY {sym}: ~${notional:.2f} (qty={qty:.8f}, cash_needed=${cash_needed:.2f})")
            except Exception as e:
                print(f"[entry] {sym} error: {e}")
    else:
        print(f"[risk] drawdown breaker active; skipping new entries")

    # Final equity
    _, usd_free, equity = portfolio_and_equity()
    ts = now_ts()
    if WRITE_EQUITY_CSV: append_equity_csv(ts, equity)
    hwm, dd = equity_one_liner(equity)
    if PRINT_EQUITY_ONE_LINER:
        delta = 0.0
        try:
            if EQUITY_CSV.exists():
                df = pd.read_csv(EQUITY_CSV)
                if len(df)>=2: delta = float(df["equity"].iloc[-1]) - float(df["equity"].iloc[-2])
        except Exception: pass
        base = max(equity - delta, 1e-9)
        pct = 100.0 * (delta / base)
        print(f"[equity] now=${equity:.2f} Δrun=${delta:+.2f} ({pct:+.2f}%) HWM=${hwm:.2f} DD={dd:.2f}%")
    print("=== END TRADING OUTPUT ===")

if __name__=="__main__":
    print("[boot] TaxLedger ready -> data/tax_ledger.csv")
    run()
