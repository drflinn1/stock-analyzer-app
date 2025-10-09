# main.py — Crypto Live (Kraken, USD) with Auto-Universe & Cleanup
# - Auto universe from Kraken markets (volume/spread/filters)
# - Optional cleanup of non-universe holdings
# - Rotation, TP/SL/Trail, daily caps, Slack pings, artifacts

from __future__ import annotations
import os, sys, json, time, csv, math, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

import requests

# ---------- Basics ----------
def utcnow() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def as_bool(x: Optional[str], d: bool) -> bool:
    if x is None or x == "": return d
    return str(x).strip().lower() in {"1","true","yes","y","on"}

def as_int(x: Optional[str], d: int) -> int:
    try: return int(x) if x is not None else d
    except: return d

def as_float(x: Optional[str], d: float) -> float:
    try: return float(x) if x is not None else d
    except: return d

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def append_csv(path: str, row: List[Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts_utc","event","symbol","side","qty","price","notional","pnl","note"])
        w.writerow(row)

# ---------- Slack ----------
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL","").strip()

def slack_post(payload: Dict[str, Any]) -> None:
    if not SLACK_WEBHOOK: return
    try:
        r = requests.post(SLACK_WEBHOOK, data=json.dumps(payload),
                          headers={"Content-Type":"application/json"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] Slack post failed: {e}")

def slack_info(text: str) -> None:
    slack_post({"text": text})

def slack_trade(symbol: str, side: str, qty: float, price: float, color: str) -> None:
    slack_post({
        "text": f"{side.upper()} {symbol}",
        "attachments": [{
            "color": color,
            "fields": [
                {"title":"Symbol","value":symbol,"short":True},
                {"title":"Side","value":side.upper(),"short":True},
                {"title":"Qty","value":f"{qty:.6f}","short":True},
                {"title":"Price","value":f"${price:.2f}","short":True},
                {"title":"Notional","value":f"${qty*price:.2f}","short":True},
            ],
        }],
    })

def slack_risk_flip(state: str, reason: str) -> None:
    color = "#2eb886" if state.upper()=="ON" else "#e01e5a"
    slack_post({"text": f"⚠️ Risk signal → *{state.upper()}* — {reason}",
                "attachments":[{"color":color}]})

# ---------- ENV ----------
EXCHANGE  = os.getenv("EXCHANGE","kraken")
QUOTE     = os.getenv("QUOTE","USD").upper()

AUTO_UNIVERSE          = as_bool(os.getenv("AUTO_UNIVERSE"), True)
UNIVERSE_STATIC        = [s.strip().upper() for s in os.getenv("UNIVERSE_STATIC","BTC,ETH,SOL,DOGE,ADA,ZEC").split(",") if s.strip()]
UNIVERSE_TOP_K         = as_int(os.getenv("UNIVERSE_TOP_K"), 6)
UNIVERSE_EXCLUDE       = set([s.strip().upper() for s in os.getenv("UNIVERSE_EXCLUDE","USDT,USDC,EUR,GBP,USD,SPX,PUMP,BABY").split(",") if s.strip()])
UNIVERSE_MIN_USD_VOL   = as_float(os.getenv("UNIVERSE_MIN_USD_VOL"), 500000.0)
MAX_SPREAD_PCT         = as_float(os.getenv("MAX_SPREAD_PCT"), 0.60)
MIN_TRADE_NOTIONAL_USD = as_float(os.getenv("MIN_TRADE_NOTIONAL_USD"), 5.0)

MAX_POSITIONS    = as_int(os.getenv("MAX_POSITIONS"), 6)
USD_PER_TRADE    = as_float(os.getenv("USD_PER_TRADE"), 20.0)
RESERVE_CASH_PCT = as_float(os.getenv("RESERVE_CASH_PCT"), 10.0)

ROTATE_WHEN_CASH_SHORT = as_bool(os.getenv("ROTATE_WHEN_CASH_SHORT"), True)
ROTATE_WHEN_FULL       = as_bool(os.getenv("ROTATE_WHEN_FULL"), False)
DUST_MIN_USD     = as_float(os.getenv("DUST_MIN_USD"), 2.0)

TAKE_PROFIT_PCT  = as_float(os.getenv("TAKE_PROFIT_PCT"), 3.0)
STOP_LOSS_PCT    = as_float(os.getenv("STOP_LOSS_PCT"), 2.0)
TRAIL_STOP_PCT   = as_float(os.getenv("TRAIL_STOP_PCT"), 1.0)
MAX_DAILY_LOSS_PCT = as_float(os.getenv("MAX_DAILY_LOSS_PCT"), 5.0)
MAX_DAILY_ENTRIES  = as_int(os.getenv("MAX_DAILY_ENTRIES"), 6)
EMERGENCY_SL_PCT   = as_float(os.getenv("EMERGENCY_SL_PCT"), 8.0)

CLEANUP_NON_UNIVERSE        = as_bool(os.getenv("CLEANUP_NON_UNIVERSE"), True)
NONUNI_SELL_IF_DOWN_PCT     = as_float(os.getenv("NONUNI_SELL_IF_DOWN_PCT"), 0.0)
NONUNI_KEEP_IF_WINNER_PCT   = as_float(os.getenv("NONUNI_KEEP_IF_WINNER_PCT"), 6.0)

RUN_SWITCH = os.getenv("RUN_SWITCH","ON").upper()
DRY_RUN    = os.getenv("DRY_RUN","ON").upper()

STATE_DIR        = os.getenv("STATE_DIR",".state")
POSITIONS_JSON   = os.getenv("POSITIONS_JSON", f"{STATE_DIR}/positions.json")
KPI_CSV          = os.getenv("KPI_CSV", f"{STATE_DIR}/kpi_history.csv")
SPEC_GATE_REPORT = os.getenv("SPEC_GATE_REPORT", f"{STATE_DIR}/spec_gate_report.txt")

API_KEY    = os.getenv("KRAKEN_API_KEY","")
API_SECRET = os.getenv("KRAKEN_API_SECRET","")

# ---------- Exchange ----------
def make_exchange() -> ccxt.Exchange:
    if EXCHANGE.lower() != "kraken":
        raise SystemExit("Only kraken supported in this file.")
    cfg = {"enableRateLimit": True, "options":{"adjustForTimeDifference": True}}
    if DRY_RUN != "ON":
        cfg.update({"apiKey": API_KEY, "secret": API_SECRET})
    ex = ccxt.kraken(cfg)
    ex.load_markets()
    return ex

def pair(base: str) -> str:
    return f"{base}/{QUOTE}"

def pick_usd_key(bal: Dict[str,float]) -> str:
    for k in ("USD","ZUSD","XUSD"):
        if k in bal: return k
    return QUOTE if QUOTE in bal else "USD"

# ---------- State ----------
def default_state() -> Dict[str, Any]:
    return {"positions":{}, "last_risk":"ON", "today":datetime.utcnow().strftime("%Y-%m-%d"),
            "daily_pnl":0.0, "daily_entries":0}

def load_state() -> Dict[str, Any]:
    s = load_json(POSITIONS_JSON, default_state())
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if s.get("today") != today:
        s["today"] = today
        s["daily_pnl"] = 0.0
        s["daily_entries"] = 0
    if "positions" not in s: s["positions"] = {}
    return s

def save_state(s: Dict[str, Any]) -> None:
    ensure_dir(STATE_DIR); save_json(POSITIONS_JSON, s)

# ---------- Reporting ----------
def append_spec(msg: str) -> None:
    ensure_dir(STATE_DIR)
    with open(SPEC_GATE_REPORT, "a", encoding="utf-8") as f:
        f.write(f"[{utcnow()}] {msg}\n")

# ---------- Auto-Universe ----------
def market_info_for_base(ex: ccxt.Exchange, base: str) -> Optional[Dict[str, Any]]:
    sym = pair(base)
    if sym not in ex.markets: return None
    m = ex.markets[sym]
    if not m.get("active", True): return None
    if m.get("type") not in (None, "spot"): return None
    # min notional check
    min_cost = None
    limits = m.get("limits") or {}
    if limits.get("cost") and limits["cost"].get("min"):
        min_cost = float(limits["cost"]["min"])
    elif limits.get("amount") and limits["amount"].get("min"):
        # estimate with last price
        try:
            t = ex.fetch_ticker(sym)
            px = float(t.get("last") or t.get("close") or 0.0)
            min_cost = float(limits["amount"]["min"]) * px if px>0 else None
        except Exception:
            pass
    if min_cost is not None and min_cost > MIN_TRADE_NOTIONAL_USD:
        return None
    try:
        t = ex.fetch_ticker(sym)
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        last = float(t.get("last") or t.get("close") or 0.0)
        if bid<=0 or ask<=0 or last<=0: return None
        spread = (ask - bid) / last * 100.0
        if spread > MAX_SPREAD_PCT: return None
        # volume proxy
        qv = t.get("quoteVolume")
        bv = t.get("baseVolume")
        usd_vol = float(qv) if qv else (float(bv or 0.0) * last)
        return {"base":base, "usd_vol":usd_vol, "spread":spread, "pct":float(t.get("percentage") or 0.0)}
    except Exception:
        return None

def build_auto_universe(ex: ccxt.Exchange) -> List[str]:
    bases = []
    for m in ex.markets.values():
        try:
            if m.get("quote") != QUOTE: continue
            b = str(m.get("base","")).upper()
            if not b or b in UNIVERSE_EXCLUDE: continue
            info = market_info_for_base(ex, b)
            if not info: continue
            if info["usd_vol"] < UNIVERSE_MIN_USD_VOL: continue
            bases.append(info)
        except Exception:
            pass
    bases.sort(key=lambda d: d["usd_vol"], reverse=True)
    pick = [d["base"] for d in bases[:UNIVERSE_TOP_K]]
    return pick if pick else UNIVERSE_STATIC[:UNIVERSE_TOP_K]

# ---------- Risk (simple breadth) ----------
def compute_risk_signal(ex: ccxt.Exchange, bases: List[str]) -> Tuple[str, str]:
    changes = []
    for b in bases:
        try:
            pct = float(ex.fetch_ticker(pair(b)).get("percentage") or 0.0)
            changes.append(pct)
        except Exception:
            pass
    if not changes: return "ON", "No data"
    avg = sum(changes)/len(changes)
    return ("OFF", f"Avg 24h% {avg:.2f} < -2%") if avg<-2.0 else ("ON", f"Avg 24h% {avg:.2f} ≥ -2%")

# ---------- Caps ----------
def violated_daily_caps(state: Dict[str, Any], equity_usd: float) -> Optional[str]:
    if MAX_DAILY_LOSS_PCT>0 and equity_usd>0:
        if (state.get("daily_pnl",0.0)/equity_usd)*100 <= -MAX_DAILY_LOSS_PCT:
            return "Daily loss cap hit"
    if MAX_DAILY_ENTRIES>0 and state.get("daily_entries",0)>=MAX_DAILY_ENTRIES:
        return "Max daily entries reached"
    return None

# ---------- Balances ----------
def fetch_cash_and_equity(ex: ccxt.Exchange, watch_bases: List[str]) -> Tuple[float,float,Dict[str,float]]:
    bal = ex.fetch_free_balance()
    usd_key = pick_usd_key(bal)
    cash = float(bal.get(usd_key, 0.0))
    equity = cash
    for b in watch_bases:
        q = float(bal.get(b, 0.0))
        if q>0:
            try:
                px = float(ex.fetch_ticker(pair(b)).get("last") or 0.0)
                equity += q*px
            except Exception:
                pass
    return cash, equity, bal

# ---------- Orders ----------
def place_order(ex: ccxt.Exchange, base: str, side: str, usd_notional: float) -> Tuple[float,float]:
    sym = pair(base)
    t = ex.fetch_ticker(sym)
    px = float(t.get("last") or t.get("close") or t.get("ask") or t.get("bid") or 0.0)
    if px<=0: raise RuntimeError(f"Bad price for {sym}")
    qty = usd_notional/px
    if DRY_RUN=="ON":
        print(f"[DRY] {side.upper()} {sym} qty={qty:.8f} px={px:.4f} notional=${usd_notional:.2f}")
        return qty, px
    typ="market"
    if side.lower()=="buy":
        o = ex.create_order(sym, typ, "buy", qty, None, {"cost": usd_notional})
    else:
        o = ex.create_order(sym, typ, "sell", qty, None, {})
    filled = float(o.get("filled") or qty)
    avg    = float(o.get("average") or px)
    return filled, avg

# ---------- Trailing ----------
def trailing_update(trail: float, price: float, trail_pct: float) -> float:
    new_trail = price * (1.0 - trail_pct/100.0)
    return max(trail, new_trail) if trail>0 else new_trail

# ---------- Main ----------
def run() -> None:
    print(f"=== Crypto Live — {EXCHANGE.upper()} — {utcnow()} ===")
    print(f"Mode: {DRY_RUN} | RUN_SWITCH: {RUN_SWITCH}")
    ensure_dir(STATE_DIR)
    with open(SPEC_GATE_REPORT,"w",encoding="utf-8") as f:
        f.write(f"Run started {utcnow()}\n")

    if RUN_SWITCH!="ON":
        append_spec("RUN_SWITCH OFF — exit"); return

    ex = make_exchange()

    # Build universe
    bases = build_auto_universe(ex) if AUTO_UNIVERSE else UNIVERSE_STATIC[:UNIVERSE_TOP_K]
    append_spec(f"Universe={bases} (AUTO={AUTO_UNIVERSE})")

    # Risk
    risk, reason = compute_risk_signal(ex, bases); append_spec(f"Risk={risk} ({reason})")

    s = load_state()
    if risk != s.get("last_risk","ON"):
        slack_risk_flip(risk, reason)
        s["last_risk"] = risk
        save_state(s)

    # Balances
    cash, equity, bal = fetch_cash_and_equity(ex, bases)
    append_spec(f"Cash={cash:.2f} Equity~={equity:.2f}")

    # Daily caps
    cap = violated_daily_caps(s, equity)
    if cap:
        append_spec(cap); print(cap); save_state(s); return

    # Current holdings map (all bases we have, not just universe)
    holdings: Dict[str,float] = {k:float(v) for k,v in bal.items() if k.isalpha() and k!=pick_usd_key(bal) and float(v)>0}

    # 1) Cleanup non-universe holdings if requested
    if CLEANUP_NON_UNIVERSE and holdings:
        for b,q in list(holdings.items()):
            if b in bases: continue  # managed by strategy
            try:
                t = ex.fetch_ticker(pair(b))
                px = float(t.get("last") or 0.0)
                pct = float(t.get("percentage") or 0.0)
                notional = q*px
            except Exception:
                px=0.0; pct=0.0; notional=0.0
            sell = False
            reason = ""
            if notional < DUST_MIN_USD:
                sell = True; reason = "non-universe dust"
            elif pct < NONUNI_SELL_IF_DOWN_PCT:
                sell = True; reason = f"non-universe red {pct:.2f}%"
            elif pct >= NONUNI_KEEP_IF_WINNER_PCT:
                sell = False; reason = f"winner {pct:.2f}% — keep"
            if sell and q>0 and px>0:
                qty, price = place_order(ex, b, "sell", q*px)
                pnl = (price - float(s.get("positions",{}).get(b,{}).get("entry_price",price))) * qty
                append_csv(KPI_CSV, [utcnow(),"SELL", b,"SELL", f"{qty:.8f}", f"{price:.6f}", f"{qty*price:.2f}", f"{pnl:.2f}", reason])
                slack_trade(b, "sell", qty, price, "#e01e5a")
                holdings.pop(b, None)
                s["positions"].pop(b, None)
                append_spec(f"Cleanup sold {b}: {reason}")
            else:
                append_spec(f"Cleanup decision {b}: {reason or 'hold'}")

    # Refresh holdings for universe only
    bal = ex.fetch_free_balance()
    holdings = {b: float(bal.get(b,0.0)) for b in bases if float(bal.get(b,0.0))>0}

    # SELL logic for managed holdings
    for b,q in list(holdings.items()):
        try:
            px = float(ex.fetch_ticker(pair(b)).get("last") or 0.0)
        except Exception:
            px = 0.0
        pos = s["positions"].get(b, {"qty":q, "entry_price":px, "trail":0.0})
        if pos["qty"]<=0: pos["qty"]=q
        if pos["entry_price"]<=0 and px>0: pos["entry_price"]=px
        pos["trail"] = trailing_update(float(pos.get("trail",0.0)), px, TRAIL_STOP_PCT)
        s["positions"][b] = pos

        entry = float(pos["entry_price"])
        pnl_pct = (px/entry-1.0)*100.0 if entry>0 and px>0 else 0.0

        sell=False; why=""
        if EMERGENCY_SL_PCT>0 and pnl_pct <= -EMERGENCY_SL_PCT:
            sell=True; why=f"emergency SL {pnl_pct:.2f}%"
        if not sell and TAKE_PROFIT_PCT>0 and pnl_pct >= TAKE_PROFIT_PCT:
            sell=True; why=f"TP {pnl_pct:.2f}%"
        if not sell and STOP_LOSS_PCT>0 and pnl_pct <= -STOP_LOSS_PCT:
            sell=True; why=f"SL {pnl_pct:.2f}%"
        if not sell and pos["trail"]>0 and px>0 and px <= pos["trail"]:
            sell=True; why=f"trail hit (<= {pos['trail']:.6f})"
        if not sell and q*px < DUST_MIN_USD:
            sell=True; why="dust"

        if sell and q>0 and px>0:
            qty, price = place_order(ex, b, "sell", q*px)
            realized = (price - entry) * qty
            s["daily_pnl"] = float(s.get("daily_pnl",0.0)) + realized
            s["positions"].pop(b, None)
            append_csv(KPI_CSV, [utcnow(),"SELL", b,"SELL", f"{qty:.8f}", f"{price:.6f}", f"{qty*price:.2f}", f"{realized:.2f}", why])
            slack_trade(b, "sell", qty, price, "#e01e5a")
            holdings.pop(b, None)

    # Balances after sells
    bal = ex.fetch_free_balance()
    usd_key = pick_usd_key(bal)
    cash = float(bal.get(usd_key, 0.0))
    cash_after_reserve = max(0.0, cash - cash*(RESERVE_CASH_PCT/100.0))

    # BUY planning: rank universe by 24h momentum
    ranks: List[Tuple[str,float]] = []
    for b in bases:
        try:
            pct = float(ex.fetch_ticker(pair(b)).get("percentage") or 0.0)
            ranks.append((b,pct))
        except Exception:
            pass
    ranks.sort(key=lambda t: t[1], reverse=True)
    desired = [b for b,_ in ranks[:MAX_POSITIONS]]

    # Buy missing
    for b in desired:
        if b in holdings: continue
        if cash_after_reserve < USD_PER_TRADE:
            # rotate worst to fund if allowed
            if ROTATE_WHEN_CASH_SHORT and holdings:
                rank_map = {bb:i for i,(bb,_) in enumerate(ranks)}
                worst = max(holdings.keys(), key=lambda bb: rank_map.get(bb,9999))
                q = float(bal.get(worst,0.0))
                try:
                    pxw = float(ex.fetch_ticker(pair(worst)).get("last") or 0.0)
                except Exception:
                    pxw = 0.0
                if q>0 and pxw>0:
                    qty, price = place_order(ex, worst, "sell", q*pxw)
                    append_csv(KPI_CSV,[utcnow(),"SELL", worst,"SELL", f"{qty:.8f}", f"{price:.6f}", f"{qty*price:.2f}","", "rotate fund"])
                    slack_trade(worst,"sell",qty,price,"#e01e5a")
                    # refresh
                    bal = ex.fetch_free_balance()
                    cash = float(bal.get(usd_key,0.0))
                    cash_after_reserve = max(0.0, cash - cash*(RESERVE_CASH_PCT/100.0))
                else:
                    append_spec("Rotate: failed to price worst")
            if cash_after_reserve < USD_PER_TRADE:
                append_spec(f"No cash for buy {b}"); continue

        try:
            qty, price = place_order(ex, b, "buy", USD_PER_TRADE)
            s["positions"][b] = {"qty":qty, "entry_price":price, "trail":0.0}
            s["daily_entries"] = int(s.get("daily_entries",0)) + 1
            append_csv(KPI_CSV,[utcnow(),"BUY", b,"BUY", f"{qty:.8f}", f"{price:.6f}", f"{qty*price:.2f}","", ""])
            slack_trade(b,"buy",qty,price,"#2eb886")
            cash_after_reserve -= USD_PER_TRADE
        except Exception as e:
            append_spec(f"Buy error {b}: {e}")

    save_state(s)
    append_spec("Run complete.")
    print("Run complete.")

if __name__ == "__main__":
    try:
        print("🚧 DRY RUN — NO REAL ORDERS SENT 🚧" if DRY_RUN=="ON" else "✅ LIVE MODE — REAL ORDERS ENABLED")
        run()
    except Exception as e:
        err = "".join(traceback.format_exception(*sys.exc_info()))
        ensure_dir(STATE_DIR)
        with open(SPEC_GATE_REPORT,"a",encoding="utf-8") as f:
            f.write("\n[ERROR]\n"); f.write(err)
        print(f"[FATAL] {e}")
        sys.exit(1)
