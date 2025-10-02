from __future__ import annotations
import os, math, time, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit(f"ccxt is required: {e}")

STATE_DIR = Path(".state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
ROTATE_CD_PATH = STATE_DIR / "rotate_cooldown.json"

# ---- ENV ----
DRY_RUN    = os.getenv("DRY_RUN", "true").lower() == "true"
RUN_SWITCH = os.getenv("RUN_SWITCH", "on").lower().strip()
EXCHANGE   = os.getenv("EXCHANGE", "kraken").lower()

KRAKEN_API_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

USD_PER_TRADE = float(os.getenv("USD_PER_TRADE", "35"))
RESERVE_USD   = float(os.getenv("RESERVE_USD", "80"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "6"))
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "6"))

GUARANTEE_MIN_NEW = int(os.getenv("GUARANTEE_MIN_NEW", "1"))
GUARANTEE_PICK = [s.strip().upper() for s in os.getenv("GUARANTEE_PICK", "").split(",") if s.strip()]

QUOTE_ALLOW   = [q.strip().upper() for q in os.getenv("QUOTE_ALLOW", "USD,USDT").split(",") if q.strip()]
CORE_WHITELIST = [s.strip().upper() for s in os.getenv("CORE_WHITELIST", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",") if s.strip()]
SPEC_SYMBOLS   = [s.strip().upper() for s in os.getenv("SPEC_SYMBOLS", "").split(",") if s.strip()]
PAIR_BLOCKLIST = set([s.strip().upper() for s in os.getenv("PAIR_BLOCKLIST", "").split(",") if s.strip()])

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_NOTIONAL_USD = float(os.getenv("MIN_NOTIONAL_USD", "18"))  # safety floor

# ---- Rotation knobs ----
ROTATE_ENABLED = os.getenv("ROTATE_ENABLED", "true").lower() == "true"
ROTATE_MIN_EDGE_PCT = float(os.getenv("ROTATE_MIN_EDGE_PCT", "2.0"))
ROTATE_MAX_SWITCHES_PER_RUN = int(os.getenv("ROTATE_MAX_SWITCHES_PER_RUN", "1"))
ROTATE_COOLDOWN_HOURS = float(os.getenv("ROTATE_COOLDOWN_HOURS", "6"))

def load_json(path: Path, default: Any) -> Any:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default
def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def now_utc() -> str: return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

def new_exchange():
    if EXCHANGE != "kraken":
        raise SystemExit(f"Only kraken supported here (got {EXCHANGE})")
    return ccxt.kraken({
        "apiKey": KRAKEN_API_KEY, "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True, "options": {"adjustForTimeDifference": True},
    })

def usd_balance(balances: Dict[str, Any]) -> float:
    total = balances.get("total") or {}; return float(total.get("USD", 0.0))

def list_open_bases(balances: Dict[str, Any]) -> Dict[str, float]:
    total = balances.get("total") or {}; out={}
    for asset, amt in total.items():
        if not isinstance(amt, (int,float)): continue
        a=float(amt)
        if asset in ("USD","USDT","EUR","GBP") or a<=0: continue
        out[asset]=a
    return out

def allowed_symbol(markets: Dict[str, Any], base: str) -> Optional[str]:
    for q in QUOTE_ALLOW:
        sym=f"{base}/{q}"
        m=markets.get(sym)
        if m and (m.get("active",True)) and sym.upper() not in PAIR_BLOCKLIST:
            return sym
    return None

def fetch_change_pct(exch, symbol: str) -> Optional[float]:
    try:
        t = exch.fetch_ticker(symbol)
        pc = t.get("percentage")
        if pc is None:
            last = t.get("last") or t.get("close"); openp=t.get("open")
            if last is None or openp in (None,0): return None
            pc = (float(last)-float(openp))/float(openp)*100.0
        return float(pc)
    except Exception:
        return None

def amount_precision(market: Dict[str, Any]) -> int:
    return int((market.get("precision",{}) or {}).get("amount",8) or 8)
def price_precision(market: Dict[str, Any]) -> int:
    return int((market.get("precision",{}) or {}).get("price",8) or 8)
def min_amount(market: Dict[str, Any]) -> float:
    return float(((market.get("limits",{}) or {}).get("amount",{}) or {}).get("min",0.0) or 0.0)
def min_cost(market: Dict[str, Any]) -> float:
    return float(((market.get("limits",{}) or {}).get("cost",{}) or {}).get("min",0.0) or 0.0)
def round_to(x: float, prec: int) -> float:
    f=10**prec; return math.floor(x*f)/f

def build_universe(exch, markets: Dict[str, Any]) -> List[str]:
    syms=[]
    for s in CORE_WHITELIST:
        if s and s in markets and s.upper() not in PAIR_BLOCKLIST: syms.append(s)
    for s in SPEC_SYMBOLS:
        if s and s in markets and s.upper() not in PAIR_BLOCKLIST and s not in syms: syms.append(s)

    movers=[]; seen=set(syms); bases_seen=set(sym.split("/")[0] for sym in seen)
    for sym,m in markets.items():
        try: base,quote=sym.split("/")
        except: continue
        if quote.upper() not in QUOTE_ALLOW: continue
        if sym.upper() in PAIR_BLOCKLIST: continue
        if base in ("USD","USDT","EUR","GBP"): continue
        if sym in seen or base in bases_seen: continue
        pct=fetch_change_pct(exch,sym)
        if pct is None: continue
        movers.append((sym,float(pct)))
    movers.sort(key=lambda x:x[1], reverse=True)
    for sym,_ in movers[:max(0,TOP_K-len(syms))]: syms.append(sym)

    # Try guarantee picks first
    prior=[]
    for p in [s for s in os.getenv("GUARANTEE_PICK","").split(",") if s.strip()]:
        p=p.strip().upper()
        if p in syms and p not in prior: prior.append(p)
    for s in syms:
        if s not in prior: prior.append(s)
    return prior[:TOP_K]

def ensure_trade_allowed(bal: float) -> Tuple[float,float]:
    return max(0.0, bal-RESERVE_USD), USD_PER_TRADE

# --- Order helpers (Kraken) ---
def kraken_market_buy(exch, symbol: str, usd_size: float, market: Dict[str,Any]) -> Tuple[bool,str]:
    t = exch.fetch_ticker(symbol)
    last = float(t.get("last") or t.get("close") or 0.0)
    if last <= 0: return False, "no price"
    amt_prec = amount_precision(market)
    prc_prec = price_precision(market)
    min_amt  = min_amount(market)
    cost_min = min_cost(market)

    target = max(usd_size, cost_min, MIN_NOTIONAL_USD)
    amt = round_to(target / last, amt_prec)
    if min_amt and amt < min_amt: amt = min_amt
    notional = amt * last

    if DRY_RUN:
        return True, f"DRY_RUN BUY {symbol} amt={amt} last={round(last,prc_prec)} notional≈${notional:.2f} (min_amt={min_amt}, min_cost={cost_min}, target=${target:.2f})"
    try:
        o = exch.create_order(symbol, type="market", side="buy", amount=None, params={
            "cost": round(target,2),
            "quoteOrderQty": round(target,2),
            "oflags": "viqc",
        })
        oid = o.get("id") or "?"
        return True, f"BUY ok {symbol} cost=${target:.2f} (amt≈{amt}) order_id={oid}"
    except Exception as e1:
        try:
            o = exch.create_order(symbol, type="market", side="buy", amount=amt)
            oid = o.get("id") or "?"
            return True, f"BUY ok {symbol} amt={amt} notional≈${notional:.2f} order_id={oid}"
        except Exception as e2:
            return False, f"BUY error {symbol}: cost_try={e1}; amount_try={e2}; min_amt={min_amt} min_cost={cost_min} last={round(last,prc_prec)} target=${target:.2f} notional≈${notional:.2f}"

def kraken_market_sell(exch, symbol: str, amount: float, market: Dict[str,Any]) -> Tuple[bool,str]:
    amt_prec = amount_precision(market)
    amt = round_to(float(amount), amt_prec)
    if DRY_RUN:
        return True, f"DRY_RUN SELL {symbol} amount={amt}"
    try:
        o = exch.create_order(symbol, type="market", side="sell", amount=amt)
        oid = o.get("id") or "?"
        return True, f"SELL ok {symbol} amount={amt} order_id={oid}"
    except Exception as e:
        return False, f"SELL error {symbol}: {e}"

def create_market_buy(exch, symbol: str, usd_size: float) -> Tuple[bool,str]:
    m = exch.markets[symbol]; return kraken_market_buy(exch, symbol, usd_size, m)
def create_market_sell(exch, symbol: str, amount: float) -> Tuple[bool,str]:
    m = exch.markets[symbol]; return kraken_market_sell(exch, symbol, amount, m)

# ---- Rotation core ----
def try_rotation(exch, markets, open_bases: Dict[str,float], universe_syms: List[str]) -> Tuple[int, List[str]]:
    """Sell worst holding and buy best candidate if edge >= ROTATE_MIN_EDGE_PCT."""
    if not ROTATE_ENABLED or not open_bases: return 0, []
    cooldowns = load_json(ROTATE_CD_PATH, {})
    now = time.time(); cd_secs = ROTATE_COOLDOWN_HOURS * 3600.0

    # Map holdings to tradable symbols and momentum
    holds: List[Tuple[str, str, float, float]] = []  # (base, sym, pct, amount)
    for base, amt in open_bases.items():
        sym = allowed_symbol(markets, base)
        if not sym: continue
        pct = fetch_change_pct(exch, sym)
        if pct is None: continue
        holds.append((base, sym, pct, float(amt)))

    # Candidates = universe not already held (by base)
    held_bases = {b for b,_,_,_ in holds}
    cands: List[Tuple[str, float]] = []  # (sym, pct)
    for sym in universe_syms:
        base = sym.split("/")[0]
        if base in held_bases: continue
        pct = fetch_change_pct(exch, sym)
        if pct is None: continue
        cands.append((sym, pct))

    if not holds or not cands: return 0, []

    # Pick best candidate and worst holding
    best_sym, best_pct = max(cands, key=lambda x: x[1])
    worst = min(holds, key=lambda x: x[2])
    worst_base, worst_sym, worst_pct, worst_amt = worst
    edge = best_pct - worst_pct

    msgs = [f"ROTATE scan: best={best_sym} {best_pct:.2f}% vs worst={worst_sym} {worst_pct:.2f}% → edge={edge:.2f}%"]

    if edge < ROTATE_MIN_EDGE_PCT:
        msgs.append(f"ROTATE skip — edge {edge:.2f}% < min {ROTATE_MIN_EDGE_PCT:.2f}%")
        return 0, msgs

    # Cooldown on the sold symbol
    cd_until = float(cooldowns.get(worst_sym, 0))
    if now < cd_until:
        rem_h = (cd_until - now)/3600.0
        msgs.append(f"ROTATE skip — {worst_sym} on cooldown {rem_h:.1f}h")
        return 0, msgs

    # Execute switch (sell worst, buy best)
    ok_s, info_s = create_market_sell(exch, worst_sym, worst_amt)
    msgs.append(info_s)
    if not ok_s:
        msgs.append("ROTATE abort — sell failed")
        return 0, msgs

    ok_b, info_b = create_market_buy(exch, best_sym, USD_PER_TRADE)
    msgs.append(info_b)
    if ok_b:
        cooldowns[worst_sym] = now + cd_secs
        save_json(ROTATE_CD_PATH, cooldowns)
        msgs.append(f"ROTATE done — started cooldown on {worst_sym} for {ROTATE_COOLDOWN_HOURS:.0f}h")
        return 1, msgs
    else:
        msgs.append("ROTATE error — buy failed (sell already completed)")
        return 0, msgs

def main() -> None:
    print("============================================================")
    print("🟢 LIVE TRADING")
    print("============================================================")
    print(f"{now_utc()} INFO: Starting trader in CRYPTO mode. Dry run={DRY_RUN}. Broker=ccxt")

    if RUN_SWITCH != "on":
        print(f"{now_utc()} INFO: RUN_SWITCH={RUN_SWITCH} → exiting early."); return

    exch = new_exchange(); markets = exch.load_markets()
    balances = exch.fetch_balance()
    usd = usd_balance(balances)
    print(f"{now_utc()} INFO: USD balance detected: ${usd:.2f}")

    open_bases = list_open_bases(balances); open_count = len(open_bases)
    symbols = build_universe(exch, markets)
    print(f"{now_utc()} INFO: Universe (auto): top {TOP_K} → {symbols}")

    cap_left = max(0, MAX_POSITIONS - open_count)
    avail, per_trade = ensure_trade_allowed(usd)

    buys = 0; reasons: List[str] = []
    if cap_left <= 0: reasons.append("cap_left=0")
    if avail < per_trade: reasons.append(f"avail ${avail:.2f} < per_trade ${per_trade:.2f}")

    # --- Normal buys ---
    if not reasons:
        for sym in symbols:
            if cap_left <= 0 or buys >= DAILY_MAX_TRADES: break
            if sym.upper() in PAIR_BLOCKLIST: continue
            ok, info = create_market_buy(exch, sym, per_trade)
            if ok:
                print(f"{now_utc()} INFO: {info}")
                buys += 1; cap_left -= 1; avail -= per_trade
            else:
                if len(reasons) < 6: reasons.append(f"{sym}: {info}")

    # --- Guarantee at least N buys ---
    if buys < GUARANTEE_MIN_NEW and cap_left > 0:
        need = GUARANTEE_MIN_NEW - buys; forced = 0
        for sym in symbols:
            if forced >= need: break
            if avail < per_trade:
                reasons.append("guarantee: insufficient avail"); break
            ok, info = create_market_buy(exch, sym, per_trade)
            if ok:
                print(f"{now_utc()} INFO: GUARANTEE BUY → {info}")
                forced += 1; buys += 1; cap_left -= 1; avail -= per_trade
            else:
                if len(reasons) < 10: reasons.append(f"guarantee {sym}: {info}")

    # --- Rotation (sell worst, buy best) when fully allocated ---
    switches = 0
    if cap_left == 0 and ROTATE_ENABLED and switches < ROTATE_MAX_SWITCHES_PER_RUN:
        did, msgs = try_rotation(exch, markets, open_bases, symbols)
        for m in msgs:
            print(f"{now_utc()} INFO: {m}")
        switches += did

    if buys == 0 and switches == 0:
        msg = f"No entry (cap_left={cap_left}, per_trade=${per_trade:.2f}, avail=${avail:.2f})"
        if reasons: msg += " — " + "; ".join(reasons)
        print(f"{now_utc()} INFO: {msg}")

    print(f"{now_utc()} INFO: KPI SUMMARY: entries={buys} switches={switches} open={open_count} cap_left={cap_left} usd=${usd:.2f}")
    print(f"{now_utc()} INFO: DONE.")

if __name__ == "__main__":
    main()
