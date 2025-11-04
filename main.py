#!/usr/bin/env python3
"""
Main unified runner — Hourly 1-Coin Rotation (Auto-Universe, LIVE-ready)

- Auto discovers Kraken USD spot pairs (filters by 24h USD volume)
- Ranks by 60-minute return; trades top 1
- Exits: STOP_-1%  |  TP_+5%  |  FAIL_<+3%@60m (after 60m)
- After any SELL: immediate re-rank & rotate (risk-aware)
- DRY_RUN safe by default; when DRY_RUN=OFF it will place real Kraken market orders

Secrets required for live:
  KRAKEN_API_KEY, KRAKEN_API_SECRET
"""

from __future__ import annotations
import csv, json, math, os, re, sys, ssl, hmac, hashlib, base64, time
import urllib.parse, urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- Env helpers ----------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except: return default

def env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except: return default

# ---------- Config ----------
STATE_DIR = Path(env_str("STATE_DIR", ".state")); STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
POSITION_JSON= STATE_DIR / "position.json"
RUN_LOG_CSV  = STATE_DIR / "hourly_rotation_runlog.csv"
CANDS_CSV    = STATE_DIR / "momentum_candidates.csv"  # optional

DRY_RUN  = env_str("DRY_RUN", "ON").upper() != "OFF"
QUOTE    = env_str("QUOTE", "USD").upper()
UNIVERSE = [s.strip().upper() for s in env_str("UNIVERSE", "").split(",") if s.strip()]

AUTO_UNIVERSE          = env_int("AUTO_UNIVERSE", 1) == 1
AUTO_UNIVERSE_TOP_K    = env_int("AUTO_UNIVERSE_TOP_K", 30)
AUTO_MIN_BASE_VOL_USD  = env_float("AUTO_MIN_BASE_VOL_USD", 25000.0)
AUTO_EXCLUDE           = {s.strip().upper() for s in env_str("AUTO_EXCLUDE", "USDT,USDC,EUR,DAI,FDUSD").split(",") if s.strip()}

MIN_BUY_USD = env_float("MIN_BUY_USD", 25.0)
FEE_BPS     = env_int("FEE_BPS", 10)

HOLD_WINDOW_MIN = 60
STOP_PCT, TP_PCT, MIN_1H_PCT = -0.01, 0.05, 0.03

RISK_ON = env_int("RISK_ON", 1) == 1
RISK_THRESH_BTC_60M = env_float("RISK_THRESH_BTC_60M", -0.005)

# ---------- Types ----------
@dataclass
class Position:
    symbol: str
    entry_ts: str
    entry_px: float
    size_usd: float
    quote: str = QUOTE
    @property
    def dt(self): return datetime.fromisoformat(self.entry_ts)

# ---------- Kraken Public ----------
_SSL = ssl.create_default_context()

def http_json(url: str, data: bytes = None, headers: Dict[str,str] = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def kr_pair(sym: str, quote: str = QUOTE) -> str:
    s = sym.upper()
    if s == "BTC": s = "XBT"
    return f"{s}{quote.upper()}"

def last_price(sym: str, quote: str = QUOTE) -> Optional[float]:
    url = f"https://api.kraken.com/0/public/Ticker?pair={urllib.parse.quote(kr_pair(sym, quote))}"
    try:
        d = http_json(url); 
        if d.get("error"): return None
        r = next(iter(d["result"].values()))
        return float(r["c"][0])
    except: return None

def ret_60m(sym: str, quote: str = QUOTE, interval: int = 1) -> Optional[float]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={urllib.parse.quote(kr_pair(sym, quote))}&interval={interval}"
    try:
        d = http_json(url); 
        if d.get("error"): return None
        rows = next(iter(d["result"].values()))
        if len(rows) < 65: return None
        c_now, c_60 = float(rows[-1][4]), float(rows[-61][4])
        if c_60 <= 0: return None
        return (c_now/c_60)-1.0
    except: return None

def discover_usd_bases() -> List[str]:
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        d = http_json(url); 
        if d.get("error"): return []
        out=[]
        for k,_ in d["result"].items():
            if not k.endswith("USD"): continue
            base = k[:-3]
            if base=="XBT": base="BTC"
            if base.upper() in AUTO_EXCLUDE: continue
            out.append(base.upper())
        return sorted(set(out))
    except: return []

def bulk_ticker(symbols: List[str], quote: str = QUOTE) -> Dict[str, Dict[str,float]]:
    res={}
    if not symbols: return res
    url = "https://api.kraken.com/0/public/Ticker?pair=" + urllib.parse.quote(",".join(kr_pair(s,quote) for s in symbols))
    try:
        d = http_json(url); 
        if d.get("error"): return res
        for k,v in d["result"].items():
            base = k[:-3];  base = "BTC" if base=="XBT" else base
            last=float(v["c"][0]); base_vol=float(v["v"][1]); usd_vol=last*base_vol
            res[base.upper()]={"last":last,"base_vol":base_vol,"usd_vol":usd_vol}
        return res
    except: return res

# ---------- Universe & ranking ----------
def load_candidates_csv(path: Path)->List[str]:
    if not path.exists(): return []
    rows=[]
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                s=row.get("symbol","").strip().upper()
                rk=row.get("rank","")
                try: rk=float(rk)
                except: rk=math.inf
                if s: rows.append((rk,s))
        rows.sort(key=lambda x:x[0])
        return [s for _,s in rows]
    except: return []

def build_universe() -> List[str]:
    if UNIVERSE: return list(dict.fromkeys(UNIVERSE))
    if AUTO_UNIVERSE:
        bases = discover_usd_bases()
        t = bulk_ticker(bases)
        filt=[s for s in bases if t.get(s,{}).get("usd_vol",0.0)>=AUTO_MIN_BASE_VOL_USD]
        return sorted(filt)[:max(1,AUTO_UNIVERSE_TOP_K)]
    csv_syms=load_candidates_csv(CANDS_CSV)
    return csv_syms[:max(1,AUTO_UNIVERSE_TOP_K)] if csv_syms else []

def rank_top(cands: List[str]) -> Optional[str]:
    best=None; br=-9e9
    for s in cands:
        r=ret_60m(s)
        if r is None: continue
        if r>br: br=r; best=s
    if best: return best
    csv_syms=load_candidates_csv(CANDS_CSV)
    for s in csv_syms:
        if s in cands: return s
    return cands[0] if cands else None

# ---------- Risk gate ----------
def risk_ok() -> bool:
    if not RISK_ON: return True
    r = ret_60m("BTC")
    if r is None: return True
    return r > RISK_THRESH_BTC_60M

# ---------- Kraken Private (LIVE) ----------
API_KEY    = os.getenv("KRAKEN_API_KEY","")
API_SECRET = os.getenv("KRAKEN_API_SECRET","")

def _kr_sign(uri_path: str, data: Dict[str,str]) -> str:
    postdata = urllib.parse.urlencode(data).encode()
    message = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    sha256 = hashlib.sha256(message).digest()
    mac = hmac.new(base64.b64decode(API_SECRET), uri_path.encode() + sha256, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest()).decode()
    return sigdigest, postdata

def kr_private(endpoint: str, data: Dict[str,str]) -> dict:
    url = f"https://api.kraken.com/0/private/{endpoint}"
    uri_path = f"/0/private/{endpoint}"
    sig, post = _kr_sign(uri_path, data)
    headers = {
        "API-Key": API_KEY,
        "API-Sign": sig,
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    return http_json(url, data=post, headers=headers)

def add_order_market(symbol: str, side: str, usd_amount: float, price_hint: Optional[float]) -> dict:
    # Kraken wants volume in BASE units; convert using price_hint
    if price_hint is None or price_hint <= 0:
        price_hint = last_price(symbol) or 0
    if price_hint <= 0:
        return {"error":["EAPI:price_unavailable"]}

    base_qty = max(0.0, usd_amount / price_hint)
    # round base amount to 6 dp (many pairs allow 6+)
    base_qty = float(f"{base_qty:.6f}")
    pair = kr_pair(symbol, QUOTE)
    data = {
        "nonce": str(int(time.time() * 1000)),
        "ordertype": "market",
        "type": side,           # buy or sell
        "pair": pair,
        "volume": str(base_qty),
        "oflags": "viqc",       # volume in quote currency calc (helps with fees); Kraken accepts on market orders
    }
    try:
        resp = kr_private("AddOrder", data)
        return resp
    except Exception as e:
        return {"error":[f"EXC:{e}"]}

# ---------- Broker ----------
class Broker:
    def __init__(self, dry_run: bool, fee_bps: int):
        self.dry_run = dry_run
        self.fee_bps = fee_bps

    def place_market(self, side: str, symbol: str, usd_amount: float, price: Optional[float]=None) -> Dict:
        if self.dry_run:
            print(f"[DRY] {side.upper()} {symbol}/{QUOTE} ~ ${usd_amount:.2f}")
            return {"status":"ok","dry_run":True}
        if not API_KEY or not API_SECRET:
            print("[LIVE] Missing Kraken API secrets; aborting order.")
            return {"status":"error","error":"missing_secrets"}

        resp = add_order_market(symbol, side, usd_amount, price)
        if resp.get("error"):
            print(f"[LIVE] {side.upper()} {symbol}: ERROR {resp['error']}")
            return {"status":"error","resp":resp}
        print(f"[LIVE] {side.upper()} {symbol}: OK {resp.get('result')}")
        return {"status":"ok","resp":resp}

# ---------- State & logging ----------
def read_pos() -> Optional[Position]:
    if not POSITION_JSON.exists(): return None
    try: return Position(**json.loads(POSITION_JSON.read_text()))
    except: return None

def write_pos(p: Optional[Position]) -> None:
    if p is None:
        if POSITION_JSON.exists(): POSITION_JSON.unlink()
        return
    POSITION_JSON.write_text(json.dumps(asdict(p), indent=2))

def now_utc(): return datetime.now(timezone.utc)
def pct(a: float, b: float) -> float: return 0.0 if a<=0 else (b/a)-1.0

def log_line(ts, event, sym, price, note, pos):
    new = not RUN_LOG_CSV.exists()
    with RUN_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        if new: w.writerow(["ts","event","symbol","price","pos_symbol","entry_px","unrealized_pct","note"])
        chg=""
        if pos and price: chg=f"{pct(pos.entry_px, price):.4%}"
        w.writerow([ts.isoformat(), event, sym, f"{price:.6f}", pos.symbol if pos else "", f"{pos.entry_px if pos else 0:.6f}", chg, note])

def write_summary(d: Dict):
    SUMMARY_JSON.write_text(json.dumps(d, indent=2))
    SUMMARY_MD.write_text("\n".join([
        f"**When:** {d.get('when')}",
        f"**DRY_RUN:** {'ON' if DRY_RUN else 'OFF'}",
        f"**Action:** {d.get('action')}",
        f"**Symbol:** {d.get('symbol','')}",
        f"**Note:** {d.get('note','')}",
    ]))

# ---------- Core ----------
def build_candidates() -> List[str]:
    if UNIVERSE: return list(dict.fromkeys(UNIVERSE))
    if AUTO_UNIVERSE:
        bases = discover_usd_bases()
        t = bulk_ticker(bases)
        filt = [s for s in bases if t.get(s,{}).get("usd_vol",0.0) >= AUTO_MIN_BASE_VOL_USD]
        return sorted(filt)[:max(1, AUTO_UNIVERSE_TOP_K)]
    csv_syms = load_candidates_csv(CANDS_CSV)
    return csv_syms[:max(1,AUTO_UNIVERSE_TOP_K)] if csv_syms else []

def main() -> int:
    ts = now_utc()
    br = Broker(DRY_RUN, FEE_BPS)
    pos = read_pos()

    # Flat → enter
    if pos is None:
        if not risk_ok():
            note="Risk-OFF → hold cash"
            write_summary({"when":ts.isoformat(),"action":"HOLD_CASH","note":note})
            log_line(ts,"HOLD_CASH","",0.0,note,None); print(note); return 0

        cands = build_candidates() or ["BTC","ETH","SOL"]
        top = rank_top(cands)
        if not top:
            note="No candidate found"
            write_summary({"when":ts.isoformat(),"action":"NO_CANDIDATE","note":note})
            log_line(ts,"NO_CANDIDATE","",0.0,note,None); print(note); return 0

        px = last_price(top)
        if px is None:
            note=f"No price for {top}/{QUOTE}"
            write_summary({"when":ts.isoformat(),"action":"NO_PRICE","symbol":top,"note":note})
            log_line(ts,"NO_PRICE",top,0.0,note,None); print(note); return 0

        fee = MIN_BUY_USD*(FEE_BPS/10_000.0); usd_net = max(0.0, MIN_BUY_USD - fee)
        br.place_market("buy", top, MIN_BUY_USD, price=px)
        new = Position(symbol=top, entry_ts=ts.isoformat(), entry_px=px, size_usd=usd_net)
        write_pos(new); log_line(ts,"BUY",top,px,f"size=${MIN_BUY_USD:.2f}, fee=${fee:.2f}",new)
        write_summary({"when":ts.isoformat(),"action":"BUY","symbol":top,"note":f"size ${MIN_BUY_USD:.2f}"})
        print(f"BUY {top}/{QUOTE} ~{px:.6f} | ticket ${MIN_BUY_USD:.2f}")
        return 0

    # Holding → exits
    px = last_price(pos.symbol)
    if px is None:
        note=f"No price for {pos.symbol}/{QUOTE}; hold"
        write_summary({"when":ts.isoformat(),"action":"HOLD_NO_PRICE","symbol":pos.symbol,"note":note})
        log_line(ts,"HOLD",pos.symbol,0.0,note,pos); print(note); return 0

    chg = pct(pos.entry_px, px)
    held_min = int((ts - pos.dt).total_seconds()//60)
    reason=None
    if chg <= STOP_PCT: reason="STOP_-1%"
    elif chg >= TP_PCT: reason="TP_+5%"
    elif held_min >= HOLD_WINDOW_MIN and chg < MIN_1H_PCT: reason="FAIL_<+3%@60m"

    if reason:
        proceeds = pos.size_usd*(1.0+chg); fee = proceeds*(FEE_BPS/10_000.0)
        br.place_market("sell", pos.symbol, proceeds, price=px)
        log_line(ts,"SELL",pos.symbol,px,f"{reason}, chg={chg:.2%}, fee=${fee:.2f}",pos)
        write_pos(None)

        if risk_ok():
            cands = build_candidates() or ["BTC","ETH","SOL"]
            nxt = rank_top(cands)
            if nxt:
                px2 = last_price(nxt)
                if px2 is not None:
                    ticket = MIN_BUY_USD; fee_b=ticket*(FEE_BPS/10_000.0)
                    br.place_market("buy", nxt, ticket, price=px2)
                    new=Position(symbol=nxt, entry_ts=ts.isoformat(), entry_px=px2, size_usd=max(0.0,ticket-fee_b))
                    write_pos(new); log_line(ts,"BUY",nxt,px2,f"rotate; size=${ticket:.2f}, fee=${fee_b:.2f}",new)
                    write_summary({"when":ts.isoformat(),"action":"ROTATE","symbol":f"{pos.symbol}→{nxt}","note":reason})
                    print(f"ROTATE {pos.symbol}→{nxt} @ ~{px2:.6f}"); return 0

        write_summary({"when":ts.isoformat(),"action":"SELL_TO_CASH","symbol":pos.symbol,"note":reason})
        print(f"SELL→CASH {pos.symbol} | {reason}"); return 0

    # Heartbeat
    log_line(ts,"HOLD",pos.symbol,px,f"chg={chg:.2%}, held={held_min}m",pos)
    write_summary({"when":ts.isoformat(),"action":"HOLD","symbol":pos.symbol,"note":f"chg={chg:.2%}, held={held_min}m"})
    print(f"HOLD {pos.symbol} | chg={chg:.2%} held={held_min}m")
    return 0

if __name__=="__main__":
    sys.exit(main())
