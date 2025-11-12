#!/usr/bin/env python3
"""
crypto_live_safe.py — single-file, import-proof, safe baseline

Key safety rules:
- Never sell first unless TP/STOP OR a valid target quote already exists.
- If target quote is missing → HOLD current coin (no churn to cash).
- DRY_RUN defaults ON (toggle via Actions → Variables).
- Uses Kraken public Ticker for prices; Kraken private API only if live.

ENV (Variables unless noted):
  DRY_RUN=[ON|OFF]        # default ON
  BUY_USD=30              # spend per buy
  TP_PCT=8                # take profit threshold
  STOP_PCT=4              # stop loss threshold (abs)
  ROTATE_WHEN_FULL=true   # allow rotation when holding a coin
  UNIVERSE_PICK=""        # lock to a single base (e.g., LSK). Empty = top candidate

Secrets (for live):
  KRAKEN_KEY, KRAKEN_SECRET

Artifacts:
  .state/positions.json
  .state/run_summary.json
  .state/momentum_candidates.csv (if present, used to choose target)
"""

from __future__ import annotations
import base64, csv, hashlib, hmac, json, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

# ------------ Config / Env ------------
def env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None else default

DRY_RUN = env("DRY_RUN", "ON").upper()         # safest default = ON
BUY_USD = float(env("BUY_USD", "30"))
TP_PCT = float(env("TP_PCT", "8"))
STOP_PCT = float(env("STOP_PCT", "4"))
ROTATE_WHEN_FULL = env("ROTATE_WHEN_FULL", "true").lower() == "true"
UNIVERSE_PICK = env("UNIVERSE_PICK", "").strip().upper()

KRAKEN_KEY = os.environ.get("KRAKEN_KEY", "")
KRAKEN_SECRET = os.environ.get("KRAKEN_SECRET", "")

ROOT = Path(".")
STATE = ROOT/".state"
STATE.mkdir(exist_ok=True)
POS_FILE = STATE/"positions.json"
SUMMARY_FILE = STATE/"run_summary.json"
CANDIDATES_CSV = STATE/"momentum_candidates.csv"

KAPI = "https://api.kraken.com/0"
S = requests.Session(); S.headers.update({"User-Agent":"crypto-live-safe/1.0"})

def log(s: str): print(s, flush=True)

# ------------ Minimal utils ------------
def normalize_pair(s: str) -> str:
    s = (s or "").upper().replace("USDT","USD").replace(" ","")
    if "/" in s:
        base, _ = s.split("/",1); return f"{base}/USD"
    if s.endswith("USD") and len(s)>3: return f"{s[:-3]}/USD"
    return f"{s}/USD"

def read_json(path: Path, default): 
    try: return json.loads(path.read_text())
    except Exception: return default

def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)

# ------------ Public quotes (robust) ------------
def kraken_ticker(paircodes: List[str]) -> Optional[float]:
    url = f"{KAPI}/public/Ticker"
    for p in paircodes:
        try:
            r = S.get(url, params={"pair": p}, timeout=15)
            r.raise_for_status()
            res = r.json().get("result", {})
            if not res: continue
            first = next(iter(res.values()))
            return float(first["c"][0])
        except Exception: 
            continue
    return None

def quote(pair_ws: str) -> Optional[float]:
    """Return last price for BASE/USD, trying LSKUSD then LSK."""
    if not pair_ws: return None
    base = normalize_pair(pair_ws).split("/")[0]
    return kraken_ticker([f"{base}USD", base])

# ------------ Candidates ------------
def load_candidates() -> List[str]:
    if not CANDIDATES_CSV.exists(): return []
    out: List[str]=[]
    with CANDIDATES_CSV.open() as f:
        for i,row in enumerate(csv.DictReader(f)):
            sym=row.get("symbol","")
            if sym: out.append(normalize_pair(sym))
            if len(out)>=25: break
    return out

def pick_target_symbol() -> Optional[str]:
    if UNIVERSE_PICK: return normalize_pair(UNIVERSE_PICK)
    cands = load_candidates()
    return cands[0] if cands else None

# ------------ Private API (only if live) ------------
def ksign(uri_path: str, data: Dict[str,str]) -> str:
    sec = base64.b64decode(KRAKEN_SECRET)
    post = "&".join(f"{k}={v}" for k,v in data.items())
    sha = hashlib.sha256((data["nonce"] + post).encode()).digest()
    mac = hmac.new(sec, uri_path.encode()+sha, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kpost(endpoint: str, data: Dict[str,str]) -> dict:
    if DRY_RUN == "ON": raise RuntimeError("live call in DRY_RUN")
    uri_path = f"/private/{endpoint}"
    data = dict(data); data["nonce"]=str(int(time.time()*1000))
    hdr = {"API-Key":KRAKEN_KEY,"API-Sign":ksign(uri_path,data)}
    r = S.post(f"{KAPI}{uri_path}", data=data, headers=hdr, timeout=30)
    r.raise_for_status()
    j=r.json()
    if j.get("error"): raise RuntimeError(f"Kraken error: {j['error']}")
    return j.get("result",{})

def buy_market_quote(pair_ws: str, usd: float) -> str:
    base = normalize_pair(pair_ws).split("/")[0]
    if DRY_RUN=="ON": log(f"[DRY] BUY {pair_ws} ${usd:.2f}"); return "DRY_BUY"
    res = kpost("AddOrder", {"pair":f"{base}USD","type":"buy","ordertype":"market","oflags":"viqc","volume":f"{usd:.2f}"})
    txid=",".join(res.get("txid",[])) or "UNKNOWN"
    log(f"[LIVE] BUY ok {pair_ws} ${usd:.2f} (tx {txid})")
    return txid

def sell_market_base(pair_ws: str, qty: float) -> str:
    base = normalize_pair(pair_ws).split("/")[0]
    if DRY_RUN=="ON": log(f"[DRY] SELL {pair_ws} qty={qty}"); return "DRY_SELL"
    res = kpost("AddOrder", {"pair":f"{base}USD","type":"sell","ordertype":"market","volume":f"{qty:.10f}"})
    txid=",".join(res.get("txid",[])) or "UNKNOWN"
    log(f"[LIVE] SELL ok {pair_ws} qty={qty} (tx {txid})")
    return txid

# ------------ Strategy core ------------
def pct(cur: float, entry: float) -> float:
    return 0.0 if entry<=0 else (cur-entry)/entry*100.0

def read_pos() -> Optional[dict]:
    p = read_json(POS_FILE,{})
    return p or None

def write_pos(p: Optional[dict]): write_json(POS_FILE, {} if p is None else p)

def run() -> int:
    cur = read_pos()
    target = pick_target_symbol()
    cur_sym = cur["symbol"] if cur else None

    cur_px = quote(cur_sym) if cur_sym else None
    tgt_px = quote(target) if target else None

    summary = {
        "ts": int(time.time()),
        "dry_run": DRY_RUN,
        "tp_pct": TP_PCT, "stop_pct": STOP_PCT,
        "rotate_when_full": ROTATE_WHEN_FULL,
        "universe_pick": UNIVERSE_PICK,
        "current": {"symbol": cur_sym, "price": cur_px},
        "target": {"symbol": target, "price": tgt_px},
        "action": "HOLD", "note": ""
    }

    # No position → open if we have a real target quote
    if not cur:
        if target and (tgt_px or 0)>0:
            qty = BUY_USD / tgt_px
            buy_market_quote(target, BUY_USD)
            write_pos({"symbol":target,"entry":tgt_px,"qty":qty,"cost":BUY_USD,"ts":int(time.time())})
            summary.update({"action":"BUY","note":f"Opened {target} with ${BUY_USD:.2f}"})
        else:
            summary.update({"action":"HOLD","note":"No valid target quote; stay in USD."})
            write_pos(None)
        write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0

    # With a position → TP/STOP first
    entry = float(cur.get("entry",0)); qty=float(cur.get("qty",0)); cs = cur["symbol"]
    if not cur_px or cur_px<=0:
        summary.update({"action":"HOLD","note":f"Quote miss for {cs}; HOLD."})
        write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0

    change = pct(cur_px, entry)
    if change >= TP_PCT:
        sell_market_base(cs, qty); write_pos(None)
        summary.update({"action":"SELL_TP","note":f"TP {change:.2f}% → sold {cs}"})
        write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0
    if change <= -abs(STOP_PCT):
        sell_market_base(cs, qty); write_pos(None)
        summary.update({"action":"SELL_SL","note":f"STOP {change:.2f}% → sold {cs}"})
        write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0

    # Rotation (only if allowed AND target has a real quote)
    if ROTATE_WHEN_FULL and target and normalize_pair(target)!=normalize_pair(cs):
        if tgt_px and tgt_px>0:
            sell_market_base(cs, qty)
            buy_market_quote(target, BUY_USD)
            write_pos({"symbol":target,"entry":tgt_px,"qty":BUY_USD/tgt_px,"cost":BUY_USD,"ts":int(time.time())})
            summary.update({"action":"ROTATE","note":f"{cs} → {target}"})
            write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0
        else:
            summary.update({"action":"HOLD","note":f"Target {target} quote miss; HOLD {cs}."})
            write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0

    # Optional: Spike alert (log-only)
    # If target exists and has price, compare to its "o" (open) quickly for a rough %; skip API churn here.

    summary.update({"action":"HOLD","note":f"Holding {cs} ({change:.2f}% vs entry)."})
    write_json(SUMMARY_FILE, summary); log(json.dumps(summary,indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(run())
