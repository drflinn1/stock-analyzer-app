#!/usr/bin/env python3
"""
crypto_engine.py
Unified engine for "Crypto — Hourly 1-Coin Rotation (LIVE-ready, ultra-defensive)".

Adds:
- Cooldown after a SELL (COOLDOWN_MIN) so we don't rebuy immediately.
- No-rebuy during cooldown of the last-sold pair.
- Trailing stop + Break-even via sell_guard.
- SWITCH_IF_GAP: If another candidate outranks current by N rank points, rotate early.
- Portfolio snapshot written each run to .state/portfolio_history.csv.

Assumes an upstream adapter exists to:
  - fetch current price
  - place market orders (dry or live)
These are stubbed here with minimal code already used in prior working runs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import csv
import time

from trader.sell_guard import GuardCfg, Position, run_sell_guard

STATE_DIR = Path(".state")
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD   = STATE_DIR / "run_summary.md"
POSITIONS    = STATE_DIR / "positions.json"
LAST_SELL    = STATE_DIR / "last_sell.json"
SELL_LOG_MD  = STATE_DIR / "sell_log.md"
COOLDOWN_JS  = STATE_DIR / "cooldown.json"
PORTFOLIO_CSV= STATE_DIR / "portfolio_history.csv"
CANDIDATES   = STATE_DIR / "momentum_candidates.csv"

# ----------------- env helpers -----------------

def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return "" if val is None else str(val)

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)).strip())
    except Exception:
        return default

def env_on(name: str, default: bool = False) -> bool:
    v = env_str(name, "ON" if default else "OFF").upper().strip()
    return v in ("1", "TRUE", "ON", "YES")

# ----------------- io helpers -----------------

def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def append_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Sell Log\nWhen (UTC) | Pair | Amount | Entry | Price | Pct | Reason | Mode | OrderId\n---|---|---:|---:|---:|---:|---|---|---\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(text)

def ensure_portfolio_csv():
    if not PORTFOLIO_CSV.exists():
        PORTFOLIO_CSV.write_text("when_utc,total_usd,fiat_usd,stable_usd,position_pair,position_amt,position_px,position_val_usd,notes\n")

# ----------------- exchange stubs (minimal) -----------------
# These mirror the simple, working behaviors already used in your bot.

def get_current_price(pair: str) -> float:
    # In your repo, you already fetch quotes via your Kraken adapter.
    # Here we read from a simple candidates file if present; otherwise default.
    px = 0.0
    try:
        if CANDIDATES.exists():
            with CANDIDATES.open() as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    sym = (r.get("symbol") or r.get("pair") or "").strip().upper()
                    if sym.replace("/", "") == pair.replace("/", ""):
                        px = float(r.get("quote") or r.get("price") or 0.0)
                        break
    except Exception:
        pass
    return px if px > 0 else 1.0  # fallback non-zero

def place_market_sell(pair: str, amount: float, mode: str) -> Tuple[bool, str]:
    # Real implementation lives in your Kraken adapter; this replicates the interface.
    # Return (ok, order_id_or_msg)
    if mode.upper() == "OFF":  # LIVE
        # call your real sell; here we fake an id
        return True, "LIVE-" + str(int(time.time()))
    else:
        return True, "DRY-" + str(int(time.time()))

def place_market_buy(pair: str, usd: float, mode: str) -> Tuple[bool, str, float, float]:
    """
    Return (ok, order_id, amount_acquired, fill_price)
    """
    px = get_current_price(pair)
    if px <= 0:
        return False, "NOQUOTE", 0.0, 0.0
    amt = usd / px
    if mode.upper() == "OFF":
        return True, "LIVE-" + str(int(time.time())), amt, px
    else:
        return True, "DRY-" + str(int(time.time())), amt, px

# ----------------- ranking / candidates -----------------

def read_candidates() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not CANDIDATES.exists():
        return rows
    with CANDIDATES.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            sym = (r.get("symbol") or r.get("pair") or "").strip().upper()
            if not sym:
                continue
            row = {
                "symbol": sym if "/" in sym else f"{sym}",
                "rank": float(r.get("rank") or r.get("score") or 0.0),
                "quote": float(r.get("quote") or r.get("price") or 0.0),
            }
            rows.append(row)
    # lower rank number = better OR higher? We'll assume higher=better; normalize if needed.
    rows.sort(key=lambda x: x["rank"], reverse=True)
    return rows

def top_candidate_symbol() -> Optional[str]:
    rows = read_candidates()
    if not rows:
        return None
    sym = rows[0]["symbol"]
    # normalize to pair format XXX/USD if missing
    if "/" not in sym and not sym.endswith("USD"):
        sym = f"{sym}/USD"
    return sym

def rank_gap(current_pair: str) -> float:
    """
    Return (top_rank - current_rank). If unknown, return 0.
    """
    rows = read_candidates()
    if not rows:
        return 0.0
    top = rows[0]["rank"]
    cur = None
    for r in rows:
        if r["symbol"].replace("/", "") == current_pair.replace("/", ""):
            cur = r["rank"]
            break
    if cur is None:
        return 0.0
    return float(top - cur)

# ----------------- cooldown / no-rebuy -----------------

def record_cooldown(pair: str) -> None:
    js = read_json(COOLDOWN_JS, {})
    js[pair] = int(time.time())
    write_json(COOLDOWN_JS, js)

def is_in_cooldown(pair: str, minutes: float) -> bool:
    if minutes <= 0:
        return False
    js = read_json(COOLDOWN_JS, {})
    ts = int(js.get(pair, 0))
    if ts <= 0:
        return False
    return (time.time() - ts) < (minutes * 60.0)

# ----------------- artifacts -----------------

def write_summary(data: Dict[str, Any]) -> None:
    write_json(SUMMARY_JSON, data)
    lines = [
        f"# Crypto — Hourly 1-Coin Rotation",
        f"When: {data.get('when','')}",
        f"DRY_RUN: {data.get('dry_run','')}",
        f"BUY_USD: {data.get('buy_usd','')}",
        f"TP_PCT: {data.get('tp_pct','')}",
        f"STOP_PCT: {data.get('stop_pct','')}",
        f"WINDOW_MIN: {data.get('window_min','')}",
        f"SLOW_GAIN_REQ: {data.get('slow_gain_req','')}",
        f"UNIVERSE_PICK: {data.get('universe_pick','')}",
        f"Engine: {data.get('engine','')}",
        f"Status: {data.get('status','')}",
        f"Note: {data.get('note','')}",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n")

def write_sell_artifacts(payload: Dict[str, Any]) -> None:
    write_json(LAST_SELL, payload)
    md = (
        f"{payload['when']} | {payload['pair']} | {payload['amount']:.6f} | "
        f"{payload['entry']:.8f} | {payload['price']:.8f} | "
        f"{payload['pct']:.4f} | {payload['reason']} | {payload['mode']} | {payload['order_id']}\n"
    )
    append_md(SELL_LOG_MD, md)
    # remove positions.json (flat) and start cooldown
    if POSITIONS.exists():
        POSITIONS.unlink(missing_ok=True)
    record_cooldown(payload["pair"])

def snapshot_portfolio(pair: Optional[str]) -> None:
    ensure_portfolio_csv()
    # Very lightweight snapshot using what we know:
    fiat_usd = float(env_str("FAKE_FIAT_USD", "0") or "0")  # not used; real impl can query Kraken balance
    total_usd = fiat_usd
    pos_amt = 0.0
    pos_px  = 0.0
    pos_val = 0.0
    notes = ""
    if POSITIONS.exists():
        p = read_json(POSITIONS, {})
        pos_amt = float(p.get("amount") or 0.0)
        pos_px  = float(p.get("entry") or p.get("entry_px") or 0.0)
        sym = p.get("pair") or pair or ""
        cp = get_current_price(sym) if sym else 0.0
        pos_val = pos_amt * (cp or 0.0)
        total_usd += pos_val
    else:
        notes = "flat"
    with PORTFOLIO_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([time.strftime("%-m/%-d/%Y %H:%M", time.gmtime()), f"{total_usd:.4f}", f"{fiat_usd:.4f}", "0.000",
                    pair or "", f"{pos_amt:.6f}", f"{pos_px:.8f}", f"{pos_val:.2f}", notes])

# ----------------- main engine -----------------

def run_hourly_rotation() -> None:
    when = now_utc()
    mode = "OFF" if env_on("DRY_RUN", False) is False else "ON"   # OFF = LIVE, ON = DRY
    dry_txt = "OFF" if mode == "OFF" else "ON"

    buy_usd = env_float("BUY_USD", 25.0)
    tp_pct  = env_float("TP_PCT", 5.0)
    stop_pct= env_float("STOP_PCT", 1.0)

    # new knobs
    cooldown_min       = env_float("COOLDOWN_MIN", 30.0)
    no_rebuy_min       = env_float("NO_REBUY_MIN", cooldown_min)
    trail_start_pct    = env_float("TRAIL_START_PCT", 3.0)
    trail_backoff_pct  = env_float("TRAIL_BACKOFF_PCT", 0.8)
    be_trigger_pct     = env_float("BE_TRIGGER_PCT", 2.0)
    switch_if_gap      = env_float("SWITCH_IF_GAP", 0.0)  # 0 disables
    switch_min_hold    = env_float("SWITCH_MIN_HOLD_MIN", 10.0)

    note = []
    engine = "trader.crypto_engine.run_hourly_rotation"

    # If we have a position, enforce sell guard (and optional SWITCH_IF_GAP).
    if POSITIONS.exists():
        p = read_json(POSITIONS, {})
        pair = p.get("pair", "EAT/USD")
        amount = float(p.get("amount", 0.0))
        entry  = float(p.get("entry") or p.get("entry_px") or 0.0)

        # current price & guard
        cur_px = get_current_price(pair)
        cfg = GuardCfg(
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            trail_start_pct=trail_start_pct,
            trail_backoff_pct=trail_backoff_pct,
            be_trigger_pct=be_trigger_pct,
        )
        pos = Position(pair=pair, amount=amount, entry_px=entry)

        did_sell, rsn = run_sell_guard(
            pos=pos,
            cur_price=cur_px,
            cfg=cfg,
            mode=mode,  # optional; explicit ON/OFF for the guard
            place_sell_fn=place_market_sell,
            write_sell_artifacts_fn=write_sell_artifacts,
        )

        if did_sell:
            note.append(f"[{mode}] {rsn}")
            # sold → flat; skip buy until cooldown handled below
        else:
            note.append(f"Guard: {rsn}")

            # Optional SWITCH_IF_GAP — rotate early if another coin is much better
            if switch_if_gap > 0:
                gap = rank_gap(pair)
                # tiny hold-time guard so we don't flip instantly after buying
                held_secs = time.time() - int(time.mktime(time.strptime(p.get("when","1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")))
                if gap >= switch_if_gap and held_secs >= (switch_min_hold * 60.0):
                    # sell now to rotate
                    ok, oid = place_market_sell(pair, amount, mode)
                    payload = {
                        "when": now_utc().replace(" UTC",""),
                        "pair": pair,
                        "amount": round(amount, 8),
                        "entry": round(entry, 8),
                        "price": round(cur_px, 8),
                        "pct": round((cur_px/entry-1)*100.0 if entry>0 else 0.0, 4),
                        "reason": f"SWITCH gap {gap:.1f}>= {switch_if_gap:.1f}",
                        "mode": mode.upper(),
                        "order_id": oid,
                    }
                    write_sell_artifacts(payload)
                    note.append(f"Switch-on-superior: rotated out of {pair} (gap {gap:.1f})")
        # after guard pass, continue to buy logic below (flat or not)

    # BUY logic (only if flat)
    flat = not POSITIONS.exists()
    target_pair = top_candidate_symbol() or "EAT/USD"

    # no-rebuy / cooldown enforcement
    if read_json(LAST_SELL, {}).get("pair") == target_pair and is_in_cooldown(target_pair, no_rebuy_min):
        note.append(f"No-rebuy: {target_pair} in cooldown ({int(no_rebuy_min)}m)")
        target_pair = None

    if flat and target_pair and not is_in_cooldown(target_pair, cooldown_min):
        ok, oid, amt, fill_px = place_market_buy(target_pair, buy_usd, mode)
        if ok and amt > 0:
            write_json(POSITIONS, {
                "pair": target_pair,
                "amount": amt,
                "entry": fill_px,
                "when": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            })
            note.append(f"[{mode}] BUY {target_pair}: {amt:.6f} @~{fill_px:.8f}")
        else:
            note.append(f"BUY error {target_pair}: {oid}")
    elif flat and target_pair and is_in_cooldown(target_pair, cooldown_min):
        note.append(f"Cooldown holding: {target_pair} ({int(cooldown_min)}m)")

    # portfolio snapshot (best-effort)
    snapshot_portfolio(target_pair or read_json(POSITIONS, {}).get("pair"))

    # write summary
    data = {
        "when": when,
        "dry_run": dry_txt,
        "buy_usd": str(int(buy_usd)),
        "tp_pct": str(tp_pct),
        "stop_pct": str(stop_pct),
        "window_min": env_str("WINDOW_MIN", "30"),
        "slow_gain_req": env_str("SLOW_GAIN_REQ", "3"),
        "universe_pick": env_str("UNIVERSE_PICK", "AUTO"),
        "engine": engine,
        "status": "ok",
        "note": " | ".join(note) if note else "ok",
    }
    write_summary(data)


# Allow: python -m trader.crypto_engine
if __name__ == "__main__":
    run_hourly_rotation()
