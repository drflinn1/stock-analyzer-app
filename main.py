#!/usr/bin/env python3
"""
Main unified runner for Crypto — Hourly 1-Coin Rotation (SELL → BUY → SNAPSHOT).
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

STATE = Path(".state"); STATE.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE / "run_summary.json"
SUMMARY_MD   = STATE / "run_summary.md"
LAST_OK      = STATE / "last_ok.txt"
POS_FILE     = STATE / "positions.json"

def env_str(n, d=""): v=os.getenv(n); return d if v is None else str(v)
def now_iso()->str: return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def write_summary(d: Dict[str, Any])->None:
    SUMMARY_JSON.write_text(json.dumps(d, indent=2))
    lines=["# Crypto — Hourly 1-Coin Rotation",
           f"**When:** {d.get('when')}",
           f"**DRY_RUN:** {d.get('dry_run')}",
           f"**BUY_USD:** {d.get('buy_usd')}",
           f"**TP_PCT:** {d.get('tp_pct')}%",
           f"**STOP_PCT:** {d.get('stop_pct')}%",
           f"**WINDOW_MIN:** {d.get('window_min')} min",
           f"**SLOW_GAIN_REQ:** {d.get('slow_gain_req')}%",
           f"**UNIVERSE_PICK:** {d.get('universe_pick') or '<auto>'}","",
           f"**Engine:** {d.get('engine')}",
           f"**Status:** {d.get('status')}",
           f"**Note:** {d.get('note') or '-'}"]
    SUMMARY_MD.write_text("\n".join(lines))

def keys_present()->bool:
    return bool((os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_KEY")) and
                (os.getenv("KRAKEN_API_SECRET") or os.getenv("KRAKEN_SECRET")))

def main()->int:
    dry_run      = env_str("DRY_RUN","ON").upper()
    buy_usd      = env_str("BUY_USD","25")
    tp_pct       = env_str("TP_PCT","5")
    stop_pct     = env_str("STOP_PCT","1")
    window_min   = env_str("WINDOW_MIN","30")
    slow_gain_req= env_str("SLOW_GAIN_REQ","3")
    universe_pick= env_str("UNIVERSE_PICK","")

    status="ok"; note=""; engine="noop"
    live_ok = keys_present()
    if dry_run=="OFF" and not live_ok:
        status="skipped"; note="LIVE requested but API keys missing — aborting orders."

    sell_note=""
    if status=="ok":
        try:
            from trader.sell_guard import run_sell_guard
            sell_res = run_sell_guard(
                dry_run=(dry_run!="OFF"),
                tp_pct=float(tp_pct), stop_pct=float(stop_pct),
                window_min=int(window_min), slow_gain_req=float(slow_gain_req),
            )
            sell_note = sell_res.get("note","")
        except ModuleNotFoundError:
            pass
        except Exception as e:
            status="error"; note=f"SellGuard {type(e).__name__}: {e}"

    if status=="ok" and not POS_FILE.exists():
        try:
            from trader.crypto_engine import run_hourly_rotation
            engine="trader.crypto_engine.run_hourly_rotation"
            run_hourly_rotation(
                dry_run=(dry_run!="OFF"),
                buy_usd=float(buy_usd),
                tp_pct=float(tp_pct), stop_pct=float(stop_pct),
                window_min=int(window_min), slow_gain_req=float(slow_gain_req),
                universe_pick=(universe_pick or None),
            )
        except ModuleNotFoundError:
            engine="noop"; time.sleep(0.5); note="Local engine not found; ran no-op. (Add trader/ to enable trading.)"
        except Exception as e:
            status="error"; note=f"{type(e).__name__}: {e}"
    elif status=="ok" and POS_FILE.exists():
        engine="sell_guard_only"; note=f"Holding; BUY skipped. {sell_note}"

    # Always snapshot portfolio at end (non-fatal if it fails)
    try:
        from trader.portfolio_snapshot import snapshot
        snapshot()
    except Exception:
        pass

    summary={"when":now_iso(),"dry_run":dry_run,"buy_usd":buy_usd,"tp_pct":tp_pct,"stop_pct":stop_pct,
             "window_min":window_min,"slow_gain_req":slow_gain_req,"universe_pick":(universe_pick or None),
             "engine":engine,"status":status,"note":(note or sell_note)}
    write_summary(summary)
    if status=="error" or (dry_run=="OFF" and not live_ok): return 1
    LAST_OK.write_text(now_iso()); return 0

if __name__=="__main__":
    sys.exit(main())
