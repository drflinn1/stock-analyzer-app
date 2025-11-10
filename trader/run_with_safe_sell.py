#!/usr/bin/env python3
"""
Compat runner that used to call run_sell_guard(dry_run=...).
Now forwarded to use `mode=` so older workflows won't break.
"""
from __future__ import annotations
from pathlib import Path
import os, json, time

from trader.sell_guard import GuardCfg, Position, run_sell_guard

STATE_DIR     = Path(".state")
POSITIONS     = STATE_DIR / "positions.json"
LAST_SELL     = STATE_DIR / "last_sell.json"
SELL_LOG_MD   = STATE_DIR / "sell_log.md"

def env_on(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "ON" if default else "OFF").upper().strip()
    return v in ("1","TRUE","ON","YES")

def append_md(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# Sell Log\nWhen (UTC) | Pair | Amount | Entry | Price | Pct | Reason | Mode | OrderId\n---|---|---:|---:|---:|---:|---|---|---\n")
    with p.open("a", encoding="utf-8") as f:
        f.write(text)

def write_sell_artifacts(payload):
    LAST_SELL.write_text(json.dumps(payload, indent=2))
    md = (
        f"{payload['when']} | {payload['pair']} | {payload['amount']:.6f} | "
        f"{payload['entry']:.8f} | {payload['price']:.8f} | "
        f"{payload['pct']:.4f} | {payload['reason']} | {payload['mode']} | {payload['order_id']}\n"
    )
    append_md(SELL_LOG_MD, md)
    if POSITIONS.exists():
        POSITIONS.unlink(missing_ok=True)

def place_market_sell(pair: str, amount: float, mode: str):
    if mode.upper() == "OFF":
        return True, "LIVE-" + str(int(time.time()))
    return True, "DRY-" + str(int(time.time()))

def get_current_price(_pair: str) -> float:
    # Minimal placeholder – your real engine handles quotes.
    return float(os.getenv("FAKE_QUOTE", "1.0"))

def main():
    mode = "OFF" if env_on("DRY_RUN", False) is False else "ON"
    if not POSITIONS.exists():
        print("flat; nothing to sell")
        return

    p = json.loads(POSITIONS.read_text())
    pos = Position(pair=p["pair"], amount=float(p["amount"]), entry_px=float(p.get("entry") or p.get("entry_px") or 0.0))
    cur_px = get_current_price(pos.pair)

    cfg = GuardCfg(
        stop_pct=float(os.getenv("STOP_PCT", "1")),
        tp_pct=float(os.getenv("TP_PCT", "5")),
        trail_start_pct=float(os.getenv("TRAIL_START_PCT", "3")),
        trail_backoff_pct=float(os.getenv("TRAIL_BACKOFF_PCT", "0.8")),
        be_trigger_pct=float(os.getenv("BE_TRIGGER_PCT", "2")),
    )

    did_sell, rsn = run_sell_guard(
        pos=pos,
        cur_price=cur_px,
        cfg=cfg,
        mode=mode,  # <<<<<<<<<<<<<< uses mode now
        place_sell_fn=place_market_sell,
        write_sell_artifacts_fn=write_sell_artifacts,
    )
    print("sell_guard:", did_sell, rsn)

if __name__ == "__main__":
    main()
