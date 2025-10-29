# tools/auto_sell_guard.py
# ---------------------------------------------------------------------
# Auto-Sell Cool-Off Guard (SAFE MODE)
# Prevents rapid consecutive sells during volatility spikes.
# Writes a small artifact so the workflow always has something to upload.
# ---------------------------------------------------------------------

import os, time, json, pathlib

COOL_OFF_MINUTES = int(os.getenv("COOL_OFF_MINUTES", "15"))
STATE_DIR = pathlib.Path(".state")
LAST_SELL_TS = STATE_DIR / "last_sell_ts.txt"

def _now_ts() -> int:
    return int(time.time())

def _read_last_sell_ts() -> int | None:
    try:
        if LAST_SELL_TS.exists():
            return int(LAST_SELL_TS.read_text().strip())
    except Exception:
        pass
    return None

def _write_last_sell_ts(ts: int) -> None:
    try:
        STATE_DIR.mkdir(exist_ok=True)
        LAST_SELL_TS.write_text(str(ts), encoding="utf-8")
    except Exception:
        pass

def run_cool_off_guard():
    STATE_DIR.mkdir(exist_ok=True)

    last_ts = _read_last_sell_ts()
    now = _now_ts()
    cool_ok = True
    reason = "no prior sells recorded"

    if last_ts is not None:
        elapsed_min = (now - last_ts) / 60.0
        if elapsed_min < COOL_OFF_MINUTES:
            cool_ok = False
            reason = f"cool-off active: {elapsed_min:.1f}m < {COOL_OFF_MINUTES}m window"
        else:
            reason = f"cool-off satisfied: {elapsed_min:.1f}m >= {COOL_OFF_MINUTES}m"

    # SAFE MODE: we don't place or block real orders here; just report status
    info = {
        "ts": now,
        "cool_off_minutes": COOL_OFF_MINUTES,
        "ok_to_sell": cool_ok,
        "reason": reason,
    }

    out_json = STATE_DIR / "auto_sell_guard.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"SUMMARY: Auto-Sell Cool-Off Guard — ok_to_sell={cool_ok} ({reason})")
    print(f"ARTIFACT: wrote {out_json}")

    # For future: when a sell actually happens, call _write_last_sell_ts(_now_ts())
    return info

if __name__ == "__main__":
    run_cool_off_guard()
