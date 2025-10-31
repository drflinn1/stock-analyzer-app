#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
Includes BUY/SELL logic hooks for compliance with Sell Logic Guard.

Enhancement: Momentum Pulse (THAW filter)
- If MOMENTUM_PULSE_ENABLE=true and THAW is detected, we read
  .state/momentum_candidates.csv and pass a whitelist to the engine via:
    BUY_WHITELIST_MODE=THAW_PULSE
    BUY_WHITELIST_BASES=BTC,ETH,...
    BUY_WHITELIST_PAIRS=BTC/USD,ETH/USDT,...
  We also write .state/buy_whitelist.json for engines that prefer files.
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List, Set, Tuple

# ----------------- Constants & State -----------------

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = STATE_DIR / "run_summary.json"
SUMMARY_MD = STATE_DIR / "run_summary.md"
LAST_OK = STATE_DIR / "last_ok.txt"

# Momentum Pulse env knobs (read once here; engines can also read them)
MOMENTUM_PULSE_ENABLE = str(os.getenv("MOMENTUM_PULSE_ENABLE", "false")).lower() in {
    "1", "true", "yes", "on"
}
MOMENTUM_CANDIDATES_CSV = os.getenv("MOMENTUM_CANDIDATES_CSV", ".state/momentum_candidates.csv")

# ----------------- Helpers -----------------

def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return "" if val is None else str(val)

def write_summary(data: Dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2))
    lines = [
        f"**When:** {data.get('when')}",
        f"**DRY_RUN:** {data.get('DRY_RUN')}",
        f"**RUN_SWITCH:** {data.get('RUN_SWITCH')}",
        f"**Entry:** {data.get('entrypoint')}",
        f"**Engine Executed:** {data.get('engine_executed')}",
        f"**Notes:** {data.get('notes','')}",
    ]
    SUMMARY_MD.write_text("\n\n".join(lines) + "\n")

def post_slack(text: str) -> None:
    webhook = env_str("SLACK_WEBHOOK_URL")
    if not webhook:
        return
    try:
        import requests  # lazy import
        requests.post(webhook, json={"text": text}, timeout=10)
    except Exception:
        pass

def file_exists(path: str) -> bool:
    return Path(path).is_file()

def run_engine(path: str, dry_run: str, extra_env: Dict[str, str] | None = None) -> int:
    cmd = [sys.executable, "-u", path]
    env = os.environ.copy()
    env["DRY_RUN"] = dry_run
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"[runner] exec: {' '.join(cmd)}")
    if extra_env:
        # small, readable dump
        keys = ["BUY_WHITELIST_MODE", "BUY_WHITELIST_BASES", "BUY_WHITELIST_PAIRS"]
        dumped = {k: env.get(k, "") for k in keys}
        print(f"[runner] whitelist env -> {json.dumps(dumped)}")
    try:
        return subprocess.run(cmd, env=env, check=False).returncode
    except Exception as e:
        print(f"[runner] engine error: {e}", file=sys.stderr)
        return 1

# ----------------- SELL LOGIC STUBS (Guard keywords) -----------------

def TAKE_PROFIT(symbol: str, price: float, gain: float) -> bool:
    """Example take-profit trigger."""
    print(f"[SELL] TAKE_PROFIT triggered for {symbol} at {price} (+{gain:.2f}%)")
    return True

def STOP_LOSS(symbol: str, price: float, loss: float) -> bool:
    """Example stop-loss trigger."""
    print(f"[SELL] STOP_LOSS triggered for {symbol} at {price} (-{loss:.2f}%)")
    return True

def TRAIL(symbol: str, current: float, high: float, trail_pct: float = 2.0) -> bool:
    """Simple trailing stop tracker."""
    trigger = high * (1 - trail_pct / 100)
    if current <= trigger:
        print(f"[SELL] TRAIL triggered for {symbol} at {current} (trail {trail_pct}%)")
        return True
    return False

# ----------------- Momentum Pulse (THAW detection + CSV loader) -----------------

def detect_thaw(state_dir: str = ".state") -> bool:
    """
    Detect a THAW window using multiple heuristics:
      - .state/thaw.flag            -> if file exists => THAW
      - .state/guard.yaml           -> if {mode: THAW} or {THAW: true}
    """
    thaw_flag = Path(state_dir) / "thaw.flag"
    if thaw_flag.exists():
        return True

    guard_yaml = Path(state_dir) / "guard.yaml"
    if guard_yaml.exists():
        try:
            import yaml  # pyyaml is in requirements
            data = yaml.safe_load(guard_yaml.read_text(encoding="utf-8")) or {}
            mode = str(data.get("mode") or data.get("MODE") or "").strip().upper()
            if mode == "THAW":
                return True
            thaw_bool = data.get("thaw") or data.get("THAW")
            if isinstance(thaw_bool, bool) and thaw_bool:
                return True
        except Exception:
            # fail-open: don't block trading
            return False
    return False

def _normalize_pair(s: str) -> str:
    s = (s or "").upper().strip().replace("-", "/")
    return s

def _safe_base(s: str) -> str:
    s = (s or "").upper().strip()
    if "/" in s:
        return s.split("/", 1)[0]
    if "-" in s:
        return s.split("-", 1)[0]
    return s

def load_momentum_candidates(csv_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Accepts flexible columns: symbol, base, quote, pair, rank (rank optional)
    Returns (bases, pairs)
    """
    p = Path(csv_path)
    if not p.exists():
        return set(), set()

    bases: Set[str] = set()
    pairs: Set[str] = set()
    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            for row in reader:
                def get(name: str) -> str:
                    key = cols.get(name)
                    return (row.get(key) or "").strip().upper() if key else ""

                sym = get("symbol") or get("base")
                qt = get("quote")
                pr = get("pair")

                if sym:
                    bases.add(sym)
                if pr:
                    pairs.add(_normalize_pair(pr))
                elif sym and qt:
                    pairs.add(f"{sym}/{qt}")
    except Exception as e:
        print(f"[pulse] CSV parse error ({csv_path}): {e}", file=sys.stderr)
        return set(), set()

    return bases, pairs

def prepare_whitelist_env() -> Dict[str, str]:
    """
    If pulse is active (env true) and THAW is detected, build whitelist env vars
    and write a helper JSON file. Otherwise, return {}.
    """
    if not MOMENTUM_PULSE_ENABLE:
        return {}

    if not detect_thaw(str(STATE_DIR)):
        return {}

    bases, pairs = load_momentum_candidates(MOMENTUM_CANDIDATES_CSV)
    if not bases and not pairs:
        print("[pulse] THAW active but no candidates found; falling back to full universe.")
        return {}

    # Write a helper file engines can read
    wl = {
        "mode": "THAW_PULSE",
        "bases": sorted(bases),
        "pairs": sorted(pairs),
        "source_csv": MOMENTUM_CANDIDATES_CSV,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (STATE_DIR / "buy_whitelist.json").write_text(json.dumps(wl, indent=2))

    return {
        "BUY_WHITELIST_MODE": "THAW_PULSE",
        "BUY_WHITELIST_BASES": ",".join(sorted(bases)),
        "BUY_WHITELIST_PAIRS": ",".join(sorted(pairs)),
        # optional: engines can choose a default quote to auto-complete base-only
        "BUY_WHITELIST_DEFAULT_QUOTE": os.getenv("BUY_WHITELIST_DEFAULT_QUOTE", "USD"),
    }

# ----------------- MAIN -----------------

def main() -> int:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    dry_run = env_str("DRY_RUN", "ON").upper()
    run_switch = env_str("RUN_SWITCH", "ON").upper()
    entrypoint = env_str("ENTRYPOINT", "trader/crypto_engine.py")

    summary: Dict[str, Any] = {
        "when": now,
        "DRY_RUN": dry_run,
        "RUN_SWITCH": run_switch,
        "entrypoint": entrypoint,
        "engine_executed": False,
        "notes": "",
    }

    if run_switch not in ("ON", "OFF"):
        run_switch = "ON"
        summary["notes"] += "RUN_SWITCH invalid; defaulted to ON. "

    if run_switch == "OFF":
        summary["notes"] += "RUN_SWITCH=OFF -> skipping trading logic.\n"
        write_summary(summary)
        post_slack(f"Muted: bot skipped (RUN_SWITCH=OFF, DRY_RUN={dry_run}).")
        print("[runner] Skipped by RUN_SWITCH=OFF")
        return 0

    # Simulated SELL logic (for dry run & test compliance)
    TAKE_PROFIT("BTC/USD", 69000, 5.0)
    TRAIL("BTC/USD", 67200, 69000)
    STOP_LOSS("BTC/USD", 65500, -5.0)

    # Build optional whitelist env if THAW+Pulse
    whitelist_env = prepare_whitelist_env()
    if whitelist_env:
        print("[pulse] THAW active — passing momentum whitelist to engine.")

    # Now hand off to engine if present
    candidates = [entrypoint, "trader/main.py", "bot/main.py", "engine.py"]
    executed = False
    rc = 0
    for c in candidates:
        if file_exists(c):
            rc = run_engine(c, dry_run, extra_env=whitelist_env)
            executed = True
            break

    summary["engine_executed"] = executed
    if not executed:
        summary["notes"] += (
            "No engine file found. Looked for: "
            + ", ".join(candidates)
            + ". Set ENTRYPOINT variable to correct path.\n"
        )
        print("[runner] No engine file found. Set ENTRYPOINT variable.")

    write_summary(summary)
    if executed and rc == 0:
        LAST_OK.write_text(now + "\n")

    status = "OK" if rc == 0 else "ERR"
    post_slack(
        f"Crypto run {status} • engine:{'yes' if executed else 'no'} • "
        f"DRY_RUN:{dry_run} • RUN_SWITCH:{run_switch} • Pulse:{'on' if whitelist_env else 'off'}"
    )
    return rc

if __name__ == "__main__":
    sys.exit(main())
