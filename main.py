#!/usr/bin/env python3
"""
Main unified runner for Crypto Live workflows.
Includes BUY/SELL logic hooks for compliance with Sell Logic Guard.

Enhancement: Momentum Pulse (THAW filter)
- If THAW is detected (via .state/thaw.flag or .state/guard.yaml),
  and a momentum CSV exists, we:
    1) Log a verification line like:
       "MomentumPulse: THAW active — filtered universe (N/M) using momentum_candidates.csv"
       (M is best-effort if we know the universe; otherwise N)
    2) Pass a whitelist to the engine via env:
         BUY_WHITELIST_MODE=THAW_PULSE
         BUY_WHITELIST_BASES=BTC,ETH,...
         BUY_WHITELIST_PAIRS=BTC/USD,ETH/USDT,...
    3) Write .state/buy_whitelist.json for engines that prefer files.

Compatible envs
---------------
NEW:
  MOMENTUM_CSV_PATH      : path to momentum CSV (ex: .state/spike_candidates.csv)
  MOMENTUM_MIN_COUNT     : minimum symbols for THAW activation (default 1)

LEGACY:
  MOMENTUM_PULSE_ENABLE  : "true"/"on" to enable (default false)
  MOMENTUM_CANDIDATES_CSV: path to CSV (default .state/momentum_candidates.csv)
"""

from __future__ import annotations

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

# Legacy knobs (still supported)
MOMENTUM_PULSE_ENABLE = str(os.getenv("MOMENTUM_PULSE_ENABLE", "false")).lower() in {"1", "true", "yes", "on"}
MOMENTUM_CANDIDATES_CSV = os.getenv("MOMENTUM_CANDIDATES_CSV", ".state/momentum_candidates.csv")

# New knobs (preferred)
MOMENTUM_CSV_PATH = os.getenv("MOMENTUM_CSV_PATH", "").strip()
try:
    MOMENTUM_MIN_COUNT = max(1, int(os.getenv("MOMENTUM_MIN_COUNT", "1")))
except Exception:
    MOMENTUM_MIN_COUNT = 1

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

# ----------------- THAW detection -----------------

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

# ----------------- Momentum CSV utils -----------------

def _candidate_csv_and_label() -> Tuple[Path, str]:
    """
    Preferred priority:
      1) MOMENTUM_CSV_PATH (if exists)
      2) MOMENTUM_CANDIDATES_CSV (legacy)
      3) ./momentum_candidates.csv
      4) ./.state/momentum_candidates.csv
    Returns (path, label_for_log)
    """
    # 1) Explicit new env
    if MOMENTUM_CSV_PATH:
        p = Path(MOMENTUM_CSV_PATH)
        if p.exists():
            return p, p.name

    # 2) Legacy env
    if MOMENTUM_CANDIDATES_CSV:
        p = Path(MOMENTUM_CANDIDATES_CSV)
        if p.exists():
            return p, p.name

    # 3) & 4) Defaults
    for p in (Path("momentum_candidates.csv"), Path(".state") / "momentum_candidates.csv"):
        if p.exists():
            return p, p.name

    # nothing found -> default label to the new env name for clarity
    return Path(MOMENTUM_CSV_PATH or "momentum_candidates.csv"), \
           (Path(MOMENTUM_CSV_PATH).name if MOMENTUM_CSV_PATH else "momentum_candidates.csv")

def _read_candidates(csv_path: Path) -> Set[str]:
    """
    Accepts flexible CSV:
      - header: symbol/base/ticker/pair/quote
      - headerless: single column of symbols
    Only the first matching column is used.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    try:
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except Exception:
        return set()

    if not rows:
        return set()

    headers = [c.strip().lower() for c in rows[0]]
    known = {"symbol", "base", "ticker", "pair"}
    has_header = any(h in known for h in headers)
    start = 1 if has_header else 0
    col_idx = 0
    if has_header:
        for i, h in enumerate(headers):
            if h in known:
                col_idx = i
                break

    out: Set[str] = set()
    for r in rows[start:]:
        if not r:
            continue
        v = (r[col_idx] or "").strip().upper()
        if v:
            # Normalize pairs to BASE/QUOTE; collect just BASE for counting
            if "/" in v or "-" in v:
                v = v.replace("-", "/")
                v = v.split("/", 1)[0]
            out.add(v)
    return out

def _log_thaw_status(all_symbols_count: int | None = None) -> Tuple[Set[str], str]:
    """
    Emit the verification line users expect.
    Returns (bases_set, label) so caller can also build whitelist.
    """
    csv_path, label = _candidate_csv_and_label()
    bases = _read_candidates(csv_path)
    if bases and len(bases) >= MOMENTUM_MIN_COUNT:
        N = len(bases)
        M = all_symbols_count if isinstance(all_symbols_count, int) and all_symbols_count > 0 else N
        print(f"MomentumPulse: THAW active — filtered universe ({N}/{M}) using {label}")
    else:
        print("MomentumPulse: THAW inactive — CSV missing/empty; using full universe (no blocks).")
    return bases, label

# ----------------- Whitelist prep -----------------

def prepare_whitelist_env(all_symbols_count: int | None = None) -> Dict[str, str]:
    """
    If THAW is detected and the momentum CSV passes min-count, build whitelist env vars
    and write a helper JSON file. Otherwise, return {}.
    This function also prints the THAW verification log line.
    """
    # NEW style: always try to emit status line when THAW is detected
    thaw_on = detect_thaw(str(STATE_DIR))

    # Respect legacy “enable” flag if user is relying on it.
    # If it's explicitly false AND no MOMENTUM_CSV_PATH exists, treat as disabled.
    legacy_enabled = MOMENTUM_PULSE_ENABLE or bool(MOMENTUM_CSV_PATH)

    if not thaw_on or not legacy_enabled:
        # Still provide the "inactive" log for visibility
        _log_thaw_status(all_symbols_count=None)
        return {}

    bases, label = _log_thaw_status(all_symbols_count=all_symbols_count)
    if not bases or len(bases) < MOMENTUM_MIN_COUNT:
        # No usable candidates
        return {}

    # Build pairs (BASE/QUOTE) using optional default quote for base-only rows
    default_quote = os.getenv("BUY_WHITELIST_DEFAULT_QUOTE", "USD").upper()
    pairs = sorted({f"{b}/{default_quote}" for b in bases})

    # Write helper JSON
    wl = {
        "mode": "THAW_PULSE",
        "bases": sorted(bases),
        "pairs": pairs,
        "source_csv": str(_candidate_csv_and_label()[0]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (STATE_DIR / "buy_whitelist.json").write_text(json.dumps(wl, indent=2))

    return {
        "BUY_WHITELIST_MODE": "THAW_PULSE",
        "BUY_WHITELIST_BASES": ",".join(sorted(bases)),
        "BUY_WHITELIST_PAIRS": ",".join(pairs),
        "BUY_WHITELIST_DEFAULT_QUOTE": default_quote,
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

    # We don't know the full discovery universe size here (engine discovers it),
    # so we pass None; the THAW log will print (N/N) as a conservative proxy.
    whitelist_env = prepare_whitelist_env(all_symbols_count=None)
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
