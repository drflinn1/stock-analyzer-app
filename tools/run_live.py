#!/usr/bin/env python3
"""
tools/run_live.py

Wrapper that:
1) Cleans a noisy pandas message in logs.
2) Dip gate: only allow BUYS when price has dropped >= DROP_PCT vs LOOKBACK_MIN SMA.
   - If the dip gate FAILS, we still run but force sells only (buys disabled).
   - If the dip gate PASSES, we run normally (buys & sells).

Env/CLI controls (defaults in parentheses):
  DROP_PCT (1.5)          # % drop vs SMA required to allow buys
  LOOKBACK_MIN (60)       # minutes of 1m candles for SMA
  SYMBOLS (from workflow) # comma-separated, e.g. BTC-USD,ETH-USD,...

If DROP_PCT <= 0, the dip gate is disabled (both buys & sells allowed).
"""

import os
import sys
import io
import runpy
from typing import List

NOISY = "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()"
CLEAN = "Indicator condition ambiguous (likely Series vs scalar). Skipped trade."

class _FilterStream(io.TextIOBase):
    def __init__(self, orig): self._orig = orig
    def write(self, s):
        try: s = s.replace(NOISY, CLEAN)
        except Exception: pass
        return self._orig.write(s)
    def flush(self):
        try: self._orig.flush()
        except Exception: pass

# Clean logs
sys.stdout = _FilterStream(sys.stdout)
sys.stderr = _FilterStream(sys.stderr)

# Build argv if launched directly (same defaults as before)
if len(sys.argv) == 1:
    sys.argv = [
        "main.py",
        "--market", "crypto",
        "--exchange", "kraken",
        "--strategy", "cautious",
        "--cap_per_trade", "10",
        "--cap_daily", "15",
        "--dry_run", "false",
    ]

# Ensure live mode (belt + suspenders)
os.environ.setdefault("TRADE_MODE", "live")
os.environ.setdefault("DRY_RUN", "false")

# -------------------------
# Dip gate (ccxt + 1m OHLC)
# -------------------------
def _parse_symbols() -> List[str]:
    env_syms = os.environ.get("SYMBOLS", "")
    if env_syms.strip():
        raw = env_syms.split(",")
    else:
        # fallback: scan argv for --symbols/--symbol if present
        raw = []
        for i, tok in enumerate(sys.argv):
            if tok in ("--symbols", "--symbol") and i + 1 < len(sys.argv):
                raw = sys.argv[i + 1].split(",")
                break
    syms = [s.strip().replace("-", "/").upper() for s in raw if s.strip()]
    return syms or ["BTC/USD"]

def _float_env(name: str, default: float) -> float:
    try: return float(os.environ.get(name, "").strip() or default)
    except Exception: return default

def _int_env(name: str, default: int) -> int:
    try: return int(float(os.environ.get(name, "").strip() or default))
    except Exception: return default

def dip_gate_allows_buys() -> bool:
    drop_pct = _float_env("DROP_PCT", 1.5)         # %
    lookback = _int_env("LOOKBACK_MIN", 60)        # minutes
    symbols = _parse_symbols()

    # Disable gate entirely with DROP_PCT<=0
    if drop_pct <= 0 or lookback <= 1:
        print(f"[dip-gate] Disabled (DROP_PCT={drop_pct}, LOOKBACK_MIN={lookback}). Buys & sells allowed.")
        return True

    try:
        import ccxt  # type: ignore
        ex = ccxt.kraken()
        timeframe = "1m"
        limit = max(lookback, 5)

        print(f"[dip-gate] Checking {symbols} for {drop_pct:.3f}% drop vs {lookback}m SMA...")
        allow_buys = False

        for sym in symbols:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 3:
                print(f"[dip-gate] {sym}: insufficient candles ({len(ohlcv) if ohlcv else 0}); skipping symbol.")
                continue

            closes = [c[4] for c in ohlcv[-lookback:]] if len(ohlcv) >= lookback else [c[4] for c in ohlcv]
            sma = sum(closes) / len(closes)
            last = closes[-1]
            if sma <= 0:
                print(f"[dip-gate] {sym}: invalid SMA; skipping.")
                continue

            drop = (sma - last) / sma * 100.0
            print(f"[dip-gate] {sym}: last={last:.6f} SMA={sma:.6f} drop={drop:.3f}%")

            if drop >= drop_pct:
                print(f"[dip-gate] {sym}: threshold met (>= {drop_pct}%). Buys allowed.")
                allow_buys = True
                # continue logging for other symbols

        if not allow_buys:
            print(f"[dip-gate] No symbol met dip threshold ({drop_pct}%). Buys will be blocked; sells allowed.")
        return allow_buys

    except Exception as e:
        # Fail open (both buys/sells allowed) if gate cannot evaluate for any reason
        print(f"[dip-gate] Error while checking dips: {e}. Buys & sells allowed.")
        return True

# Decide how to run main.py
buys_allowed = dip_gate_allows_buys()

# If buys are NOT allowed, force sells only for this run by appending --force_side sell
argv = list(sys.argv)
if "--force_side" not in argv:
    if not buys_allowed:
        print("[dip-gate] Enforcing sells-only this run (--force_side sell).")
        argv.extend(["--force_side", "sell"])
    else:
        # explicit clarity
        argv.extend(["--force_side", "(none)"])

# Hand off to your app with possibly-adjusted argv
sys.argv = argv
runpy.run_path("main.py", run_name="__main__")
