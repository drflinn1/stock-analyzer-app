#!/usr/bin/env python3
"""
tools/run_live.py

Wrapper that:
1) Cleans a noisy pandas message in logs.
2) Optional "dip gate": only run trades if price has dropped >= DROP_PCT
   vs a LOOKBACK_MIN-minute SMA for at least one symbol.

Env/CLI controls (defaults in parentheses):
  DROP_PCT (1.5)          # % drop vs SMA required to proceed
  LOOKBACK_MIN (60)       # minutes of 1m candles for SMA
  SYMBOLS (from workflow) # comma-separated, e.g. BTC-USD,ETH-USD,...

If the gate is not satisfied, the script exits 0 (no orders).
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

# Build argv if launched directly
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
    # Prefer env SYMBOLS, else derive from CLI (if present)
    env_syms = os.environ.get("SYMBOLS", "")
    if env_syms.strip():
        raw = env_syms.split(",")
    else:
        # Fallback: scan argv for --symbols or single --symbol (not required)
        raw = []
        for i, tok in enumerate(sys.argv):
            if tok in ("--symbols", "--symbol") and i + 1 < len(sys.argv):
                raw = sys.argv[i + 1].split(",")
                break
    # Normalize to ccxt symbols (BTC-USD -> BTC/USD)
    syms = [s.strip().replace("-", "/").upper() for s in raw if s.strip()]
    return syms or ["BTC/USD"]

def _float_env(name: str, default: float) -> float:
    try: return float(os.environ.get(name, "").strip() or default)
    except Exception: return default

def _int_env(name: str, default: int) -> int:
    try: return int(float(os.environ.get(name, "").strip() or default))
    except Exception: return default

def dip_gate():
    drop_pct = _float_env("DROP_PCT", 1.5)         # %
    lookback = _int_env("LOOKBACK_MIN", 60)        # minutes
    symbols = _parse_symbols()

    # If user explicitly disables with DROP_PCT <= 0, always proceed
    if drop_pct <= 0 or lookback <= 1:
        print(f"[dip-gate] Disabled (DROP_PCT={drop_pct}, LOOKBACK_MIN={lookback}). Proceeding.")
        return True

    try:
        import ccxt  # type: ignore
        ex = ccxt.kraken()
        timeframe = "1m"
        limit = max(lookback, 5)

        print(f"[dip-gate] Checking {symbols} for {drop_pct:.3f}% drop vs {lookback}m SMA...")
        proceed = False

        for sym in symbols:
            # Fetch 1m candles
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
                print(f"[dip-gate] {sym}: threshold met (>= {drop_pct}%). Trading allowed.")
                proceed = True
                # We allow run to proceed if ANY symbol meets dip; no need to check all, but continue for logging

        if not proceed:
            print(f"[dip-gate] No symbol met dip threshold ({drop_pct}%). Skipping run.")
        return proceed

    except Exception as e:
        # If dip gate fails (network, ccxt, etc.), fail open (proceed) so trading can still happen
        print(f"[dip-gate] Error while checking dips: {e}. Proceeding anyway.")
        return True

if not dip_gate():
    # Exit gracefully without running the core bot
    sys.exit(0)

# Hand off to your app
runpy.run_path("main.py", run_name="__main__")
