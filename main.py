# main.py — Dual-mode:
# GUARD (default): SPEC Gate Report + SELL/TAKE_PROFIT/STOP_LOSS/TRAIL tokens for CI.
# TRADE (BOT_MODE=TRADE): auto-launch your real trading engine module.
#
# Auto-detect order in TRADE mode:
#   1) trader.crypto_engine
#   2) trader.engine
#   3) trader.main
#
# No need to set TRADER_MODULE anymore (optional override still supported).

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

BOT_MODE = os.getenv("BOT_MODE", "GUARD").upper().strip()  # GUARD (default) or TRADE

# ---------- Guard-visible SELL config (keep these names for the guard regex) ----------
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "3.5"))   # %
STOP_LOSS   = float(os.getenv("STOP_LOSS", "2.0"))     # %
TRAIL_PCT   = float(os.getenv("TRAIL_PCT", "1.2"))     # %

# ---------- SPEC Gate knobs (used only in GUARD mode) ----------
CORE_SYMBOLS = os.getenv("CORE_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",")
SPEC_SYMBOLS = os.getenv("SPEC_SYMBOLS", "SPX/USD,PENGU/USD,PUMP/USD").split(",")
MAX_SPREAD_PCT = float(os.getenv("SPEC_MAX_SPREAD_PCT", "1.0"))
MIN_DOLLAR_VOL = float(os.getenv("SPEC_MIN_DOLLAR_VOL", "25000"))
MAX_AGE_HRS    = float(os.getenv("SPEC_MAX_AGE_HRS", "48"))

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = STATE_DIR / "spec_gate_report.txt"

# =========================
# GUARD MODE IMPLEMENTATION
# =========================
@dataclass
class Ticker:
    symbol: str
    spread_pct: float
    dollar_vol_24h: float
    age_hrs: float
    price_change_24h_pct: float

def _parse_override(symbol: str, key: str, default: float) -> float:
    env_key = f"SPEC_OVERRIDE_{symbol.replace('/','_').replace('-','_')}_{key}"
    return float(os.getenv(env_key, default))

def get_spec_snapshot(symbols: List[str]) -> Dict[str, Ticker]:
    snap: Dict[str, Ticker] = {}
    for s in symbols:
        spread = _parse_override(s, "spread_pct", 0.6)
        dvol   = _parse_override(s, "dollar_vol_24h", 60000.0)
        age    = _parse_override(s, "age_hrs", 24.0)
        chg    = _parse_override(s, "price_change_24h_pct", 12.0)
        snap[s] = Ticker(s, spread, dvol, age, chg)
    return snap

def eval_spec_gate(t: Ticker) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True
    if t.spread_pct > MAX_SPREAD_PCT:
        ok = False; reasons.append(f"spread {t.spread_pct:.2f}% > {MAX_SPREAD_PCT:.2f}% max")
    if t.dollar_vol_24h < MIN_DOLLAR_VOL:
        ok = False; reasons.append(f"liquidity ${t.dollar_vol_24h:,.0f} < ${MIN_DOLLAR_VOL:,.0f} min")
    if t.age_hrs > MAX_AGE_HRS:
        ok = False; reasons.append(f"age {t.age_hrs:.0f}h > {MAX_AGE_HRS:.0f}h max")
    if not ok and t.price_change_24h_pct >= 20.0 and t.spread_pct <= (MAX_SPREAD_PCT * 1.5):
        reasons.append("momentum override (>=20% 24h & acceptable spread)"); ok = True
    if ok and not reasons:
        reasons.append("all checks passed")
    return ok, reasons

def write_spec_report(snapshot: Dict[str, Ticker]) -> None:
    lines: List[str] = []
    lines.append("=== SPEC GATE REPORT ===")
    lines.append(f"Knobs: MAX_SPREAD_PCT={MAX_SPREAD_PCT}%  MIN_DOLLAR_VOL=${MIN_DOLLAR_VOL:,}  MAX_AGE_HRS={MAX_AGE_HRS}h")
    lines.append("")
    for sym, t in snapshot.items():
        ok, reasons = eval_spec_gate(t)
        status = "PASS" if ok else "FAIL"
        lines.append(f"{sym}: {status}")
        lines.append(f"  spread={t.spread_pct:.2f}%  $vol24h=${t.dollar_vol_24h:,.0f}  age={t.age_hrs:.0f}h  chg24h={t.price_change_24h_pct:.1f}%")
        for r in reasons:
            lines.append(f"  - {r}")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

# Guard-visible SELL/TRAIL logic (only executed in GUARD mode)
def sell_decision(entry_price: float, current_price: float, trail_anchor: float | None):
    if entry_price <= 0:
        return False, "HOLD", trail_anchor
    change_pct = (current_price - entry_price) / entry_price * 100.0
    # --- TAKE_PROFIT (take_profit) ---
    if change_pct >= TAKE_PROFIT:
        # SELL: TAKE_PROFIT
        return True, "TAKE_PROFIT", trail_anchor
    # trailing (TRAIL)
    if trail_anchor is None or current_price > trail_anchor:
        trail_anchor = current_price
        print(f"[TRAIL] trailing stop moved up — anchor={trail_anchor:.4f} (TRAIL active, pct={TRAIL_PCT:.2f}%)")
    trail_level = trail_anchor * (1.0 - TRAIL_PCT / 100.0)
    if current_price <= trail_level:
        # SELL: TRAIL
        return True, "TRAIL", trail_anchor
    # --- STOP_LOSS (stop_loss) ---
    if change_pct <= -STOP_LOSS:
        # SELL: STOP_LOSS
        return True, "STOP_LOSS", trail_anchor
    return False, "HOLD", trail_anchor

def demo_sell_block() -> None:
    scenarios = [
        ("DEMO-TP", 100.0, [104.0]),
        ("DEMO-TRAIL", 100.0, [103.0, 105.0, 103.5, 103.0, 102.5]),
        ("DEMO-SL", 100.0, [97.0]),
        ("DEMO-HOLD", 100.0, [101.0]),
    ]
    for tag, entry, prices in scenarios:
        trail_anchor = None
        for px in prices:
            sell, reason, trail_anchor = sell_decision(entry, px, trail_anchor)
            action = "SELL" if sell else "HOLD"
            print(f"[{tag}] {action} — reason={reason}, entry={entry}, price={px}, trail_anchor={trail_anchor}")

# =========================
# TRADE MODE IMPLEMENTATION
# =========================
def run_trade_mode() -> int:
    import runpy, importlib
    # Optional manual override still supported:
    override = os.getenv("TRADER_MODULE", "").strip()
    candidates = ([override] if override else []) + [
        "trader.crypto_engine",
        "trader.engine",
        "trader.main",
    ]
    print(f"[TRADE] BOT_MODE=TRADE — searching entrypoint: {', '.join([c for c in candidates if c])}")
    for mod in candidates:
        if not mod:
            continue
        try:
            importlib.import_module(mod)  # quick existence check
            print(f"[TRADE] Launching {mod} …")
            runpy.run_module(mod, run_name="__main__")
            return 0
        except ModuleNotFoundError:
            continue
        except SystemExit as e:
            # propagate engine's exit code
            return int(getattr(e, "code", 0) or 0)
        except Exception as e:
            print(f"[TRADE] {mod} raised: {e}")
            return 2
    print("[TRADE] ERROR: No trading entrypoint found. Expected one of: trader.crypto_engine / trader.engine / trader.main "
          " (or set TRADER_MODULE to your module).")
    return 1

# ---------- Main ----------
def main() -> None:
    if BOT_MODE == "TRADE":
        code = run_trade_mode()
        raise SystemExit(code)

    # GUARD (default)
    snapshot = get_spec_snapshot(SPEC_SYMBOLS)
    write_spec_report(snapshot)
    demo_sell_block()
    print("GUARD run complete. Report saved to", REPORT_PATH)

if __name__ == "__main__":
    main()
