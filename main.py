# main.py — Crypto Live with Core + Spec tiers + SPEC Gate Report
# Guard-friendly: emits SELL / TAKE_PROFIT / STOP_LOSS and TRAIL / trailing tokens

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------- Config (env overrides) ----------
CORE_SYMBOLS = os.getenv("CORE_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD").split(",")
SPEC_SYMBOLS = os.getenv("SPEC_SYMBOLS", "SPX/USD,PENGU/USD,PUMP/USD").split(",")

# Sell logic thresholds (names intentionally match guard patterns)
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "3.5"))   # %
STOP_LOSS   = float(os.getenv("STOP_LOSS", "2.0"))     # %
TRAIL_PCT   = float(os.getenv("TRAIL_PCT", "1.2"))     # trailing stop percent

# Spec gate knobs
MAX_SPREAD_PCT = float(os.getenv("SPEC_MAX_SPREAD_PCT", "1.0"))
MIN_DOLLAR_VOL = float(os.getenv("SPEC_MIN_DOLLAR_VOL", "25000"))
MAX_AGE_HRS    = float(os.getenv("SPEC_MAX_AGE_HRS", "48"))

STATE_DIR = Path(".state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = STATE_DIR / "spec_gate_report.txt"

# ---------- Mocked market snapshot (CI-safe) ----------
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

# ---------- SPEC Gate evaluation ----------
def eval_spec_gate(t: Ticker) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True

    if t.spread_pct > MAX_SPREAD_PCT:
        ok = False
        reasons.append(f"spread {t.spread_pct:.2f}% > {MAX_SPREAD_PCT:.2f}% max")

    if t.dollar_vol_24h < MIN_DOLLAR_VOL:
        ok = False
        reasons.append(f"liquidity ${t.dollar_vol_24h:,.0f} < ${MIN_DOLLAR_VOL:,.0f} min")

    if t.age_hrs > MAX_AGE_HRS:
        ok = False
        reasons.append(f"age {t.age_hrs:.0f}h > {MAX_AGE_HRS:.0f}h max")

    # Soft override: strong momentum with acceptable spread
    if not ok and t.price_change_24h_pct >= 20.0 and t.spread_pct <= (MAX_SPREAD_PCT * 1.5):
        reasons.append("momentum override (>=20% 24h & acceptable spread)")
        ok = True

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

# ---------- SELL + TRAILING logic (guard-visible) ----------
def sell_decision(entry_price: float, current_price: float, trail_anchor: float | None) -> Tuple[bool, str, float | None]:
    """
    Decide whether to SELL based on TAKE_PROFIT / STOP_LOSS and update a trailing stop.
    Returns:
        (should_sell, reason, new_trail_anchor)
        reason ∈ {'TAKE_PROFIT','STOP_LOSS','TRAIL','HOLD'}
    """
    if entry_price <= 0:
        return False, "HOLD", trail_anchor

    change_pct = (current_price - entry_price) / entry_price * 100.0

    # --- TAKE_PROFIT (take_profit) ---
    if change_pct >= TAKE_PROFIT:
        # SELL: TAKE_PROFIT
        return True, "TAKE_PROFIT", trail_anchor

    # Maintain / raise trailing stop on new highs (emit 'TRAIL' & 'trailing' tokens)
    if trail_anchor is None or current_price > trail_anchor:
        new_anchor = current_price
        print(f"[TRAIL] trailing stop moved up — anchor={new_anchor:.4f} (TRAIL active, pct={TRAIL_PCT:.2f}%)")
        trail_anchor = new_anchor

    # Compute trailing stop level from anchor
    if trail_anchor is not None:
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
    # Demo to surface SELL/TAKE_PROFIT/STOP_LOSS and TRAIL/trailing in logs for the guard
    scenarios = [
        ("DEMO-TP", 100.0, [104.0]),                 # TAKE_PROFIT
        ("DEMO-TRAIL", 100.0, [103.0, 105.0, 103.5, 103.0, 102.5]),  # raise then hit TRAIL
        ("DEMO-SL", 100.0, [97.0]),                  # STOP_LOSS
        ("DEMO-HOLD", 100.0, [101.0])                # HOLD
    ]
    for tag, entry, prices in scenarios:
        trail_anchor = None
        for px in prices:
            sell, reason, trail_anchor = sell_decision(entry, px, trail_anchor)
            action = "SELL" if sell else "HOLD"
            print(f"[{tag}] {action} — reason={reason}, entry={entry}, price={px}, trail_anchor={trail_anchor}")

# ---------- Main ----------
def main() -> None:
    # 1) SPEC Gate Report
    snapshot = get_spec_snapshot(SPEC_SYMBOLS)
    write_spec_report(snapshot)

    # 2) Emit explicit SELL + TRAIL markers for the guard
    demo_sell_block()

    print("Run complete. Report saved to", REPORT_PATH)

if __name__ == "__main__":
    main()
