#!/usr/bin/env python3
# (same code I gave—safe to paste over)
# --- start ---
from __future__ import annotations
import csv, os
from pathlib import Path
from typing import Iterable, List, Set, Tuple

_HEADERS = {"symbol", "base", "ticker", "pair"}

def _candidate_csv_path() -> Tuple[Path, str]:
    env_path = os.getenv("MOMENTUM_CSV_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p, p.name
    for p in (Path("momentum_candidates.csv"), Path(".state") / "momentum_candidates.csv"):
        if p.exists():
            return p, p.name
    return Path("momentum_candidates.csv"), "momentum_candidates.csv"

def _read_candidates(csv_path: Path) -> Set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return set()
    first = [c.strip() for c in rows[0]]
    header_lc = [c.lower() for c in first]
    has_header = any(h in _HEADERS for h in header_lc)
    start_idx = 1 if has_header else 0
    col_idx = 0
    if has_header:
        for i, name in enumerate(header_lc):
            if name in _HEADERS:
                col_idx = i; break
    symbols: Set[str] = set()
    for r in rows[start_idx:]:
        if not r: continue
        sym = r[col_idx].strip().upper()
        if sym: symbols.add(sym)
    return symbols

def _min_count() -> int:
    try:
        return max(1, int(os.getenv("MOMENTUM_MIN_COUNT", "1")))
    except Exception:
        return 1

def maybe_filter_universe(all_symbols: Iterable[str]) -> List[str]:
    all_list = [s.upper() for s in all_symbols]
    csv_path, label = _candidate_csv_path()
    candidates = _read_candidates(csv_path)
    min_needed = _min_count()
    if candidates and len(candidates) >= min_needed:
        filtered = [s for s in all_list if s in candidates]
        if filtered:
            print(f"MomentumPulse: THAW active — filtered universe ({len(filtered)}/{len(all_list)}) using {label}")
            return filtered
    print("MomentumPulse: THAW inactive — CSV missing/empty; using full universe (no blocks).")
    return all_list

if __name__ == "__main__":
    demo_all = ["BTC","ETH","SOL","ADA","DOGE"]
    out = maybe_filter_universe(demo_all)
    print("Universe:", out)
# --- end ---
