#!/usr/bin/env python3
"""
Normalize candidate CSVs so rotation always sees the columns it expects.

Targets (if present):
  .state/momentum_candidates.csv
  .state/spike_candidates.csv

Guarantees each row has:
  - pair  (normalized like 'SOL/USD')
  - rank  (int; higher is better)
  - quote (float; current price via Kraken public API if missing)

Any other columns in the source CSVs are preserved.

Usage:
  python tools/csv_schema_fix.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# Reuse our robust helpers
from trader.crypto_engine import get_public_quote, normalize_pair

STATE_DIR = Path(".state")
FILES = [
    STATE_DIR / "momentum_candidates.csv",
    STATE_DIR / "spike_candidates.csv",
]

def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if math.isfinite(v) and v > 0:
            return v
        return None
    except Exception:
        return None

def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def _rank_from_scores(rows: List[Dict[str, Any]], score_key: str = "score") -> List[int]:
    """
    Create a simple descending rank list based on 'score' (or 100.. down).
    Highest score -> highest rank number.
    If score missing, fallback to 100, 99, ...
    """
    # Try score-based
    scored = []
    has_any_score = False
    for i, r in enumerate(rows):
        s = r.get(score_key)
        try:
            val = float(s)
            has_any_score = True
        except Exception:
            val = float("-inf")
        scored.append((i, val))
    if has_any_score:
        # sort by score desc, keep original index
        scored.sort(key=lambda t: t[1], reverse=True)
        rank_map = {}
        # Assign descending ranks starting at 100
        base = 100
        for idx, (_i, _score) in enumerate(scored):
            rank_map[_i] = base - idx
        return [rank_map[i] for i in range(len(rows))]

    # No usable scores -> just 100, 99, ...
    base = 100
    return [base - i for i in range(len(rows))]

def _normalize_file(path: Path) -> None:
    if not path.exists():
        return

    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            # normalize keys to keep original casing for non-core fields
            norm_row = {k.strip(): v for k, v in row.items()}
            rows.append(norm_row)

    if not rows:
        return

    # ensure pair normalized
    for r in rows:
        p = r.get("pair") or r.get("symbol") or r.get("PAIR") or r.get("SYMBOL") or ""
        r["pair"] = normalize_pair(str(p))

    # ensure quote present & valid
    for r in rows:
        q = r.get("quote")
        qv = _to_float(q)
        if qv is None:
            qv = _to_float(get_public_quote(r["pair"]))
        r["quote"] = f"{qv:.12f}" if qv is not None else ""

    # ensure rank present; prefer existing numeric rank
    have_all_ranks = True
    ranks_existing: List[Optional[int]] = []
    for r in rows:
        rv = _to_int(r.get("rank"))
        ranks_existing.append(rv)
        if rv is None:
            have_all_ranks = False

    if not have_all_ranks:
        # compute from score (if present) else descending index
        auto_ranks = _rank_from_scores(rows, score_key="score")
        for i, r in enumerate(rows):
            if ranks_existing[i] is None:
                r["rank"] = auto_ranks[i]
            else:
                r["rank"] = ranks_existing[i]
    else:
        # coerce to int strings for consistency
        for r in rows:
            r["rank"] = _to_int(r["rank"])

    # Choose output headers:
    # Start with the core columns, then append all original columns except duplicates
    core = ["pair", "rank", "quote"]
    extras = []
    seen = set(core)
    for h in headers:
        hn = h.strip()
        if hn.lower() in ("pair", "rank", "quote"):
            continue
        if hn not in seen:
            extras.append(hn)
            seen.add(hn)
    out_headers = core + extras

    # Write back (overwrite in place)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        for r in rows:
            # keep extra fields as-is
            out = {h: r.get(h, "") for h in extras}
            out["pair"] = r.get("pair", "")
            out["rank"] = r.get("rank", "")
            out["quote"] = r.get("quote", "")
            writer.writerow(out)
    tmp.replace(path)

def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        try:
            _normalize_file(p)
        except Exception as e:
            # Non-fatal: continue with the next file
            (STATE_DIR / "csv_fix_error.log").write_text(f"{p.name}: {e}\n", encoding="utf-8")

    (STATE_DIR / "last_csv_fix.txt").write_text("ok\n", encoding="utf-8")

if __name__ == "__main__":
    main()
