# momentum_pulse.py
import os
import re
import yaml
from typing import Iterable, List, Set, Tuple

# --- Public API --------------------------------------------------------------

def detect_thaw(state_dir: str = ".state") -> bool:
    """
    Detect a THAW window using multiple heuristics so we work with your existing
    guards without breaking older states.
    - .state/thaw.flag            -> if file exists => THAW
    - .state/guard.yaml           -> if {mode: THAW} or {THAW: true}
    """
    thaw_flag = os.path.join(state_dir, "thaw.flag")
    if os.path.exists(thaw_flag):
        return True

    guard_yaml = os.path.join(state_dir, "guard.yaml")
    if os.path.exists(guard_yaml):
        try:
            with open(guard_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Accept a few shapes
            mode = str(data.get("mode") or data.get("MODE") or "").strip().upper()
            if mode == "THAW":
                return True
            thaw_bool = data.get("thaw") or data.get("THAW")
            if isinstance(thaw_bool, bool) and thaw_bool:
                return True
        except Exception:
            # Don't block trading if state is malformed
            return False
    return False


def load_momentum_candidates(csv_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Load candidate symbols from CSV. We accept flexible column names:
      - symbol, pair, base, quote, rank (rank optional)
    Output is two sets:
      bases:  {"BTC", "ETH", ...}
      pairs:  {"BTC/USD", "ETH/USD", "SOL/USDT", ...}  (normalized with slash)
    """
    if not os.path.exists(csv_path):
        return set(), set()

    try:
        import csv
        bases: Set[str] = set()
        pairs: Set[str] = set()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize fieldnames
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            for row in reader:
                def get(name: str):
                    key = cols.get(name)
                    return (row.get(key) if key else None)

                sym = (get("symbol") or get("base") or "").strip().upper()
                quote = (get("quote") or "").strip().upper()
                pair = (get("pair") or "").strip().upper()

                if sym:
                    bases.add(sym)
                if pair:
                    pairs.add(normalize_pair(pair))
                elif sym and quote:
                    pairs.add(f"{sym}/{quote}")
        return bases, pairs
    except Exception:
        # On any parse issue, fail safe (no filter)
        return set(), set()


def filter_universe_for_thaw(
    universe: Iterable[str],
    pulse_enabled: bool,
    csv_path: str,
    logger=None,
) -> List[str]:
    """
    If pulse is enabled *and* we're currently in THAW, restrict the buy universe
    to the CSV candidates. Otherwise, return the original universe.

    `universe` may contain base symbols (BTC, ETH) or pairs (BTC/USD, ETH-USDT).
    We try to match either.
    """
    if not pulse_enabled:
        return list(universe)

    in_thaw = detect_thaw()
    if not in_thaw:
        return list(universe)

    bases, pairs = load_momentum_candidates(csv_path)
    if not bases and not pairs:
        if logger:
            logger.warning(
                "MOMENTUM_PULSE_ENABLE is true, THAW detected, but no candidates found in %s — falling back to full universe.",
                csv_path,
            )
        return list(universe)

    keep: List[str] = []
    for u in universe:
        ub = safe_base_from_symbol(u)
        up = normalize_pair(u)
        if ub in bases or up in pairs:
            keep.append(u)

    if logger:
        logger.info(
            "MomentumPulse: THAW active — filtered universe from %d → %d using %s",
            len(list(universe)), len(keep), os.path.basename(csv_path)
        )
    return keep if keep else list(universe)

# --- Internals ---------------------------------------------------------------

_pair_re = re.compile(r"^([A-Z0-9]+)[-/]?([A-Z0-9]+)?$")

def safe_base_from_symbol(s: str) -> str:
    s = (s or "").upper().strip()
    if "/" in s:
        return s.split("/")[0]
    if "-" in s:
        return s.split("-")[0]
    m = _pair_re.match(s)
    if m:
        return m.group(1)
    return s

def normalize_pair(s: str) -> str:
    s = (s or "").upper().strip().replace("-", "/")
    if "/" not in s:
        return s
    base, quote = s.split("/", 1)
    return f"{base}/{quote}"
