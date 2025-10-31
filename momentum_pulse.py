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
- .state/thaw.flag -> if file exists => THAW
- .state/guard.yaml -> if {mode: THAW} or {THAW: true}
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
bases: {"BTC", "ETH", ...}
pairs: {"BTC/USD", "ETH/USD", "SOL/USDT", ...} (normalized with slash)
"""
if not os.path.exists(csv_path):
return set(), set()


return f"{base}/{quote}"
