# runner.py
import os, json, argparse, time, pathlib, csv
from typing import Dict, Any
import engine

ART_DIR = pathlib.Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

def write_csv(returns: Dict[str, float], path: pathlib.Path) -> None:
    # Write header + rows: symbol, percent_return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "percent_return"])
        for sym, val in sorted(returns.items()):
            w.writerow([sym, "" if val is None else f"{val:.6f}"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", default=os.environ.get("DRY_RUN", "true"))
    args = parser.parse_args()

    res: Dict[str, Any] = engine.run_engine()

    # Attach dry_run flag
    dry = str(args.dry_run).lower() in ("1", "true", "yes", "y")
    res["dry_run"] = dry

    # Persist artifacts
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = ART_DIR / f"signals-{ts}.csv"
    json_path = ART_DIR / f"signals-{ts}.json"

    returns = res.get("computed_returns") or {}
    write_csv(returns, csv_path)

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(res, jf, ensure_ascii=False, indent=2)

    # Print compact JSON to the log (unchanged behavior)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
