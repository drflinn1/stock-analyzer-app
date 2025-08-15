# runner.py
import os, json, argparse
import engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", default=os.environ.get("DRY_RUN", "true"))
    args = parser.parse_args()

    # call the engine
    res = engine.run_engine()

    # if you later add commits/orders, gate them on not dry-run
    dry = str(args.dry_run).lower() in ("1", "true", "yes", "y")
    res["dry_run"] = dry
    print(json.dumps(res))

if __name__ == "__main__":
    main()
