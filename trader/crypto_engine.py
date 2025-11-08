name: Crypto — Hourly 1-Coin Rotation (LIVE-ready, ultra-defensive)

on:
  schedule:
    - cron: "0 * * * *"   # hourly at minute 0 UTC
  workflow_dispatch:

permissions:
  contents: write
  actions: read

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 12

    env:
      # RISK MODE (repo variable DRY_RUN can still override this default)
      DRY_RUN: ${{ vars.DRY_RUN || 'ON' }}             # ON=paper, OFF=live

      # --- Your engine's knobs (crypto_engine.py expects these) ---
      MIN_BUY_USD:       ${{ vars.MIN_BUY_USD || '15' }}
      MAX_BUYS_PER_RUN:  ${{ vars.MAX_BUYS_PER_RUN || '1' }}
      MAX_POSITIONS:     ${{ vars.MAX_POSITIONS || '3' }}
      RESERVE_CASH_PCT:  ${{ vars.RESERVE_CASH_PCT || '0' }}
      UNIVERSE_TOP_K:    ${{ vars.UNIVERSE_TOP_K || '25' }}
      MIN_24H_PCT:       ${{ vars.MIN_24H_PCT || '0' }}
      MIN_BASE_VOL_USD:  ${{ vars.MIN_BASE_VOL_USD || '10000' }}
      WHITELIST:         ${{ vars.WHITELIST || '' }}

      # SELL GUARD knobs
      SELL_HARD_STOP_PCT:   ${{ vars.SELL_HARD_STOP_PCT || '3' }}
      SELL_TRAIL_PCT:       ${{ vars.SELL_TRAIL_PCT || '2' }}
      SELL_TAKE_PROFIT_PCT: ${{ vars.SELL_TAKE_PROFIT_PCT || '5' }}
      BACKFILL_LOOKBACK_DAYS: ${{ vars.BACKFILL_LOOKBACK_DAYS || '60' }}

      # Keys (canonical or legacy)
      KRAKEN_API_KEY:    ${{ secrets.KRAKEN_API_KEY || secrets.KRAKEN_KEY || '' }}
      KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET || secrets.KRAKEN_SECRET || '' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies (force-install core deps)
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt || true
          fi
          # hard requirements for your engine & utils
          pip install --upgrade --no-input ccxt requests pandas python-dateutil pyyaml

      - name: Echo critical env (no secrets)
        run: |
          echo "DRY_RUN=${DRY_RUN}"
          echo "MIN_BUY_USD=${MIN_BUY_USD}"
          echo "MAX_BUYS_PER_RUN=${MAX_BUYS_PER_RUN}"
          echo "MAX_POSITIONS=${MAX_POSITIONS}"
          echo "RESERVE_CASH_PCT=${RESERVE_CASH_PCT}"
          echo "UNIVERSE_TOP_K=${UNIVERSE_TOP_K}"
          echo "MIN_24H_PCT=${MIN_24H_PCT}"
          echo "MIN_BASE_VOL_USD=${MIN_BASE_VOL_USD}"
          echo "WHITELIST=${WHITELIST:-<empty>}"
          echo "SELL_HARD_STOP_PCT=${SELL_HARD_STOP_PCT}"
          echo "SELL_TRAIL_PCT=${SELL_TRAIL_PCT}"
          echo "SELL_TAKE_PROFIT_PCT=${SELL_TAKE_PROFIT_PCT}"
          echo "BACKFILL_LOOKBACK_DAYS=${BACKFILL_LOOKBACK_DAYS}"

      - name: Key sanity check (no secret values shown)
        run: |
          missing=0
          if [ -z "${KRAKEN_API_KEY}" ]; then echo "::error::Missing KRAKEN_API_KEY (or KRAKEN_KEY)"; missing=1; fi
          if [ -z "${KRAKEN_API_SECRET}" ]; then echo "::error::Missing KRAKEN_API_SECRET (or KRAKEN_SECRET)"; missing=1; fi
          if [ "${DRY_RUN}" = "OFF" ] && [ $missing -ne 0 ]; then
            echo "::error::LIVE mode requires API keys in repo Secrets."; exit 1
          fi
          echo "Keys present check: OK."

      # Always have something to upload
      - name: Prepare .state placeholder
        run: |
          mkdir -p .state
          echo "Placeholder (pre-run) $(date -u)" > .state/run_summary.md

      - name: Run rotation
        env:
          PYTHONPATH: ${{ github.workspace }}:${{ github.workspace }}/src:${{ github.workspace }}/trader
        run: |
          python -u main.py || true
          echo "Contents of .state after run:"
          ls -la .state || true
          tar -czf state_bundle.tgz .state || true
          echo "Created state_bundle.tgz size:" $(du -h state_bundle.tgz | cut -f1)

      - name: Upload .state artifacts (folder + tarball)
        uses: actions/upload-artifact@v4
        with:
          name: crypto-hourly-rotation-${{ github.run_id }}
          path: |
            .state
            state_bundle.tgz
          if-no-files-found: warn
