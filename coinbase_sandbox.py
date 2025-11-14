#!/usr/bin/env python3
"""
coinbase_sandbox.py

Simple Coinbase Advanced Trade *Sandbox* smoke test.

- Uses the static sandbox endpoint:
    https://api-sandbox.coinbase.com/api/v3/brokerage/...

- DRY_RUN = ON  -> only previews an order (no real trade, even in sandbox)
- DRY_RUN = OFF -> actually calls the sandbox "create order" endpoint
                   (still fake money / mocked responses)

Environment (wired from GitHub Actions workflow):

  COINBASE_SANDBOX_KEY      (from GitHub vars, for future live use)
  COINBASE_SANDBOX_SECRET   (from GitHub vars, for future live use)
  PRODUCT_ID                (e.g. BTC-USD)
  SIDE                      (BUY or SELL)
  NOTIONAL_USD              (e.g. 10)
  DRY_RUN                   (ON / OFF)
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict

import requests

BASE_URL = "https://api-sandbox.coinbase.com/api/v3/brokerage"
HEADERS = {
    "Content-Type": "application/json",
    # X-Sandbox is a special header the sandbox understands
    "X-Sandbox": "true",
}


def log_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def mask(value: str | None) -> str:
    """Mask a key/secret so logs don't show the full thing."""
    if not value:
        return "(not set)"
    if len(value) <= 10:
        return "(set)"
    return f"{value[:6]}...{value[-4:]}"


def get_env() -> Dict[str, str]:
    env = {
        "product_id": os.getenv("PRODUCT_ID", "BTC-USD"),
        "side": os.getenv("SIDE", "BUY").upper(),
        "notional_usd": os.getenv("NOTIONAL_USD", "10"),
        "dry_run": os.getenv("DRY_RUN", "ON").upper(),
        "key": os.getenv("COINBASE_SANDBOX_KEY"),
        "secret": os.getenv("COINBASE_SANDBOX_SECRET"),
    }
    return env


def get_accounts() -> Dict[str, Any]:
    url = f"{BASE_URL}/accounts"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def preview_order(product_id: str, side: str, notional_usd: str) -> Dict[str, Any]:
    """
    Call the sandbox Preview Order endpoint.

    Minimal market IOC request:
      - market_market_ioc with quote_size (USD)
    """
    url = f"{BASE_URL}/orders/preview"
    body = {
        "product_id": product_id,
        "side": side,
        "order_configuration": {
            "market_market_ioc": {
                "quote_size": str(notional_usd),
            }
        },
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_order(product_id: str, side: str, notional_usd: str) -> Dict[str, Any]:
    """
    Call the sandbox Create Order endpoint.

    NOTE: still sandbox/fake; responses are static per docs.
    """
    url = f"{BASE_URL}/orders"
    body = {
        "product_id": product_id,
        "side": side,
        "order_configuration": {
            "market_market_ioc": {
                "quote_size": str(notional_usd),
            }
        },
        # Optional client_order_id could be added later
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=30)
    resp.raise_for_status()
    return resp.json()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def main() -> int:
    env = get_env()

    log_header("Coinbase Sandbox Smoke Test — Settings")
    print(
        textwrap.dedent(
            f"""
            PRODUCT_ID      : {env['product_id']}
            SIDE            : {env['side']}
            NOTIONAL_USD    : {env['notional_usd']}
            DRY_RUN         : {env['dry_run']}

            COINBASE_KEY    : {mask(env['key'])}
            COINBASE_SECRET : {mask(env['secret'])}

            Note: In the official Advanced Trade sandbox, these endpoints
            don't require authentication. We still load your key/secret
            so they're ready when we flip to LIVE later.
            """
        ).strip()
    )

    # 1) List accounts from sandbox
    try:
        log_header("Step 1 — List sandbox accounts")
        accounts = get_accounts()
        print("Raw sandbox accounts JSON (truncated):")
        print(pretty(accounts)[:2000])  # keep log manageable
    except requests.RequestException as e:
        print("\nERROR while calling /accounts on sandbox:")
        print(e)
        return 1

    # 2) Preview or create an order, depending on DRY_RUN
    if env["dry_run"] == "ON":
        try:
            log_header("Step 2 — DRY_RUN = ON → Preview sandbox order only")
            preview = preview_order(env["product_id"], env["side"], env["notional_usd"])
            print("Preview response (sandbox, no real trade):")
            print(pretty(preview)[:2000])
            print(
                "\nResult: Sandbox preview succeeded. No real funds moved "
                "(sandbox is static/fake responses)."
            )
        except requests.RequestException as e:
            print("\nERROR while previewing sandbox order:")
            print(e)
            return 1
    else:
        try:
            log_header("Step 2 — DRY_RUN = OFF → Create sandbox order")
            order = create_order(env["product_id"], env["side"], env["notional_usd"])
            print("Create-order response (sandbox):")
            print(pretty(order)[:2000])
            print(
                "\nResult: Sandbox order call succeeded. This is still fake money; "
                "responses are mocked in sandbox."
            )
        except requests.RequestException as e:
            print("\nERROR while creating sandbox order:")
            print(e)
            return 1

    log_header("Done")
    print("Coinbase Sandbox smoke test finished without fatal errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
