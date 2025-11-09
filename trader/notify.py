#!/usr/bin/env python3
"""
Simple Slack webhook notifier that doesn't require 'requests'.
Set SLACK_WEBHOOK_URL in repo Variables/Secrets to enable.
"""
from __future__ import annotations
import json, os, sys, urllib.request

def notify_slack(text: str) -> None:
    url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
    except Exception as e:
        print(f"[notify] Slack send failed: {e}", file=sys.stderr)
