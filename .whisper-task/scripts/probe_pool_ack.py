#!/usr/bin/env python3
"""Minimal driver: run the pool's real ack_probe against candidate models,
optionally sweeping User-Agent headers. No business logic is duplicated —
this is the same code path refresh_model_pool uses to gate the pool."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refresh_model_pool as rpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base-url", default=rpm.ZEN_BASE)
    ap.add_argument("--api-key-env", default="ZEN_API_KEY")
    ap.add_argument("--user-agent", action="append", default=[],
                    metavar="UA", help="repeatable; omit to test requests default")
    args = ap.parse_args()

    if not os.environ.get(args.api_key_env):
        print(f"[pool-ack] {args.api_key_env} unset", file=sys.stderr)

    uas = args.user_agent or [None]
    for model in args.models:
        entry = {"model": model, "baseurl": args.base_url.rstrip("/")}
        for ua in uas:
            tag = ua or "python-requests(default)"
            ok, detail = rpm.ack_probe(entry, timeout=30, user_agent=ua)
            print(f"model={model} | ua={tag} | ok={ok} | {detail}")


if __name__ == "__main__":
    main()