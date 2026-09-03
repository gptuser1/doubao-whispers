#!/usr/bin/env python3
"""Minimal driver: run the pool's real ack_probe against candidate models.
No business logic is duplicated — this is the same code path
refresh_model_pool uses to gate the pool."""
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
    args = ap.parse_args()

    if not os.environ.get(args.api_key_env):
        print(f"[pool-ack] {args.api_key_env} unset", file=sys.stderr)

    for model in args.models:
        entry = {"model": model, "baseurl": args.base_url.rstrip("/")}
        ok, detail = rpm.ack_probe(entry, timeout=30)
        print(f"model={model} | ok={ok} | {detail}")


if __name__ == "__main__":
    main()