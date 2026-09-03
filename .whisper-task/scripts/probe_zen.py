#!/usr/bin/env python3
"""ACK liveness probe for opencode zen models served via the OpenAI
Responses API (https://opencode.ai/zen/v1/responses).

The model-pool refetch uses a chat/completions ACK probe (see
refresh_model_pool.ack_probe), which may 404 on models only exposed
through the /responses endpoint. This script pings the same candidate set
through /v1/responses to check whether ACK actually works there.
"""

import json
import os
import sys
import argparse
import urllib.error
import urllib.request

ZEN_BASE = "https://opencode.ai/zen/v1"
ACK_PROMPT = "Reply with exactly: ack"


def responses_ack(base_url, api_key, model, max_output_tokens=128, timeout=60):
    """Send an ACK prompt over the Responses API; return (status, body)."""
    payload = {
        "model": model,
        "input": ACK_PROMPT,
        "max_output_tokens": max_output_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "doubao-whispers/1.0",
        },
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def extract_text(body):
    """Pulls assistant output text out of a Responses API body."""
    texts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
    return "\n".join(texts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base-url", default=ZEN_BASE)
    ap.add_argument("--api-key-env", default="ZEN_API_KEY")
    ap.add_argument("--max-output-tokens", type=int, default=128)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[probe-zen] {args.api_key_env} unset — sending without auth header",
              file=sys.stderr)

    for model in args.models:
        print(f"\n{'='*64}\nMODEL: {model}  (ACK via {args.base_url}/responses)\n{'='*64}")
        try:
            status, body = responses_ack(
                args.base_url, api_key, model,
                max_output_tokens=args.max_output_tokens,
            )
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:500]}")
            continue
        except Exception as e:
            print(f"ERROR: {repr(e)}")
            continue

        out = extract_text(body)
        usage = body.get("usage") or {}
        acked = out.strip().lower().startswith("ack")
        print(f"HTTP {status} | status={body.get('status')} | acked={acked} | "
              f"itok={usage.get('input_tokens')} otok={usage.get('output_tokens')}")
        print("---- output ----")
        print(out or "(empty)")
        if not acked:
            print("---- raw body (first 800 chars) ----")
            print(json.dumps(body, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()