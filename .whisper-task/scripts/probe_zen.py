#!/usr/bin/env python3
"""Endpoint + User-Agent probe for opencode zen models.

Tests the same ACK prompt across:
  - endpoint: /responses (OpenAI Responses API) or /chat/completions
  - user-agent variants (zen may gate/livelock UA, e.g. only opencode client UAs)
Prints per (model, endpoint, UA) the HTTP status, status, acked, and token usage.
"""

import json
import os
import sys
import time
import argparse
import urllib.error
import urllib.request

ZEN_BASE = "https://opencode.ai/zen/v1"
ACK_PROMPT = "Reply with exactly: ack"
DEFAULT_UA = "doubao-whispers/1.0"

# UA <- application semantics: opencode binary, ai-sdk provider-utils, runtime
UA_OPENCODE = "opencode/1.15.0 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13"


def zen_ack(base_url, api_key, model, endpoint, max_output_tokens=128,
            reasoning_effort=None, user_agent=DEFAULT_UA, timeout=90):
    """ACK over the requested endpoint; returns (status, body)."""
    if endpoint == "responses":
        payload = {
            "model": model,
            "input": ACK_PROMPT,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
    else:  # chat/completions
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ACK_PROMPT}],
            "max_tokens": max_output_tokens,
            "temperature": 0,
        }
    req = urllib.request.Request(
        f"{base_url}/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode()), time.time() - t0


def extract_text(body, endpoint):
    if endpoint == "responses":
        texts = []
        for item in body.get("output") or []:
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
        return "\n".join(texts).strip()
    choices = body.get("choices") or []
    return (choices[0].get("message") or {}).get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--endpoint", default="responses", choices=["responses", "chat"])
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "low", "medium", "high"])
    ap.add_argument("--base-url", default=ZEN_BASE)
    ap.add_argument("--api-key-env", default="ZEN_API_KEY")
    ap.add_argument("--user-agent", action="append", dest="uas",
                    help="repeatable; defaults to opencode-style UA set")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[probe-zen] {args.api_key_env} unset — sending without auth header",
              file=sys.stderr)

    uas = args.uas or [UA_OPENCODE, "opencode/1.15.0", DEFAULT_UA, "curl/8.5.0"]

    for model in args.models:
        for ua in uas:
            print(f"\n{'='*70}\nMODEL: {model} | endpoint: {args.base_url}/{args.endpoint} "
                  f"| UA: {ua}\n{'='*70}")
            try:
                status, body, secs = zen_ack(
                    args.base_url, api_key, model, args.endpoint,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    user_agent=ua,
                )
            except urllib.error.HTTPError as e:
                print(f"UA: {ua} -> HTTP {e.code} in ?s: {e.read().decode()[:300]}")
                continue
            except Exception as e:
                print(f"UA: {ua} -> ERROR: {repr(e)}")
                continue

            out = extract_text(body, args.endpoint)
            usage = body.get("usage") or {}
            otok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            rtok = (usage.get("output_tokens_details") or
                    usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            acked = out.strip().lower().startswith("ack")
            print(f"UA: {ua} -> HTTP {status} | status={body.get('status')} "
                  f"| acked={acked} | {secs:.1f}s | otok={otok} reasoning={rtok}")
            print("    output:", (out or "(empty)")[:80])
            if not acked:
                print("    raw:", json.dumps(body, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()