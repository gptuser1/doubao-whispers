#!/usr/bin/env python3
"""Probe zen free-tier gates (2026-09 change).

Community reverse-engineering (9router #4124 / #4132) found zen now
fingerprints requests with four gates:
  1. UA must look like the real opencode client
  2. session id must be a proper "ses_" shape
  3. the payload must declare the opencode tool set (0-3 tools -> 403,
     the bash/glob/grep/read quartet -> 200)
  4. streaming: `stream:false` -> 403 on both endpoints

This script tests which case is enough to get a usable text reply (content,
not tool_calls) for each candidate free model, on the exact

base the project uses: POST {zen}/chat/completions with the shared shim
headers from ai_client._oc_shim_headers(). Business logic is NOT touched.

Usage:
    python probe_zen_fingerprint.py --models ling-3.0-flash-fin-free mimo-v2.5-free
"""
import argparse
import json
import os
import sys
import time
import uuid

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_client import _oc_shim_headers

ZEN_BASE = "https://opencode.ai/zen/v1"


# --- opencode tool signatures -------------------------------------------------
# Names must be exact (fake tool names fail the gate). Schemas are permissive;
# zen checks the declared tool set, not the schema details.

def _fn(name, desc, props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(props)[:1],
            },
        },
    }


TOOLS_QUARTET = [
    _fn("bash", "Safely execute commands in a bash shell within a persistent session.",
        {"command": {"type": "string"}}),
    _fn("glob", "Find files by glob pattern.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    _fn("grep", "Find lines in files matching a regex.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    _fn("read", "Read a file from the filesystem.",
        {"filePath": {"type": "string"}}),
]

TOOLS_ALL = TOOLS_QUARTET + [
    _fn("edit", "Edit a file in place.", {"filePath": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
    _fn("write", "Write a file to the filesystem.", {"filePath": {"type": "string"}, "content": {"type": "string"}}),
    _fn("webfetch", "Fetch a URL and return its content.", {"url": {"type": "string"}}),
    _fn("task", "Run a task in a sub-agent.", {"description": {"type": "string"}}),
    _fn("todowrite", "Write a todo list for the session.", {"todos": {"type": "array", "items": {"type": "object"}}}),
    _fn("skill", "Use a skill.", {"skill": {"type": "string"}}),
]


# --- cases --------------------------------------------------------------------
# Each case is a dict of payload tweaks relative to the baseline below.

def build_body(model, tools=None, stream=True, tool_choice=None,
               system=None, user="只用文字回答：ack"):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return body


CASES = [
    {
        "name": "A_no_tools_nostream",
        "desc": "baseline: current ai_client behavior (no tools, stream=false) -> should 403",
        "body": lambda m: build_body(m, tools=None, stream=False),
    },
    {
        "name": "B_quartet_nostream",
        "desc": "tools quartet, stream=false -> tests tool gate alone",
        "body": lambda m: build_body(m, tools=TOOLS_QUARTET, stream=False),
    },
    {
        "name": "C_quartet_stream",
        "desc": "tools quartet, stream=true -> community-confirmed 200 combo",
        "body": lambda m: build_body(m, tools=TOOLS_QUARTET, stream=True),
    },
    {
        "name": "D_all_stream",
        "desc": "full 10-tool set, stream=true",
        "body": lambda m: build_body(m, tools=TOOLS_ALL, stream=True),
    },
    {
        "name": "E_quartet_stream_choice_none",
        "desc": "quartet + stream + tool_choice=none -> force text reply",
        "body": lambda m: build_body(m, tools=TOOLS_QUARTET, stream=True,
                                     tool_choice="none",
                                     user="写一句不超过30字的中文问候语。只用文字回答。"),
    },
    {
        "name": "F_quartet_stream_no_call_prompt",
        "desc": "quartet + stream + prompt forbids tool calls",
        "body": lambda m: build_body(m, tools=TOOLS_QUARTET, stream=True,
                                     system="这是一个纯文本回答任务，禁止调用任何工具。",
                                     user="写一句不超过30字的中文问候语。只用文字回答。"),
    },
]


def parse_response(resp, stream):
    """Return (status, content, tool_calls_summary, usage, raw_head).

    Handles both SSE (stream=true) and plain JSON (stream=false).
    """
    ctype = resp.headers.get("Content-Type", "")
    content = ""
    tool_calls = []
    usage = {}

    if stream and "text/event-stream" in ctype:
        for line in resp.iter_lines(decode_unicode=True):
            line = (line or "").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj.get("usage") or {}
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                fn = (tc.get("function") or {}) or {}
                tool_calls.append(fn.get("name") or (fn.get("arguments") or "")[:80])
            if choices[0].get("finish_reason"):
                pass
    else:
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.status_code, "", [], {}, resp.text[:200]
        if data.get("usage"):
            usage = data.get("usage") or {}
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_calls.append(fn.get("name") or "")
    return resp.status_code, content, tool_calls, usage, ""


def run_case(model, case, api_key, timeout=120):
    body = case["body"](model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(_oc_shim_headers())
    url = f"{ZEN_BASE}/chat/completions"
    start = time.time()
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=timeout,
                             stream=body.get("stream", False))
    except requests.RequestException as e:
        return {"error": f"request exception: {e}", "secs": round(time.time() - start, 1)}

    status, content, tool_calls, usage, raw = parse_response(resp, stream=body.get("stream", False))
    return {
        "status": status,
        "content": content.strip(),
        "tool_calls": tool_calls,
        "usage": usage,
        "err_head": resp.text[:200] if status >= 400 else raw[:200],
        "secs": round(time.time() - start, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="zen free model ids to test")
    ap.add_argument("--cases", nargs="+", choices=[c["name"] for c in CASES],
                    default=None, help="run only these cases (default: all)")
    ap.add_argument("--api-key-env", default="ZEN_API_KEY")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[zen-probe] FATAL: {args.api_key_env} unset", file=sys.stderr)
        sys.exit(1)

    cases = [c for c in CASES if args.cases is None or c["name"] in args.cases]

    for model in args.models:
        print(f"\n{'#'*70}\n## MODEL: {model}\n{'#'*70}")
        for case in cases:
            print(f"\n--- case: {case['name']} --- {case['desc']}")
            r = run_case(model, case, api_key)
            print(f"    status: {r.get('status', 'ERR')} | secs: {r.get('secs')}")
            if "error" in r:
                print(f"    ERROR: {r['error']}")
                continue
            if r["status"] >= 400:
                print(f"    err: {r['err_head'][:160]}")
                continue
            print(f"    tool_calls: {r['tool_calls']}")
            print(f"    usage: {r['usage']}")
            print(f"    content: {r['content'][:300]!r}")
            if r["content"]:
                verdict = "TOOL_CALLS" if r["tool_calls"] else "TEXT_OK"
                print(f"    verdict: {verdict}")


if __name__ == "__main__":
    main()