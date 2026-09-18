#!/usr/bin/env python3
"""Probe zen free-tier gates (2026-09 change).

Two hypotheses are tested side by side:

H1 — "within OpenCode" fingerprint. Zen rejects requests whose headers do
     not look like the real opencode CLI. The real headers (from
     sst/opencode @ 1.18.31 source) are:

       User-Agent:            opencode/{channel}/{version}/cli
                              (channel = npm dist-tag "latest" for stable)
       x-opencode-session:    ses_ + 26 chars (12 hex of descending(time<<12+counter)
                              + 14 base62 random)   — NOT ses_+hex-only, NOT a UUID
       x-opencode-request:    msg_ + 26 chars (same 26-char shape, ascending)
       x-opencode-client:     cli
       x-opencode-project:    <project id> (only sent for opencode provider)

H2 — anonymous sentinel. The opencode CLI itself sends
     `Authorization: Bearer public` when no key is configured, and zen
     documents 6 models with anonymous access (mimo-v2.5-free,
     ling-3.0-flash-fin-free, nemotron-3-ultra-free,
     nemotron-3.5-lightning-free, muse-spark-1.3-contributor-free, big-pickle).
     A real ZEN_API_KEY + wrong fingerprint currently yields
     FreeTierError too (Z0 control proved auth isn't the gate).

Each case also picks the right WIRE per model: zen routes chat-protocol
models to /chat/completions and responses-protocol models (Muse family)
to /responses. Hitting the wrong wire yields 400/500 noise, not a
FreeTier verdict.

Usage:
    python probe_zen_fingerprint.py --models ling-3.0-flash-fin-free mimo-v2.5-free
"""
import argparse
import json
import os
import sys
import time

import requests

from ai_client import _oc_shim_headers

ZEN_BASE = "https://opencode.ai/zen/v1"
OC_VERSION = "1.18.31"
OC_CHANNEL = "latest"

# Model -> wire protocol (from zen docs / @namzu/zen catalogue).
RESPONSES_MODELS = {
    "muse-spark-1.3-contributor-free",
    "muse-spark-1.2-contributor-free",
}
ANONYMOUS_MODELS = {
    "mimo-v2.5-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.3-contributor-free",
    "muse-spark-1.2-contributor-free",
    "big-pickle",
}


# --- opencode tool signatures -------------------------------------------------

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


# --- ID generation from opencode source ---------------------------------------

_ID_LEN = 26
_ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_ID_HEX = "0123456789abcdef"


def _oc_id(prefix, descending=True):
    """Reproduce opencode's identifier.ts create(): 12 hex chars of
    (timestamp<<12 | counter) [bitwise-NOT when descending] + 14 random
    base62 chars."""
    ts_ms = int(time.time() * 1000)
    counter = 1
    current = ts_ms * 0x1000 + counter
    if descending:
        current = ~current
    value = current & ((1 << 64) - 1)  # keep 64 bits; only low 48 are used
    time_hex = "".join(
        f"{(value >> (40 - 8 * i)) & 0xFF:02x}" for i in range(6)
    )
    rand = "".join(_ID_CHARS[os.urandom(1)[0] % 62] for _ in range(_ID_LEN - 12))
    return f"{prefix}{time_hex}{rand}"


def oc_headers(project="global"):
    """Exactly what opencode 1.18.31 sends for an opencode provider call."""
    return {
        "User-Agent": f"opencode/{OC_CHANNEL}/{OC_VERSION}/cli",
        "x-opencode-client": "cli",
        "x-opencode-project": project,
        "x-opencode-request": _oc_id("msg_", descending=False),
        "x-opencode-session": _oc_id("ses_", descending=True),
    }


# --- bodies -------------------------------------------------------------------

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


def build_responses_body(model, tools=None, stream=True,
                         instructions="这是一个纯文本回答任务，禁止调用任何工具。",
                         user="只用文字回答：ack"):
    """OpenAI Responses wire format (zen /v1/responses, Muse family).

    Uses `instructions` + `input` instead of `messages`, and function tools
    without the chat-style nested `function` wrapper.
    """
    body = {
        "model": model,
        "instructions": instructions,
        "input": user,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = [
            {
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]
    return body


# --- cases --------------------------------------------------------------------
# Each entry: (name, desc, auth_mode, headers_mode, wire, body)
#   auth_mode: "key" | "public" | "none"
#   headers_mode: "real" | "old"  (old = current ai_client shim)
#   wire: "chat" | "responses"

def cases_for(model):
    if model in RESPONSES_MODELS:
        # Muse family is responses-wire only (chat wire returns 500).
        return [
            ("G07_real_public_responses",
             "responses wire + real headers + Bearer public + quartet + stream",
             "public", "real", "responses",
             build_responses_body(model, tools=TOOLS_QUARTET, stream=True)),
            ("G08_real_key_responses",
             "responses wire + real headers + ZEN key + quartet + stream",
             "key", "real", "responses",
             build_responses_body(model, tools=TOOLS_QUARTET, stream=True)),
            ("G09_real_public_responses_notools",
             "responses wire + real headers + Bearer public + NO tools + stream",
             "public", "real", "responses",
             build_responses_body(model, tools=None, stream=True)),
        ]
    cases = [
        ("G01_real_key_chat_quartet_stream",
         "real headers + ZEN key + quartet + stream (H1 on chat wire)",
         "key", "real", "chat", build_body(model, tools=TOOLS_QUARTET, stream=True)),
        ("G02_real_public_chat_quartet_stream",
         "real headers + Bearer public + quartet + stream (H2, chat wire)",
         "public", "real", "chat", build_body(model, tools=TOOLS_QUARTET, stream=True)),
        ("G03_real_public_chat_notools_stream",
         "real headers + Bearer public + NO tools + stream (is tool gate still needed?)",
         "public", "real", "chat", build_body(model, tools=None, stream=True)),
        ("G04_real_public_chat_notools_nostream",
         "real headers + Bearer public + NO tools + stream:false (old ai_client payload)",
         "public", "real", "chat", build_body(model, tools=None, stream=False)),
        ("G05_old_headers_key_chat_quartet_stream",
         "OLD shim headers + ZEN key + quartet + stream (must still fail = control)",
         "key", "old", "chat", build_body(model, tools=TOOLS_QUARTET, stream=True)),
        ("G06_real_key_chat_quartet_nostream",
         "real headers + ZEN key + quartet + stream:false (stream gate?)",
         "key", "real", "chat", build_body(model, tools=TOOLS_QUARTET, stream=False)),
    ]
    if wire_auto == "responses":
        # Muse family routes to /v1/responses with the responses wire format.
        cases.append(("G07_real_public_responses",
                      "real headers + Bearer public + quartet + stream on /responses",
                      "public", "real", "responses",
                      build_body(model, tools=TOOLS_QUARTET, stream=True)))
        cases.append(("G08_real_key_responses",
                      "real headers + ZEN key + quartet + stream on /responses",
                      "key", "real", "responses",
                      build_body(model, tools=TOOLS_QUARTET, stream=True)))
    return cases


def parse_response(resp, stream, wire="chat"):
    ctype = resp.headers.get("Content-Type", "")

    # OpenAI Responses wire: events like response.output_text.delta.
    if wire == "responses":
        content = ""
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
                if obj.get("type") == "response.output_text.delta":
                    content += obj.get("delta") or ""
                if obj.get("type") == "response.completed":
                    item = (obj.get("response") or {}).get("output") or []
                    for it in item:
                        if it.get("type") == "function_call":
                            return resp.status_code, content, [it.get("name", "")], usage, ""
        else:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return resp.status_code, "", [], {}, resp.text[:200]
            if data.get("usage"):
                usage = data.get("usage") or {}
            for it in data.get("output") or []:
                if it.get("type") == "message":
                    for c in it.get("content") or []:
                        if c.get("type") == "output_text":
                            content += c.get("text") or ""
        return resp.status_code, content, [], usage, ""

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
            msg = choices[0].get("delta") or choices[0].get("message") or {}
            content += msg.get("content") or ""
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}) or {}
                tool_calls.append(fn.get("name") or (fn.get("arguments") or "")[:80])
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


def run_case(model, apikey, headers_mode, wire, body, timeout=120):
    if headers_mode == "real":
        headers = oc_headers()
    else:
        headers = _oc_shim_headers()
    headers["Content-Type"] = "application/json"
    if apikey == "public":
        headers["Authorization"] = "Bearer public"
    elif apikey == "key":
        headers["Authorization"] = f"Bearer {apikey}"

    url = f"{ZEN_BASE}/chat/completions" if wire == "chat" else f"{ZEN_BASE}/responses"
    start = time.time()
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=timeout,
                             stream=body.get("stream", False))
    except requests.RequestException as e:
        return {"error": f"request exception: {e}", "secs": round(time.time() - start, 1)}

    status, content, tool_calls, usage, raw = parse_response(resp, stream=body.get("stream", False),
                                                             wire=wire)
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
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--api-key-env", default="ZEN_API_KEY")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    print(f"[zen-probe] {args.api_key_env}: {'set (' + api_key[:6] + '...)' if api_key else 'UNSET'}")

    for model in args.models:
        print(f"\n{'#'*70}\n## MODEL: {model}  (wire={'responses' if model in RESPONSES_MODELS else 'chat'}"
              f", anonymous={'yes' if model in ANONYMOUS_MODELS else 'no'})\n{'#'*70}")
        for name, desc, auth_mode, headers_mode, wire, body in cases_for(model):
            apikey = "public" if auth_mode == "public" else ("none" if auth_mode == "none" else api_key)
            print(f"\n--- case: {name} --- {desc}")
            r = run_case(model, apikey, headers_mode, wire, body)
            print(f"    status: {r.get('status', 'ERR')} | secs: {r.get('secs')} | wire: {wire} | auth: {auth_mode}")
            if "error" in r:
                print(f"    ERROR: {r['error']}")
                continue
            if r["status"] >= 400:
                print(f"    err: {r['err_head'][:160]}")
                continue
            print(f"    tool_calls: {r['tool_calls']}")
            print(f"    usage: {r['usage']}")
            print(f"    content: {r['content'][:300]!r}")
            if r["content"] or r["tool_calls"]:
                verdict = "TOOL_CALLS" if r["tool_calls"] else "TEXT_OK"
                print(f"    verdict: {verdict}")


if __name__ == "__main__":
    main()