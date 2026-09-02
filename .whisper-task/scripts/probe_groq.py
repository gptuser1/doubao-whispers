#!/usr/bin/env python3
"""Real-scenario probe for candidate text models.

Tests models with the actual doubao-whispers task: writing a casual Chinese
"碎碎念" post as a specific in-universe character, with their personality and
catchphrase, following project rules (50-200 chars, colloquial, first person,
no future dates, etc). max_tokens is generous because real posts are longer
than an ACK probe.

Models are taken from CLI args. Each is hit with a few realistic case
prompts. Prints the raw output and a rough assessment (length, whether it
honored the catchphrase / personality, whether it stayed Chinese).
"""

import json
import os
import sys
import argparse
import urllib.error
import urllib.request


def chat(base_url, api_key, model, system, user,
         max_tokens=10240, temperature=0.7,
         reasoning_effort=None, reasoning_format="hidden"):
    """Hit Groq chat/completions with the semantic equivalent of the project's
    generate() call. The project expresses 'thinking on/off' via
    enable_thinking; Groq has no such field, so we express the same intent
    with Groq's own reasoning params (reasoning_effort / reasoning_format),
    and keep content clean (no chain-of-thought leaked into the body) via
    reasoning_format="hidden". Payload faithfully mirrors project parameter
    values unless overridden.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Semantic translation of project enable_thinking=True onto Groq's schema.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    payload["reasoning_format"] = reasoning_format

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "doubao-whispers/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode())


CASES = [
    {
        "name": "guga-workday-morning",
        "system": (
            "你是「咕嘎」，大家庭里圆滚滚的企鹅系女孩，呆萌纯粹，有点笨拙，"
            "说话直来直去、想到什么说什么。你的口头禅是「咕咕嘎嘎」（还有各种"
            "咕和嘎的组合，如咕嘎、咕嘎嘎）。说两三句就会带一句口头禅，"
            "口头禅后面带标点，开心用「！」疑惑用「？」陈述用「。」卖萌用「～」。"
        ),
        "user": (
            "今天是工作日早上（周二的 9 点左右），你刚睡醒准备去上学。"
            "请以你的第一人称口吻，发一条 50-200 字的中文朋友圈碎碎念，"
            "口语化、轻松，可以提到早起很困、吃了早餐之类的小事，"
            "要符合你呆萌直率的性格并自然带上口头禅。只输出正文，不要标题"
            "不要解释。日期写成今天 2026-09-02 上午，不要写未来时间。"
        ),
    },
    {
        "name": "doro-weekend-afternoon",
        "system": (
            "你是「Doro」，一个粉色头发的小狗系女孩，温柔粘人、反应慢半拍，"
            "特别喜欢吃欧润吉（橘子），对橘子有执念。说话软软的、慢吞吞、很真诚。"
        ),
        "user": (
            "今天是周日午后（2026-09-06 下午 14 点左右），你宅在家里吃橘子。"
            "请以第一人称发一条 50-200 字的中文朋友圈碎碎念，突出你爱吃欧润吉"
            "和对橘子的执念，语气软软的真诚的。只输出正文，不要标题。"
            "日期写 2026-09-06 下午，不要未来时间。"
        ),
    },
]


def assess(out, case):
    """Rough, honest assessment for a real-scenario Chinese post."""
    notes = []
    n = len(out)
    if n < 50:
        notes.append(f"偏短({n}字,要求50-200)")
    if n > 200:
        notes.append(f"偏长({n}字)")
    if not any('\u4e00' <= c <= '\u9fff' for c in out):
        notes.append("几乎无中文字符")
    if "咕" in out or "嘎" in out:
        notes.append("含口头禅关键词")
    if "未来" not in system_hint(case) and "2026-09-0" not in out:
        pass
    return "; ".join(notes) if notes else "OK"


def system_hint(case):
    return case["system"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="model ids (each: host|model|KEY_ENV)")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1",
                    help="base url for the provider (default Groq)")
    ap.add_argument("--api-key-env", default="GROQ_API_KEY",
                    help="env var holding the API key")
    ap.add_argument("--max-tokens", type=int, default=10240,
                    help="match project whisper-generation max_tokens")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="match project whisper-generation temperature")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high", "default"],
                    help="Groq semantic equivalent of project enable_thinking=True")
    ap.add_argument("--reasoning-format", default="hidden",
                    choices=["raw", "parsed", "hidden"],
                    help="keep content clean: hidden = reasoning not leaked into body")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[probe] missing {args.api_key_env}", file=sys.stderr)
        sys.exit(1)

    for model in args.models:
        print(f"\n{'='*64}\nMODEL: {model}  ({args.base_url})\n{'='*64}")
        for case in CASES:
            print(f"\n--- case: {case['name']} ---")
            try:
                status, data = chat(
                    args.base_url, api_key, model,
                    case["system"], case["user"],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_format=args.reasoning_format,
                )
            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code}: {e.read().decode()[:300]}")
                continue
            except Exception as e:
                print(f"ERROR: {repr(e)}")
                continue
            usage = data.get("usage") or {}
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
            content = (data.get("choices") or [{}])[0].get("message") or {}
            out = (content.get("content") or "").strip()
            print(f"HTTP {status} | finish={finish} | "
                  f"ptok={usage.get('prompt_tokens')} ctok={usage.get('completion_tokens')}")
            print("---- output ----")
            print(out)
            print("---- assess ----")
            print(assess(out, case))


if __name__ == "__main__":
    main()