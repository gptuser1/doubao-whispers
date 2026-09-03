#!/usr/bin/env python3
"""Refresh the free-model pool from OpenRouter + opencode zen, sorted by the
Artificial Analysis AI index, and write it to model-pool.json at repo root.

Run as a step inside whisper_runner.py (not a separate cron): heartbeat reads
model-pool.json's updated_at and triggers a refresh only when it is stale
(>= REFRESH_INTERVAL). On any failure the existing file is kept untouched so a
broken pool never ships.

This mirrors the logic previously hosted in the kbox model-pool service —
moved here so no external service is involved and the consumer reads a local,
versioned list from the repo.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# Import requests lazily so unit tests that don't need network still load.
import requests

# Paths. model-pool.json lives next to config.json (both under .whisper-task/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
POOL_PATH = os.path.join(PROJECT_ROOT, ".whisper-task", "model-pool.json")

# How often (seconds) before the pool is considered stale and refreshed.
REFRESH_INTERVAL = 12 * 3600  # 12h

# Only models with an AA index >= this threshold are kept.
POOL_MIN_SCORE = 35

FETCH_TIMEOUT = 30

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE}/models?limit=500"

ZEN_BASE = "https://opencode.ai/zen/v1"
ZEN_MODELS_URL = f"{ZEN_BASE}/models"

AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# baseurl (host) -> api key env var name, same map as ai_client.BASEURL_KEY_ENV.
BASEURL_KEY_ENV = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    "opencode.ai": "ZEN_API_KEY",
}


def _now():
    return datetime.now(timezone.utc)


def _read_pool():
    """Return the current pool dict {'updated_at': ms, 'entries': [...]}, or
    None if the file is missing or unreadable/invalid."""
    if not os.path.exists(POOL_PATH):
        return None
    try:
        with open(POOL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except Exception:
        return None
    return None


def is_stale(now=None):
    """True when the pool is missing or older than REFRESH_INTERVAL."""
    now = now or _now()
    pool = _read_pool()
    if pool is None:
        return True
    updated = pool.get("updated_at")
    if not isinstance(updated, (int, float)):
        return True
    age = now - datetime.fromtimestamp(updated / 1000, tz=timezone.utc)
    return age.total_seconds() >= REFRESH_INTERVAL


def _fetch_json(url, headers=None, timeout=FETCH_TIMEOUT):
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# Normalization & AA scoring
# --------------------------------------------------------------------------

def normalize(name):
    """Loose key for matching provider ids to AA names/slugs."""
    s = (name or "").lower()
    s = re.sub(r":free$", "", s)
    s = re.sub(r"-free$", "", s)
    s = re.sub(r":[a-z]+$", "", s)
    s = re.sub(r"^[a-z0-9-]+/", "", s)  # provider prefix "google/"
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s.strip()


# Trailing variant tokens (contributor/fin/it/...) that we strip layer by layer
# on a miss, so "muse-spark-1.2-contributor" matches "Muse Spark v1.2"
# "inclusion-ai/ling-3.0-flash-fin" matches the AA "fin" row even if AA names
# it slightly differently.
_VARIANT_WORDS = {
    "contributor", "fin", "it", "code", "preview", "reasoning", "thinking",
    "instruct", "chat", "nano", "mini", "small", "fini", "tt", "beta",
}


def aa_score(model_id, aa_index):
    """Return the AA index for a provider model id, or None if unmatched.

    Tries exact normalized match, then strips trailing variant/free tokens one
    at a time (max 3 layers) before giving up. Works on the raw id split by
    non-alphanumerics so token boundaries stay visible to normalize().
    Provider prefixes ("inclusionai/") are tried both included and dropped, so
    strip matching is done on the "/"-suffix too."""
    def try_key(key):
        if key in aa_index:
            return aa_index[key]
        return None

    v = try_key(normalize(model_id))
    if v is not None:
        return v

    # Token-strip candidates: the whole id, and (if it has a provider prefix)
    # just the part after the last "/".
    id_lower = (model_id or "").lower()
    candidates = [id_lower]
    if "/" in id_lower:
        candidates.append(id_lower.split("/")[-1])

    for base in candidates:
        tokens = [t for t in re.split(r"[^a-z0-9]+", base) if t]
        for _ in range(3):
            if not tokens:
                break
            last = tokens[-1]
            if not (last == "free" or last in _VARIANT_WORDS):
                break
            tokens.pop()
            v = try_key(normalize(" ".join(tokens)))
            if v is not None:
                return v
    return None


# --------------------------------------------------------------------------
# Source fetchers
# --------------------------------------------------------------------------

def fetch_openrouter_free():
    """OpenRouter models with ':free' suffix, both prices 0."""
    data = _fetch_json(OPENROUTER_MODELS_URL)
    out = []
    for m in data.get("data") or []:
        mid = m.get("id") or ""
        if not mid.endswith(":free"):
            continue
        pricing = m.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt", 1) or 1)
            completion = float(pricing.get("completion", 1) or 1)
        except (TypeError, ValueError):
            continue
        if prompt == 0 and completion == 0:
            out.append({"model": mid, "baseurl": OPENROUTER_BASE})
    return out


def fetch_zen_free():
    """opencode zen models whose id ends with '-free'."""
    data = _fetch_json(ZEN_MODELS_URL)
    out = []
    for m in data.get("data") or []:
        mid = m.get("id") or ""
        if mid.endswith("-free"):
            out.append({"model": mid, "baseurl": ZEN_BASE})
    return out


def build_aa_index():
    """Return normalized slug -> AI index map (slug only). Requires AA_API_KEY;
    raises if missing so callers can keep the old file."""
    api_key = os.environ.get("AA_API_KEY", "").strip()
    if not api_key:
        print("[model-pool] AA_API_KEY unset, cannot score models", file=sys.stderr)
        raise ValueError("AA_API_KEY required to refresh model pool")
    data = _fetch_json(AA_MODELS_URL, headers={"x-api-key": api_key})
    index = {}
    for m in data.get("data") or []:
        try:
            score = m["evaluations"]["artificial_analysis_intelligence_index"]
        except (KeyError, TypeError):
            continue
        if not isinstance(score, (int, float)):
            continue
        score = float(score)
        # Index the SLUG only: AA slugs are clean, stable identifiers
        # ("deepseek-v4-flash"), while names carry noisy suffixes like
        # "DeepSeek V4 Flash 0731 (Reasoning, Max Effort)" that never match a
        # short provider id. Slug-only matching is both accurate and enough
        # once aa_score also strips variant/free tokens on a miss.
        if m.get("slug"):
            index.setdefault(normalize(m["slug"]), score)
    return index


# --------------------------------------------------------------------------
# Liveness probe (ack test)
# --------------------------------------------------------------------------

ACK_TIMEOUT = 15
ACK_PROMPT = "Reply with exactly: ack"
ACK_MAX_ATTEMPTS = 3
ACK_BASE_DELAY = 8  # seconds; backoff = base * attempt (8s, 16s)


def _extract_host(baseurl):
    try:
        from urllib.parse import urlparse
        return urlparse(baseurl).netloc.lower()
    except Exception:
        return ""


def ack_probe(entry, timeout=ACK_TIMEOUT, max_attempts=ACK_MAX_ATTEMPTS):
    """Send a minimal request and require a usable reply, with retry.

    Probes /chat/completions first; on failure falls back to the Responses
    API (/responses) so models served on either endpoint pass. Liveness is
    judged by the HTTP status code (200 = alive); truncated output does not
    matter. Retries only cover transient noise (network error, 429, 5xx); a
    deterministic 4xx (auth 401/403, missing model 404) fails immediately.
    Missing provider key also fails so dead models never enter the pool.
    Returns (ok, detail).
    """
    model = entry.get("model", "")
    baseurl = (entry.get("baseurl") or "").rstrip("/")
    key_env = BASEURL_KEY_ENV.get(_extract_host(entry.get("baseurl") or ""), "")
    api_key = os.environ.get(key_env, "").strip() if key_env else ""
    if not api_key:
        return False, f"no {key_env or 'api key'} configured"

    headers = {"Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in baseurl:
        headers["HTTP-Referer"] = "https://github.com/doubao-whispers"
        headers["X-OpenRouter-Title"] = "doubao-whispers"
        headers["X-OpenRouter-Categories"] = "cli-agent,personal-agent"

    def attempt(endpoint, body):
        """One endpoint's ack loop; retry semantics mirror the original."""
        url = f"{baseurl}/{endpoint}"
        for i in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            except Exception as e:
                if i < max_attempts:
                    time.sleep(ACK_BASE_DELAY * i)
                    continue
                return False, f"request error: {e}"

            if resp.status_code == 200:
                return True, f"http {resp.status_code}"

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Transient — retry with backoff before blaming the model.
                if i < max_attempts:
                    time.sleep(ACK_BASE_DELAY * i)
                    continue
                return False, f"http {resp.status_code}: {resp.text[:120]}"

            # Deterministic 4xx (auth / missing model / bad request) — no retry.
            return False, f"http {resp.status_code}: {resp.text[:120]}"
        return False, "max attempts exhausted"

    ok, detail = attempt(
        "chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": ACK_PROMPT}],
            "temperature": 0,
        },
    )
    if ok:
        return True, detail

    print("[model-pool] chat/completions ack failed; trying /responses",
          file=sys.stderr)
    ok, rdetail = attempt(
        "responses",
        {
            "model": model,
            "input": ACK_PROMPT,
        },
    )
    if ok:
        return True, rdetail
    return False, f"chat/completions {detail}; /responses {rdetail}"


# --------------------------------------------------------------------------
# Compile & write
# --------------------------------------------------------------------------

def compile_pool():
    """Fetch free models from both sources, score with AA (slug match, keep
    score>=threshold), then pass each surviving candidate a liveness ACK test
    before it may enter the pool. Sort descending by score. Raises on any
    source failure so the caller can keep the old file."""
    sources = []
    sources += fetch_openrouter_free()
    sources += fetch_zen_free()
    if not sources:
        raise ValueError("no free models found from any source")

    aa_index = build_aa_index()

    scored = []  # (score, entry)
    for entry in sources:
        score = aa_score(entry["model"], aa_index)
        if score is None or score < POOL_MIN_SCORE:
            continue
        scored.append((score, entry))

    # Liveness gate: every quality-passed candidate must answer an ACK probe.
    # A dead endpoint, missing provider key, auth failure, or empty reply drops
    # the model here. Fails fast individually; one bad model never blocks the rest.
    alive = []
    dropped = 0
    for score, entry in scored:
        ok, detail = ack_probe(entry)
        if ok:
            alive.append((score, entry))
        else:
            dropped += 1
            print(f"[model-pool] ack failed, dropping {entry['model']}: {detail}",
                  file=sys.stderr)

    alive.sort(key=lambda t: t[0], reverse=True)
    entries = [e for _, e in alive]
    if not entries:
        raise ValueError("no models passed AA threshold and ACK liveness probe")

    return {
        "updated_at": int(_now().timestamp() * 1000),
        "entries": entries,
    }


def write_pool(pool, path=None):
    path = path or POOL_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    return path


def refresh(force=False):
    """Refresh and write the pool. Returns True if the file was rewritten, or
    False if nothing was done. On failure the existing file is preserved and
    the exception propagates to the caller (which may keep the old file and
    continue the whisper run)."""
    if not force and not is_stale():
        return False
    pool = compile_pool()
    write_pool(pool)
    return True


if __name__ == "__main__":
    if refresh(force=("--force" in sys.argv)):
        print(f"[model-pool] refreshed -> {POOL_PATH}")
    else:
        print("[model-pool] pool is fresh, nothing to do")