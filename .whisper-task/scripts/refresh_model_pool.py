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
from datetime import datetime, timezone

# Import requests lazily so unit tests that don't need network still load.
import requests

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
POOL_PATH = os.path.join(PROJECT_ROOT, "model-pool.json")

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
    non-alphanumerics so token boundaries stay visible to normalize()."""
    key = normalize(model_id)
    if key in aa_index:
        return aa_index[key]
    # Split raw lowercased id into tokens; try dropping trailing tokens that
    # are free-suffix or a known variant word.
    tokens = re.split(r"[^a-z0-9]+", (model_id or "").lower())
    tokens = [t for t in tokens if t]
    for _ in range(3):
        if not tokens:
            break
        last = tokens[-1]
        if last == "free" or last in _VARIANT_WORDS:
            tokens.pop()
            k = normalize(" ".join(tokens))
            if k in aa_index:
                return aa_index[k]
        else:
            break
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
    """Return slug/name -> AI index map. Requires AA_API_KEY; returns {} if
    missing so callers can choose to skip scoring."""
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
        name = m.get("name") or m.get("slug") or ""
        index[normalize(name)] = float(score)
    return index


# --------------------------------------------------------------------------
# Compile & write
# --------------------------------------------------------------------------

def compile_pool():
    """Fetch free models from both sources, score with AA, keep score>=threshold
    (drop unscored / low), sort descending by score. Raises on any source
    failure so the caller can keep the old file."""
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

    scored.sort(key=lambda t: t[0], reverse=True)
    entries = [e for _, e in scored]
    if not entries:
        raise ValueError("no models passed the AA quality threshold")

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