#!/usr/bin/env python3
"""
AI client with adapter pattern.
Supports multiple providers for text and image generation.

Providers:
  - workers_ai: Cloudflare Workers AI
  - openai: OpenAI-compatible APIs (DeepSeek, Moonshot, GLM, etc.)

Usage:
    from ai_client import create_text_provider, create_image_provider

    text_provider = create_text_provider(config)
    response = text_provider.generate(messages=[...])

    image_provider = create_image_provider(config)
    image_path = image_provider.generate(prompt="...")
"""

import json
import os
import sys
import time
import base64
import requests
from abc import ABC, abstractmethod


# Default User-Agent — custom string avoids Cloudflare bot detection (1010)
# that blocks the default urllib/requests User-Agents on some endpoints.
_DEFAULT_UA = "doubao-whispers/1.0"


# ==================== Retry Helpers ====================

# Keywords that indicate a rate-limit / quota-exceeded response
_RATE_LIMIT_KEYWORDS = (
    "rate limit", "too many requests", "rate_limit",
    "频率", "过多", "频繁", "超出", "quota", "throttl",
)


def _should_retry(status_code=None, body=""):
    """Check if an error response warrants a retry."""
    # Rate limiting
    if status_code == 429:
        return True
    # Transient server errors (5xx)
    if status_code and 500 <= status_code < 600:
        return True
    # Rate-limit keywords in body
    body_lower = body.lower()
    return any(kw in body_lower for kw in _RATE_LIMIT_KEYWORDS)


def _retry_sleep(attempt, reason, base_delay=2):
    """Sleep before a retry with exponential backoff, logging the reason."""
    delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
    print(f"[AI retry] {reason}, retrying in {delay}s "
          f"(attempt {attempt + 1}/3)...", file=sys.stderr)
    time.sleep(delay)


# ==================== Text Providers ====================

class TextProvider(ABC):
    """Abstract base class for text generation."""

    @abstractmethod
    def generate(self, messages, max_tokens=1024, temperature=0.8, enable_thinking=False):
        """
        Generate text from chat messages.

        Args:
            messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
            max_tokens: max tokens to generate
            temperature: sampling temperature
            enable_thinking: whether to enable the model's thinking/reasoning
                mode. Defaults to False — callers must explicitly opt in.
                Providers that don't support thinking ignore this flag.

        Returns:
            str: generated text
        """
        pass


class WorkersAIText(TextProvider):
    """Cloudflare Workers AI text provider."""

    def __init__(self, config):
        self.model = config.get("model", "@cf/zai-org/glm-4.7-flash")
        # Credentials support both inline values (preferred, from a pool JSON
        # env var) and a fallback to discrete env vars by name.
        self.account_id = (config.get("account_id")
                           or os.environ.get(config.get("account_id_env", "CF_DEFAULT_ACCOUNT_ID"), ""))
        self.api_token = (config.get("api_token")
                          or os.environ.get(config.get("api_token_env", "CF_DEFAULT_API_TOKEN"), ""))

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires account_id/api_token (inline or CF_DEFAULT_ACCOUNT_ID/CF_DEFAULT_API_TOKEN env)")

    def generate(self, messages, max_tokens=1024, temperature=0.8, enable_thinking=False):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": { "enable_thinking": enable_thinking }
        }

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "User-Agent": _DEFAULT_UA,
        }

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=600)
            except requests.RequestException as e:
                if attempt < max_retries:
                    _retry_sleep(attempt, f"URL error: {e}")
                    continue
                raise RuntimeError(f"WorkersAI request failed: {e}")

            if resp.status_code >= 400:
                err_body = resp.text
                if _should_retry(status_code=resp.status_code, body=err_body) and attempt < max_retries:
                    _retry_sleep(attempt, f"HTTP {resp.status_code}")
                    continue
                raise RuntimeError(f"WorkersAI request failed: HTTP {resp.status_code} {err_body[:200]}")

            result = resp.json()
            if result.get("success"):
                return result.get("result", {}).get("response", "").strip()
            else:
                errors = result.get("errors", [])
                err_msg = errors[0].get("message", "unknown error") if errors else "unknown error"
                err_code = errors[0].get("code", 0) if errors else 0
                if _should_retry(status_code=err_code, body=err_msg) and attempt < max_retries:
                    _retry_sleep(attempt, f"WorkersAI rate limit: {err_msg[:80]}")
                    continue
                raise RuntimeError(f"WorkersAI error: {err_msg}")


class OpenAIText(TextProvider):
    """OpenAI-compatible text provider (works with DeepSeek, Moonshot, GLM, etc.)."""

    def __init__(self, config):
        self.model = config.get("model", "deepseek-chat")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        # Inline api_key is preferred (pool JSON env var); falls back to a
        # discrete env var named by api_key_env if inline is absent.
        self.api_key = (config.get("api_key")
                        or os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), ""))

        if not self.api_key:
            raise ValueError("OpenAI provider requires api_key (inline or OPENAI_API_KEY env)")

        # Token usage tracking: last_usage = most recent call, usage_total = accumulated across all calls
        self.last_usage = None
        self.usage_total = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0}

    @property
    def _err_ctx(self):
        """Short provider identity to prepend to error messages."""
        return f"OpenAI[{self.model}@{self.base_url}]"

    def generate(self, messages, max_tokens=1024, temperature=0.8, enable_thinking=False):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "thinking": {"type": "enabled" if enable_thinking else "disabled"},
            "enable_thinking": enable_thinking,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": _DEFAULT_UA,
        }

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=600)
            except requests.RequestException as e:
                if attempt < max_retries:
                    _retry_sleep(attempt, f"URL error: {e}")
                    continue
                raise RuntimeError(f"{self._err_ctx} request failed: {e}")

            if resp.status_code >= 400:
                err_body = resp.text
                if _should_retry(status_code=resp.status_code, body=err_body) and attempt < max_retries:
                    _retry_sleep(attempt, f"HTTP {resp.status_code}")
                    continue
                raise RuntimeError(f"{self._err_ctx} request failed: HTTP {resp.status_code} {err_body[:200]}")

            result = resp.json()

            # Some APIs return 200 + error body for rate limiting
            if result.get("error"):
                err_str = str(result["error"])
                if _should_retry(body=err_str) and attempt < max_retries:
                    _retry_sleep(attempt, f"Rate limited: {err_str[:80]}")
                    continue
                raise RuntimeError(f"{self._err_ctx} error: {result['error']}")

            # Log token usage from API response (DeepSeek/SiliconFlow return cache stats too)
            usage = result.get("usage") or {}
            if usage:
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", 0)
                cache_hit = usage.get("prompt_cache_hit_tokens", 0)
                cache_miss = usage.get("prompt_cache_miss_tokens", 0)
                self.last_usage = {"prompt": prompt, "completion": completion,
                                   "total": total, "cache_hit": cache_hit}
                self.usage_total["prompt"] += prompt
                self.usage_total["completion"] += completion
                self.usage_total["total"] += total
                self.usage_total["cache_hit"] += cache_hit
                cache_note = ""
                if cache_hit or cache_miss:
                    cache_note = f" (cache hit={cache_hit}, miss={cache_miss})"
                print(f"[AI usage] model={self.model} prompt={prompt} "
                      f"completion={completion} total={total}{cache_note}",
                      file=sys.stderr)

            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


# ==================== Text Pool (fallback chain) ====================

class FallbackTextProvider(TextProvider):
    """Try an ordered list of text providers, returning the first success.

    Each candidate is tried in order. A candidate is skipped when its call
    raises (network/API error, model unavailable, rate limit) OR returns empty
    / whitespace content. Fails only when every candidate fails.

    Usage/error context of the winning model is surfaced via ``last_identity``
    and accumulated token usage is aggregated across the underlying providers
    that have a ``usage_total`` attribute.
    """

    def __init__(self, providers, name="pool"):
        self.providers = list(providers)
        self.name = name
        if not self.providers:
            raise ValueError("FallbackTextProvider requires at least one provider")
        self.last_identity = ""
        self.last_usage = None
        self.usage_total = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0}

    def _model_label(self, provider):
        return getattr(provider, "model", "?")

    def generate(self, messages, max_tokens=1024, temperature=0.8, enable_thinking=False):
        errors = []
        for provider in self.providers:
            label = self._model_label(provider)
            try:
                out = provider.generate(
                    messages, max_tokens=max_tokens,
                    temperature=temperature, enable_thinking=enable_thinking,
                )
            except Exception as e:
                errors.append(f"{label}: {e}")
                print(f"[pool:{self.name}] {label} failed, trying next: {e}",
                      file=sys.stderr)
                self._absorb_usage(provider)
                continue

            if out and out.strip():
                self.last_identity = f"{label}@{getattr(provider, 'base_url', '?')}"
                self.last_usage = getattr(provider, "last_usage", None)
                self._absorb_usage(provider)
                return out.strip()

            errors.append(f"{label}: empty output")
            print(f"[pool:{self.name}] {label} returned empty content, trying next",
                  file=sys.stderr)
            self._absorb_usage(provider)

        raise RuntimeError(
            f"All text providers in pool[{self.name}] failed: " + " | ".join(errors)
        )

    def _absorb_usage(self, provider):
        """Merge a provider's usage into the pool aggregate."""
        provider_usage = getattr(provider, "usage_total", None)
        if not provider_usage:
            return
        for key in self.usage_total:
            self.usage_total[key] += provider_usage.get(key, 0)

    def __str__(self):
        return f"FallbackTextProvider[{self.name}]({' -> '.join(self._model_label(p) for p in self.providers)})"


# ==================== Image Providers ====================

class ImageProvider(ABC):
    """Abstract base class for image generation."""

    @abstractmethod
    def generate(self, prompt, output_path, reference_images=None, size="landscape_4_3"):
        """
        Generate an image from a prompt.

        Args:
            prompt: text description of the image
            output_path: where to save the generated image
            reference_images: list of paths to reference images (for character consistency)
            size: image size hint

        Returns:
            str: path to the generated image, or None if failed
        """
        pass


class WorkersAIImage(ImageProvider):
    """Cloudflare Workers AI image provider.

    Uses multipart/form-data (required by flux-2 models) and supports up to
    4 reference images for character consistency. Output is 1024x768.
    """

    def __init__(self, config):
        self.model = config.get("model", "@cf/black-forest-labs/flux-2-klein-4b")
        # Inline credentials preferred (from a pool JSON env var), else env by name.
        self.account_id = (config.get("account_id")
                           or os.environ.get(config.get("account_id_env", "CF_IMAGE_ACCOUNT_ID"), ""))
        self.api_token = (config.get("api_token")
                          or os.environ.get(config.get("api_token_env", "CF_IMAGE_API_TOKEN"), ""))

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires account_id/api_token (inline or CF_IMAGE_ACCOUNT_ID/CF_IMAGE_API_TOKEN env)")

    def generate(self, prompt, output_path, reference_images=None, size="landscape_4_3"):
        """Generate an image.

        Args:
            prompt: text description of the image
            output_path: where to save the generated image
            reference_images: list of reference images (max 4). Each item can
                be either a file path (str) or a (name, png_bytes) tuple with
                pre-processed PNG data. CF flux-2 models use these for
                character/style consistency. Images should be <=512x512;
                larger images are downscaled automatically by _prepare_reference.
            size: ignored for flux-2 (always 1024x768), kept for interface compat

        Returns:
            str: output_path on success, None on failure.
            Raises RuntimeError with .flagged=True attribute if blocked by
            CF safety filter (code 3030), so callers can retry with rephrasing.
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "User-Agent": _DEFAULT_UA,
        }

        # Form fields
        data = {
            "prompt": prompt,
            "width": "1024",
            "height": "768",
        }

        # Optional reference images (max 4). CF flux-2 accepts up to 4 512x512 tiles.
        files = []
        if reference_images:
            refs = reference_images[:4]
            for idx, ref in enumerate(refs):
                try:
                    if isinstance(ref, tuple):
                        # Pre-processed: (name, png_bytes)
                        ref_bytes = ref[1]
                    else:
                        # File path: load and process on the fly
                        ref_bytes = _prepare_reference_image(ref)
                    if ref_bytes:
                        # CF flux-2 expects reference images as input_image_0,
                        # input_image_1, ... (0-indexed, up to 4). This matches
                        # the see-u-say playground frontend, which is verified
                        # to produce good character-consistency results. The
                        # previous "image"/"image[]" field names were NOT
                        # recognized by CF as reference inputs, so the model
                        # was effectively doing pure text-to-image.
                        field_name = f"input_image_{idx}"
                        files.append((field_name, (f"ref_{idx}.png", ref_bytes, "image/png")))
                except Exception as e:
                    print(f"[image] Skipping reference {ref}: {e}", file=sys.stderr)

        try:
            resp = requests.post(url, data=data, files=files, headers=headers, timeout=600)
        except requests.RequestException as e:
            err = RuntimeError(f"CF image gen request failed: {e}")
            err.flagged = False
            raise err

        try:
            result = resp.json()
        except ValueError:
            err = RuntimeError(f"CF image gen HTTP {resp.status_code}: {resp.text[:200]}")
            err.flagged = False
            raise err

        # flux-2 output schema: {"image": "<base64>", ...}  (no success/errors wrapper)
        image_b64 = result.get("image", "")
        if image_b64:
            image_data = base64.b64decode(image_b64)
            with open(output_path, "wb") as f:
                f.write(image_data)
            return output_path

        # Fallback: legacy {success, result, errors} schema
        if result.get("success"):
            image_b64 = result.get("result", {}).get("image", "")
            if image_b64:
                image_data = base64.b64decode(image_b64)
                with open(output_path, "wb") as f:
                    f.write(image_data)
                return output_path

        errors = result.get("errors", [])
        err_msg = errors[0].get("message", "unknown") if errors else "no image in response"
        err_code = errors[0].get("code", 0) if errors else 0
        err = RuntimeError(f"CF image gen failed: {err_msg}")
        err.flagged = (err_code == 3030 or "flagged" in err_msg.lower())
        raise err


def _prepare_reference_image(path, max_size=512):
    """Load an image file and return PNG bytes sized to fit within max_size x max_size.

    Used to prepare character avatar reference images for CF flux-2 models,
    which accept up to 4 512x512 tiles. Images smaller than 512x512 are kept
    as-is (not upscaled) per project convention.

    Returns PNG bytes, or None if Pillow is unavailable or the image can't be read.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(path)
        # Composite onto a white background before dropping alpha. This mirrors
        # the see-u-say playground frontend (canvas white fill + drawImage) so
        # transparent reference images are handled identically. Without this,
        # PIL's convert("RGB") turns transparent regions black, which confuses
        # the model about the character's appearance.
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        # Downscale only if larger than max_size; never upscale
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[image] Failed to prepare reference {path}: {e}", file=sys.stderr)
        return None


# ==================== Usage Helpers ====================

def merge_usage_into_state(state, usage_total, now_str, max_recent=20):
    """
    Merge accumulated token usage into state['usage'] for D1 persistence.

    Call once at the end of a run, before save_state().
    Structure stored:
        state["usage"] = {
            "total_prompt_tokens": int,
            "total_completion_tokens": int,
            "total_tokens": int,
            "total_cache_hit_tokens": int,
            "runs": int,
            "recent": [{"ts", "prompt", "completion", "total", "cache_hit"}, ...]
        }
    """
    if not usage_total or usage_total.get("total", 0) == 0:
        return  # no AI calls this run

    u = state.get("usage")
    if u is None:
        u = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cache_hit_tokens": 0,
            "runs": 0,
            "recent": [],
        }
        state["usage"] = u

    u["total_prompt_tokens"] = u.get("total_prompt_tokens", 0) + usage_total["prompt"]
    u["total_completion_tokens"] = u.get("total_completion_tokens", 0) + usage_total["completion"]
    u["total_tokens"] = u.get("total_tokens", 0) + usage_total["total"]
    u["total_cache_hit_tokens"] = u.get("total_cache_hit_tokens", 0) + usage_total["cache_hit"]
    u["runs"] = u.get("runs", 0) + 1
    u.setdefault("recent", []).append({
        "ts": now_str,
        "prompt": usage_total["prompt"],
        "completion": usage_total["completion"],
        "total": usage_total["total"],
        "cache_hit": usage_total["cache_hit"],
    })
    # Keep only the most recent N entries to avoid unbounded growth
    if len(u["recent"]) > max_recent:
        u["recent"] = u["recent"][-max_recent:]


# ==================== Factory Functions ====================

def create_text_provider(config):
    """
    Create a text provider from config.

    Config format:
        {"provider": "workers_ai", "model": "...", "account_id": "...", "api_token": "..."}
        {"provider": "openai", "model": "...", "base_url": "...", "api_key": "..."}
    Credentials may be inline or referenced by *_env key names.
    """
    provider_name = config.get("provider", "workers_ai")

    if provider_name == "workers_ai":
        return WorkersAIText(config)
    elif provider_name == "openai":
        return OpenAIText(config)
    else:
        raise ValueError(f"Unknown text provider: {provider_name}")


def create_fallback_text_provider(configs, name="pool"):
    """
    Create a global fallback text pool from an ordered list of provider configs.

    Each entry is passed to create_text_provider; entries that fail to
    initialize (e.g. missing credentials) are logged and skipped so one bad
    entry doesn't kill the pool. Fails only when no provider could be built.
    """
    providers = []
    for idx, cfg in enumerate(configs):
        try:
            providers.append(create_text_provider(cfg))
        except Exception as e:
            model = cfg.get("model", "?")
            print(f"[pool:{name}] skipping provider #{idx} ({model}): {e}",
                  file=sys.stderr)
    if not providers:
        raise ValueError(f"pool[{name}]: no text provider could be initialized")
    return FallbackTextProvider(providers, name=name)


def create_image_provider(config):
    """
    Create an image provider from config.

    Config format:
        {"provider": "workers_ai", "model": "...", "account_id_env": "...", "api_token_env": "..."}
    """
    provider_name = config.get("provider", "workers_ai")

    if provider_name == "workers_ai":
        return WorkersAIImage(config)
    else:
        raise ValueError(f"Unknown image provider: {provider_name}")


# ==================== Model Pool (local file) ====================
#
# The free-model pool is a minimal, pre-sorted list written to model-pool.json
# at repo root by refresh_model_pool.py (run as a step of whisper_runner, not a
# separate cron). Entries are [{model, baseurl}, ...] with ordering already
# decided at refresh time. Each baseurl encodes the provider, so the client
# maps baseurl -> api key env var, builds OpenAI-compatible configs, and feeds
# them into the shared fallback pool.

# baseurl (host) -> api key env var name. Extend as new providers are added.
BASEURL_KEY_ENV = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    "opencode.ai": "ZEN_API_KEY",
}


def model_pool_path():
    """.whisper-task/model-pool.json (next to config.json), produced by
    refresh_model_pool.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "model-pool.json"))


def fetch_model_pool(timeout=20):
    """Read the minimal free-model pool from the local model-pool.json.

    Returns a list of {"model": str, "baseurl": str} entries, or [] when the
    file is missing or invalid. Raises only on an unreadable-but-present file.
    """
    path = model_pool_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return data.get("entries") or []


def create_pool_provider_configs(pool_entries):
    """Map {model, baseurl} entries to OpenAI provider configs.

    Each entry whose baseurl host maps to a configured env key becomes an
    OpenAI-compatible provider config with an inline api_key. Entries with no
    key, unknown host, or empty baseurl are skipped. Returns list of configs.
    """
    configs = []
    for entry in pool_entries or []:
        model = (entry.get("model") or "").strip()
        baseurl = (entry.get("baseurl") or "").strip().rstrip("/")
        if not model or not baseurl:
            continue
        try:
            from urllib.parse import urlparse
            host = urlparse(baseurl).netloc.lower()
        except Exception:
            host = ""
        key_env = BASEURL_KEY_ENV.get(host, "")
        api_key = os.environ.get(key_env, "").strip() if key_env else ""
        if not api_key:
            print(f"[model-pool] skipping {model}@{baseurl}: no {key_env or 'key env'}",
                  file=sys.stderr)
            continue
        configs.append({
            "provider": "openai",
            "model": model,
            "base_url": baseurl,
            "api_key": api_key,
        })
    return configs


def create_pool_text_provider(name="model-pool"):
    """Build a fallback text pool from the local model-pool.json.

    Missing file -> returns None (no pool). Never substitutes a default.
    """
    try:
        entries = fetch_model_pool()
    except Exception as e:
        print(f"[model-pool] read failed: {e}", file=sys.stderr)
        raise
    configs = create_pool_provider_configs(entries)
    if not configs:
        raise ValueError(f"pool[{name}]: no usable provider from model pool (check model-pool.json/api keys)")
    return create_fallback_text_provider(configs, name=name)


# ==================== CLI for testing ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI client test")
    parser.add_argument("--type", choices=["text", "image"], default="text")
    parser.add_argument("--provider", default="workers_ai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    config = {"provider": args.provider}
    if args.model:
        config["model"] = args.model

    if args.type == "text":
        provider = create_text_provider(config)
        messages = [{"role": "user", "content": args.prompt}]
        result = provider.generate(messages, temperature=0.85, enable_thinking=True)
        print(result)
    else:
        provider = create_image_provider(config)
        result = provider.generate(args.prompt, args.output)
        if result:
            print(f"Image saved to {result}")
        else:
            print("Image generation failed")
