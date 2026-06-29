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
import urllib.request
import urllib.error
import base64
from abc import ABC, abstractmethod


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
    def generate(self, messages, max_tokens=1024, temperature=0.8):
        """
        Generate text from chat messages.

        Args:
            messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
            max_tokens: max tokens to generate
            temperature: sampling temperature

        Returns:
            str: generated text
        """
        pass


class WorkersAIText(TextProvider):
    """Cloudflare Workers AI text provider."""

    def __init__(self, config):
        self.model = config.get("model", "@cf/zai-org/glm-4.7-flash")
        self.account_id = os.environ.get(config.get("account_id_env", "CF_DEFAULT_ACCOUNT_ID"), "")
        self.api_token = os.environ.get(config.get("api_token_env", "CF_DEFAULT_API_TOKEN"), "")

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires CF_DEFAULT_ACCOUNT_ID and CF_DEFAULT_API_TOKEN environment variables")

    def generate(self, messages, max_tokens=1024, temperature=0.8):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": { "enable_thinking": True }
        }

        data = json.dumps(payload).encode("utf-8")

        max_retries = 3
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Authorization", f"Bearer {self.api_token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "doubao-whispers/1.0")

            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

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
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                if _should_retry(status_code=e.code, body=err_body) and attempt < max_retries:
                    _retry_sleep(attempt, f"HTTP {e.code}")
                    continue
                raise RuntimeError(f"WorkersAI request failed: HTTP {e.code} {err_body[:200]}")
            except urllib.error.URLError as e:
                if attempt < max_retries:
                    _retry_sleep(attempt, f"URL error: {e}")
                    continue
                raise RuntimeError(f"WorkersAI request failed: {e}")


class OpenAIText(TextProvider):
    """OpenAI-compatible text provider (works with DeepSeek, Moonshot, GLM, etc.)."""

    def __init__(self, config):
        self.model = config.get("model", "deepseek-chat")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")

        if not self.api_key:
            raise ValueError("OpenAI provider requires API key environment variable")

        # Token usage tracking: last_usage = most recent call, usage_total = accumulated across all calls
        self.last_usage = None
        self.usage_total = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0}

    def generate(self, messages, max_tokens=1024, temperature=0.8):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Disable thinking mode (DeepSeek V4 defaults to thinking,
            # which consumes tokens on reasoning_content instead of content)
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        }

        data = json.dumps(payload).encode("utf-8")

        max_retries = 3
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Authorization", f"Bearer {self.api_key}")
            req.add_header("Content-Type", "application/json")
            # Custom User-Agent to avoid Cloudflare bot detection (1010)
            req.add_header("User-Agent", "doubao-whispers/1.0")

            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                # Some APIs return 200 + error body for rate limiting
                if result.get("error"):
                    err_str = str(result["error"])
                    if _should_retry(body=err_str) and attempt < max_retries:
                        _retry_sleep(attempt, f"Rate limited: {err_str[:80]}")
                        continue
                    raise RuntimeError(f"OpenAI error: {result['error']}")

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
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                if _should_retry(status_code=e.code, body=err_body) and attempt < max_retries:
                    _retry_sleep(attempt, f"HTTP {e.code}")
                    continue
                raise RuntimeError(f"OpenAI request failed: HTTP {e.code} {err_body[:200]}")
            except urllib.error.URLError as e:
                if attempt < max_retries:
                    _retry_sleep(attempt, f"URL error: {e}")
                    continue
                raise RuntimeError(f"OpenAI request failed: {e}")


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
        self.account_id = os.environ.get(config.get("account_id_env", "CF_IMAGE_ACCOUNT_ID"), "")
        self.api_token = os.environ.get(config.get("api_token_env", "CF_IMAGE_API_TOKEN"), "")

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires CF_IMAGE_ACCOUNT_ID and CF_IMAGE_API_TOKEN environment variables")

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

        # Build multipart form-data body manually (stdlib only, no requests)
        boundary = "----WorkersAIBoundary" + os.urandom(8).hex()
        body_parts = []

        def add_field(name, value):
            body_parts.append(f"--{boundary}\r\n".encode())
            body_parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body_parts.append(value.encode("utf-8") if isinstance(value, str) else value)
            body_parts.append(b"\r\n")

        def add_file_field(name, filename, content_bytes, content_type="image/png"):
            body_parts.append(f"--{boundary}\r\n".encode())
            body_parts.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
            body_parts.append(content_bytes)
            body_parts.append(b"\r\n")

        # Required fields
        add_field("prompt", prompt)
        add_field("width", "1024")
        add_field("height", "768")

        # Optional reference images (max 4). CF flux-2 accepts up to 4 512x512 tiles.
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
                        # Field name "image" for single, "image[]" for multiple.
                        # Using "image[]" for all so CF treats them as an array.
                        field_name = "image[]" if len(refs) > 1 else "image"
                        add_file_field(field_name, f"ref_{idx}.png", ref_bytes, "image/png")
                except Exception as e:
                    print(f"[image] Skipping reference {ref}: {e}", file=sys.stderr)

        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_token}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # CF returns 400 with JSON body when output is flagged (code 3030).
            try:
                result = json.loads(e.read().decode("utf-8"))
            except Exception:
                err = RuntimeError(f"CF image gen HTTP {e.code}: {e.reason}")
                err.flagged = False
                raise err
        except urllib.error.URLError as e:
            err = RuntimeError(f"CF image gen request failed: {e}")
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
        # Convert to RGB (drop alpha for PNG compatibility with the model)
        if img.mode in ("RGBA", "LA", "P"):
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
        {"provider": "workers_ai", "model": "...", "account_id_env": "...", "api_token_env": "..."}
        {"provider": "openai", "model": "...", "base_url": "...", "api_key_env": "..."}
    """
    provider_name = config.get("provider", "workers_ai")

    if provider_name == "workers_ai":
        return WorkersAIText(config)
    elif provider_name == "openai":
        return OpenAIText(config)
    else:
        raise ValueError(f"Unknown text provider: {provider_name}")


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
        result = provider.generate(messages, temperature=0.85)
        print(result)
    else:
        provider = create_image_provider(config)
        result = provider.generate(args.prompt, args.output)
        if result:
            print(f"Image saved to {result}")
        else:
            print("Image generation failed")
