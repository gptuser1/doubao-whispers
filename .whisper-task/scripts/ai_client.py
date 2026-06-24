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
import urllib.request
import urllib.error
import base64
from abc import ABC, abstractmethod


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
        self.model = config.get("model", "@cf/qwen/qwen1.5-14b-chat-awq")
        self.account_id = os.environ.get(config.get("account_id_env", "CF_ACCOUNT_ID"), "")
        self.api_token = os.environ.get(config.get("api_token_env", "CF_API_TOKEN"), "")

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires CF_ACCOUNT_ID and CF_API_TOKEN environment variables")

    def generate(self, messages, max_tokens=1024, temperature=0.8):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if result.get("success"):
                return result.get("result", {}).get("response", "").strip()
            else:
                errors = result.get("errors", [])
                err_msg = errors[0].get("message", "unknown error") if errors else "unknown error"
                raise RuntimeError(f"WorkersAI error: {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"WorkersAI request failed: {e}")


class OpenAIText(TextProvider):
    """OpenAI-compatible text provider (works with DeepSeek, Moonshot, GLM, etc.)."""

    def __init__(self, config):
        self.model = config.get("model", "deepseek-chat")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")

        if not self.api_key:
            raise ValueError("OpenAI provider requires API key environment variable")

    def generate(self, messages, max_tokens=1024, temperature=0.8):
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
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
    """Cloudflare Workers AI image provider."""

    def __init__(self, config):
        self.model = config.get("model", "@cf/black-forest-labs/flux-1-schnell")
        self.account_id = os.environ.get(config.get("account_id_env", "CF_ACCOUNT_ID"), "")
        self.api_token = os.environ.get(config.get("api_token_env", "CF_API_TOKEN"), "")

        if not self.account_id or not self.api_token:
            raise ValueError("WorkersAI requires CF_ACCOUNT_ID and CF_API_TOKEN environment variables")

    def generate(self, prompt, output_path, reference_images=None, size="landscape_4_3"):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        payload = {"prompt": prompt}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")

                if "image" in content_type:
                    # Direct image response
                    image_data = resp.read()
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                    return output_path
                else:
                    # JSON response with base64 or error
                    result = json.loads(resp.read().decode("utf-8"))
                    if result.get("success"):
                        image_b64 = result.get("result", {}).get("image", "")
                        if image_b64:
                            image_data = base64.b64decode(image_b64)
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            return output_path

                    errors = result.get("errors", [])
                    err_msg = errors[0].get("message", "unknown error") if errors else "unknown error"
                    print(f"WorkersAI image error: {err_msg}", file=sys.stderr)
                    return None
        except urllib.error.URLError as e:
            print(f"WorkersAI image request failed: {e}", file=sys.stderr)
            return None


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
        result = provider.generate(messages)
        print(result)
    else:
        provider = create_image_provider(config)
        result = provider.generate(args.prompt, args.output)
        if result:
            print(f"Image saved to {result}")
        else:
            print("Image generation failed")
