#!/usr/bin/env python3
"""Cloudflare KV REST API client for the runner.

Used by apply_image_replacements() to pick up pending image-replacement
requests that the diagnostic endpoint (functions/api/_diag/replace-image.js)
wrote into KV.

Env vars (set in GitHub Actions):
  CF_KV_ACCOUNT_ID   — Cloudflare account ID
  CF_KV_NAMESPACE_ID — KV namespace ID (the diagnostic namespace)
  CF_KV_API_TOKEN    — API token with Workers KV read/write on that namespace

Usage:
    from kv_client import KVClient
    kv = KVClient()  # reads env vars
    keys = kv.list_keys(prefix="pending_replace:")
    value = kv.get_value(key)
    kv.delete_key(key)
"""

import os
import sys
import requests


_KV_BASE = "https://api.cloudflare.com/client/v4/accounts/{aid}/storage/kv/namespaces/{nid}"


class KVClient:
    """Cloudflare KV REST API client (runner side)."""

    def __init__(self, account_id=None, namespace_id=None, api_token=None):
        self.account_id = account_id or os.environ.get("CF_KV_ACCOUNT_ID", "")
        self.namespace_id = namespace_id or os.environ.get("CF_KV_NAMESPACE_ID", "")
        self.api_token = api_token or os.environ.get("CF_KV_API_TOKEN", "")

        if not (self.account_id and self.namespace_id and self.api_token):
            raise ValueError(
                "KVClient requires CF_KV_ACCOUNT_ID, CF_KV_NAMESPACE_ID, "
                "and CF_KV_API_TOKEN environment variables"
            )

    def _url(self, suffix=""):
        base = _KV_BASE.format(aid=self.account_id, nid=self.namespace_id)
        return f"{base}/{suffix}" if suffix else base

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_token}",
            "User-Agent": "doubao-whispers/1.0",
        }

    def list_keys(self, prefix="", limit=1000):
        """List keys in the namespace, optionally filtered by prefix.

        Returns list of key name strings. KV list returns at most 1000 keys
        per call; for simplicity we fetch one page (the diagnostic use case
        won't have thousands of pending replacements).
        """
        params = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        resp = requests.get(
            self._url("keys"),
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"KV list failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        if not data.get("success"):
            errors = data.get("errors", [])
            msg = errors[0].get("message", "unknown") if errors else "unknown"
            raise RuntimeError(f"KV list error: {msg}")
        return [k["name"] for k in (data.get("result") or [])]

    def get_value(self, key):
        """Get a key's value as a string (the endpoint stores JSON strings)."""
        resp = requests.get(
            self._url(f"values/{key}"),
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"KV get failed: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.text

    def delete_key(self, key):
        """Delete a key."""
        resp = requests.delete(
            self._url(f"values/{key}"),
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"KV delete failed: HTTP {resp.status_code} {resp.text[:200]}")
        return True
