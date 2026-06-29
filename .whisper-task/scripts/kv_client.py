#!/usr/bin/env python3
"""Cloudflare KV REST API client.

Generic client for the diagnostic KV namespace. Reads/writes/deletes keys
via the Cloudflare REST API. The client itself is feature-agnostic —
callers decide the key prefix/structure, so future diagnostic features
(e.g. diag:reload, diag:flush) can share this same client without
per-feature env vars.

The client is constructed with explicit account_id / namespace_id /
api_token; the caller chooses which env vars to read them from (e.g.
CF_DEFAULT_ACCOUNT_ID + CF_DEFAULT_API_TOKEN + CF_DIAG_KV_ID, so the
diag namespace reuses the existing default CF credentials rather than
needing its own token).

Usage:
    from kv_client import KVClient
    kv = KVClient(account_id, namespace_id, api_token)
    keys = kv.list_keys(prefix="diag:replace:")
    value = kv.get_value(key)
    kv.delete_key(key)
    kv.put_value(key, value)
"""

import requests


_KV_BASE = "https://api.cloudflare.com/client/v4/accounts/{aid}/storage/kv/namespaces/{nid}"


class KVClient:
    """Cloudflare KV REST API client (feature-agnostic)."""

    def __init__(self, account_id, namespace_id, api_token):
        if not (account_id and namespace_id and api_token):
            raise ValueError(
                "KVClient requires account_id, namespace_id, and api_token"
            )
        self.account_id = account_id
        self.namespace_id = namespace_id
        self.api_token = api_token

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
        won't have thousands of pending requests).
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
        """Get a key's value as a string."""
        resp = requests.get(
            self._url(f"values/{key}"),
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"KV get failed: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.text

    def put_value(self, key, value, ttl=None):
        """Write a key's value. value is a string. ttl is optional seconds."""
        data = {"value": value}
        if ttl:
            data["expiration_ttl"] = ttl
        resp = requests.put(
            self._url(f"values/{key}"),
            headers=self._headers(),
            data=data,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"KV put failed: HTTP {resp.status_code} {resp.text[:200]}")
        return True

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
