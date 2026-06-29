#!/usr/bin/env python3
"""
D1 REST API client.
Handles state read/write and replies query/delete.

Uses the REST API at data.klinux.dpdns.org with Bearer token auth.
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone, timedelta


# Beijing timezone
TZ_BEIJING = timezone(timedelta(hours=8))


class D1Client:
    """D1 REST API client for state management and replies."""

    def __init__(self, api_url=None, api_key=None):
        self.api_url = (api_url or os.environ.get("D1_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("D1_API_KEY", "")

        if not self.api_url or not self.api_key:
            raise ValueError("D1Client requires D1_API_URL and D1_API_KEY")

    def _query(self, sql, params=None):
        """Execute a SQL query via the REST API."""
        url = f"{self.api_url}/query"
        payload = {"query": sql, "params": params or []}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "WhisperRunner/1.0",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"D1 request failed: {e}")

        if resp.status_code >= 400:
            raise RuntimeError(f"D1 request failed: HTTP {resp.status_code} {resp.text[:200]}")

        result = resp.json()
        if not result.get("success", True) and "error" in result:
            raise RuntimeError(f"D1 query error: {result['error']}")

        return result.get("results", [])

    # ==================== State Management ====================

    def get_state(self):
        """
        Read the global state from D1.
        Returns a dict with last_run, next_random_offset, stats.
        Returns empty state if not found.
        """
        results = self._query("SELECT value FROM state WHERE key = ?;", ["whisper_state"])

        if not results:
            return self._default_state()

        row = results[0]
        value_str = row.get("value", "{}") if isinstance(row, dict) else "{}"

        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            print("Warning: state JSON parse failed, returning default", file=sys.stderr)
            return self._default_state()

    def save_state(self, state):
        """
        Save the global state to D1.
        Uses UPSERT to handle both insert and update.
        """
        value_str = json.dumps(state, ensure_ascii=False)
        now_str = datetime.now(TZ_BEIJING).strftime("%Y-%m-%d %H:%M:%S")

        self._query(
            "INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?;",
            ["whisper_state", value_str, now_str, value_str, now_str]
        )

    def _default_state(self):
        """Return a default empty state."""
        return {
            "last_run": {},
            "next_random_offset": {},
            "stats": {
                "total_heartbeats": 0,
                "total_tasks_executed": 0,
            }
        }

    # ==================== Replies Management ====================

    def get_pending_replies(self):
        """
        Query unprocessed user replies (is_doubao = 0) from the replies table.
        Returns list of reply dicts, ordered by timestamp ascending.
        Returns empty list if no replies or table doesn't exist.
        """
        try:
            results = self._query(
                "SELECT * FROM replies WHERE is_doubao = 0 ORDER BY timestamp ASC;",
                []
            )
            return results if results else []
        except RuntimeError as e:
            print(f"Warning: failed to query replies: {e}", file=sys.stderr)
            return []

    def delete_replies(self, reply_ids):
        """
        Delete user replies from D1 after they've been synced to the repo json.

        D1 is only a transient cache for pending user replies — the canonical
        store is data/replies/*.json in the repo. Once the runner has written
        them to the repo, the D1 rows are dead data and should be removed so
        the cache doesn't grow unbounded.
        """
        if not reply_ids:
            return

        for reply_id in reply_ids:
            try:
                self._query(
                    "DELETE FROM replies WHERE id = ?;",
                    [reply_id]
                )
            except RuntimeError as e:
                print(f"Warning: failed to delete reply {reply_id}: {e}", file=sys.stderr)

    def get_max_floor(self, whisper_id):
        """Get the current max floor number for a whisper."""
        try:
            results = self._query(
                "SELECT MAX(floor) as max_floor FROM replies WHERE whisper_id = ?;",
                [whisper_id]
            )
            if results and results[0].get("max_floor") is not None:
                return results[0]["max_floor"]
            return 0
        except RuntimeError:
            return 0


# ==================== CLI for testing ====================

if __name__ == "__main__":
    client = D1Client()

    print("=== Current State ===")
    state = client.get_state()
    print(json.dumps(state, ensure_ascii=False, indent=2))

    print("\n=== Pending Replies ===")
    replies = client.get_pending_replies()
    print(f"Found {len(replies)} pending replies")
    for r in replies:
        print(f"  {r}")
