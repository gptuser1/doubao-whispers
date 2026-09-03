#!/usr/bin/env python3
"""Unified logger for whisper task scripts.

Replaces bare `print(...)` calls with structured, tag-based logging.
Every log message is followed by a blank line for readability in CI logs.

Usage:
    from log import log

    log("pool is fresh, nothing to do")                      # plain
    log("AA_API_KEY unset", tag="model-pool")                 # [model-pool] ...
    log(f"retrying in {delay}s", tag="AI retry")              # [AI retry] ...
    log("some output", file=sys.stdout)                       # stdout
"""

import sys


def log(msg, tag=None, file=sys.stderr):
    """Print a formatted log message followed by a blank line."""
    if tag is not None:
        line = f"[{tag}] {msg}"
    else:
        line = str(msg)
    print(line, file=file)
    print(file=file)  # trailing blank line for readability