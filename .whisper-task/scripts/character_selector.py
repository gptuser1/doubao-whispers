#!/usr/bin/env python3
"""
Character selector for whisper publishing (fallback only).

Primary character selection is done by AI (see whisper_runner.py).
This module provides a weighted-random fallback when AI generation fails.

Chooses which character publishes the next whisper, based on:
- Recent authors (avoid repetition)
- Character activity weight (lively characters post more)
- Random selection with weighting
"""

import json
import os
import random
import sys


# Character weights: lively characters post more, quiet ones less
CHARACTER_WEIGHTS = {
    "guga": 2.0,    # lively, penguin girl
    "doro": 2.0,    # lively, puppy girl
    "feibi": 2.0,   # energetic, blonde girl
    "nuonuo": 1.8,  # gamer girl, slightly less but still active
    "doubao": 1.5,  # big sister, organizes but doesn't post as much
    "baizi": 1.0,   # quiet, wolf girl, posts less
}

# How many recent authors to exclude
RECENT_EXCLUDE_COUNT = 2


def get_recent_authors(whispers_dir, count=5):
    """
    Get the authors of the most recent whispers.

    Args:
        whispers_dir: path to data/whispers/
        count: how many recent whispers to check

    Returns:
        list of author IDs, most recent first
    """
    posts = []

    if not os.path.exists(whispers_dir):
        return []

    for name in os.listdir(whispers_dir):
        if not name.endswith(".json"):
            continue
        filepath = os.path.join(whispers_dir, name)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                whispers = json.load(f)
            for slug, whisper in whispers.items():
                date_str = whisper.get("date", "")
                author = whisper.get("author", "")
                if date_str and author:
                    posts.append((date_str, author))
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by date descending
    posts.sort(key=lambda x: x[0], reverse=True)

    return [author for _, author in posts[:count]]


def select_character(whispers_dir, random_seed=None):
    """
    Select a character to publish the next whisper.

    Logic:
    1. Get recent authors
    2. Exclude the most recent N authors to avoid repetition
    3. Weight remaining characters by activity level
    4. Random selection

    Args:
        whispers_dir: path to data/whispers/
        random_seed: optional seed for reproducibility

    Returns:
        str: selected character ID (e.g. "guga", "doro")
    """
    if random_seed is not None:
        random.seed(random_seed)

    recent_authors = get_recent_authors(whispers_dir)

    # Build candidate list with weights
    candidates = []
    weights = []

    for char_id, weight in CHARACTER_WEIGHTS.items():
        # Exclude recent authors
        if char_id in recent_authors[:RECENT_EXCLUDE_COUNT]:
            continue

        # Reduce weight if character appeared in recent 5 (but not excluded)
        recent_count = recent_authors.count(char_id)
        adjusted_weight = weight * (0.5 ** recent_count)

        candidates.append(char_id)
        weights.append(max(adjusted_weight, 0.1))  # floor at 0.1

    # If all characters were excluded (unlikely), fall back to all
    if not candidates:
        candidates = list(CHARACTER_WEIGHTS.keys())
        weights = list(CHARACTER_WEIGHTS.values())

    # Weighted random selection
    selected = random.choices(candidates, weights=weights, k=1)[0]

    return selected


# ==================== CLI for testing ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Character selector test")
    parser.add_argument("--whispers-dir", default="data/whispers", help="Whispers data directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--count", type=int, default=10, help="Run N times to see distribution")
    args = parser.parse_args()

    if args.count == 1:
        char = select_character(args.whispers_dir, args.seed)
        print(f"Selected character: {char}")
    else:
        distribution = {}
        for i in range(args.count):
            char = select_character(args.whispers_dir, random_seed=i)
            distribution[char] = distribution.get(char, 0) + 1

        print(f"Distribution over {args.count} runs:")
        for char, count in sorted(distribution.items(), key=lambda x: -x[1]):
            print(f"  {char}: {count} ({count/args.count*100:.0f}%)")
