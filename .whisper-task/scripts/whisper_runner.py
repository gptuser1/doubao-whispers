#!/usr/bin/env python3
"""
Whisper runner - main orchestration script.

Executed by GitHub Actions on a cron schedule.
Handles:
1. Check if publish_whisper should trigger (probabilistic)
2. If triggered: select character, generate content via AI, save to JSON
3. Check and reply to user comments from D1
4. Update D1 state
5. Git commit and push

Usage:
    python .whisper-task/scripts/whisper_runner.py [--dry-run] [--force-publish]
"""

import argparse
import json
import os
import subprocess
import sys
import random
import re
import time
import base64
from datetime import datetime, timezone, timedelta

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ai_client import create_text_provider, create_image_provider, merge_usage_into_state
from d1_client import D1Client
from kv_client import KVClient
from character_selector import select_character, CHARACTER_WEIGHTS
from reply_utils import add_replies, load_month_file

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, ".whisper-task", "config.json")
CHARACTERS_PATH = os.path.join(PROJECT_ROOT, ".whisper-task", "characters.md")
HOLIDAYS_PATH = os.path.join(PROJECT_ROOT, ".whisper-task", "holidays.json")
WHISPERS_DIR = os.path.join(PROJECT_ROOT, "data", "whispers")
REPLIES_DIR = os.path.join(PROJECT_ROOT, "data", "replies")
AUTHORS_PATH = os.path.join(PROJECT_ROOT, "data", "authors.json")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "static", "images")
PACK_IMAGES_SCRIPT = os.path.join(SCRIPT_DIR, "pack_images.py")
PROCESS_IMAGE_SCRIPT = os.path.join(SCRIPT_DIR, "process_image.py")

# Image generation config
# - 70% of recent K whispers should have images (pure-text ratio <= 30%)
# - Required image when: mentions other characters (group photo), or has
#   concrete scene/object, or has photo-related words
PURE_TEXT_RATIO_THRESHOLD = 0.30
RECENT_K_FOR_IMAGE_STATS = 10
IMAGE_PROBABILITY = 0.70  # when not mandatory and stats allow pure-text
MAX_REFERENCE_IMAGES = 4

# Hardcoded appearance constraint - ALWAYS PREPENDED to the AI-generated scene
# description, placed at the very front of the final prompt (highest priority
# position). Lightweight two-line form (validated to work well with the 4B flux
# model): explicitly point the model to the `image[]` multipart fields where the
# reference avatar images live, then list which attributes are FIXED (names
# only, never described - the reference images define how they look). Kept short
# on purpose: a long absolute constraint eats the 4B model's limited context
# without improving fidelity. Code-fixed (not AI-generated) so the AI cannot
# drift into describing appearance.
IMAGE_APPEARANCE_HARD_CONSTRAINT = (
    "Reference image(s) show the character(s).\nFor single reference image: apply all fixed attributes to the sole character.\nFor multiple reference images (reference image 1, reference image 2, ...): each image represents a distinct character. All characters must appear together in the rendered scene. Apply fixed attributes to their respective referenced character.\nKeep the reference image's clothing style. No exposed skin.\nStyle anchor: kawaii chibi anime illustration, thick clean black outlines, soft cel shading, pastel warm colors, rounded cute proportions, big glossy eyes, gentle warm ambient lighting, cozy daily scene, soft blush, clean flat coloring.\nFixed attributes: face shape, hairstyle, skin tone, eye color, body build, clothing style, hand details."
)

# Beijing timezone
TZ_BEIJING = timezone(timedelta(hours=8))

# Default character moods
DEFAULT_MOODS = {
    "guga": "happy", "doro": "happy", "feibi": "excited",
    "nuonuo": "content", "doubao": "content", "baizi": "calm",
}

# Topic keywords for content extraction
TOPIC_KEYWORDS = [
    "端午", "粽子", "游戏", "番剧", "奶茶", "咖啡", "橘子", "蛋糕",
    "零食", "炸鸡", "电影", "健身", "跑步", "逛街", "超市", "假期",
    "周末", "上班", "上学", "摸鱼", "睡觉", "散步", "龙舟", "加班",
    "下雨", "晴天", "云", "花", "猫", "狗", "茶", "书", "音乐",
]


def _extract_topics(content):
    """Extract 1-2 topic keywords from content."""
    found = []
    for t in TOPIC_KEYWORDS:
        if t in content:
            found.append(t)
    return found[:2]


def _infer_mood(content, character_id):
    """Infer character mood from content text."""
    happy_words = ["开心", "好吃", "好玩", "快乐", "好棒", "开心", "幸福", "喜欢", "好看"]
    sad_words = ["困", "累", "好烦", "不开心", "难过", "委屈", "好惨", "讨厌", "无聊"]
    for w in happy_words:
        if w in content:
            return "happy"
    for w in sad_words:
        if w in content:
            return "tired"
    return DEFAULT_MOODS.get(character_id, "content")


MOOD_CN = {
    "happy": "开心", "sad": "低落", "tired": "有点累", "excited": "兴奋",
    "calm": "平静", "grumpy": "烦躁", "content": "满足",
}


def _get_character_state_hint(character_id, character_name, character_states):
    """Build a state hint string for prompts. Returns empty string if no state."""
    cs = character_states.get(character_id) if character_states else None
    if not cs:
        return ""

    mood = MOOD_CN.get(cs.get("mood", ""), "平静")
    energy = cs.get("energy", 50)
    energy_desc = "很充沛" if energy > 70 else "一般" if energy > 40 else "有点累"
    topics = "、".join(cs.get("recent_topics", [])[:2])

    parts = [f"【{character_name}的当前状态】心情：{mood}，精力：{energy_desc}"]
    if topics:
        parts.append(f"最近聊过：{topics}")
    return "\n".join(parts) + "\n"


def _evolve_character_states(states, now_dt):
    """Evolve character states based on time."""
    hour = now_dt.hour
    changed = False

    # Initialize default states for characters without one
    all_char_ids = ["guga", "doro", "feibi", "nuonuo", "doubao", "baizi"]
    for cid in all_char_ids:
        if cid not in states:
            states[cid] = {
                "mood": DEFAULT_MOODS.get(cid, "content"),
                "energy": 60,
                "recent_topics": [],
                "last_active": "",
            }
            changed = True

    for char_id, cs in states.items():
        old_energy = cs.get("energy", 50)

        # Energy changes by time of day
        if 6 <= hour < 9:
            cs["energy"] = min(100, cs.get("energy", 50) + 20)
        elif 13 <= hour < 15:
            cs["energy"] = max(30, cs.get("energy", 50) - 10)
        elif 22 <= hour or hour < 6:
            cs["energy"] = max(15, cs.get("energy", 50) - 20)

        # Gradual mood drift toward default (30% chance)
        if random.random() < 0.3:
            default = DEFAULT_MOODS.get(char_id, "content")
            # Only drift if not already at default
            if cs.get("mood") != default:
                cs["mood"] = default
                changed = True

        if cs.get("energy") != old_energy:
            changed = True

    if changed:
        pass  # caller saves state

    return states, changed

# Active hours (only run between these hours, Beijing time)
ACTIVE_HOUR_START = 7
ACTIVE_HOUR_END = 23


def now_beijing():
    """Get current Beijing time."""
    return datetime.now(TZ_BEIJING)


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """Save data to a JSON file."""
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_script(cmd, cwd=PROJECT_ROOT):
    """Run a subprocess and return stdout."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result.stdout, result.returncode


def get_timeline_text(count=15):
    """Get recent timeline as text for AI context."""
    stdout, rc = run_script([
        sys.executable,
        os.path.join(SCRIPT_DIR, "get_timeline.py"),
        "--count", str(count)
    ])
    return stdout if rc == 0 else ""


def check_trigger(task_name, last_run, now_str, random_offset=None):
    """Check if a task should trigger using check_trigger.py."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "check_trigger.py"),
        "--task", task_name,
        "--last-run", last_run,
        "--now", now_str,
    ]
    if random_offset is not None:
        cmd.extend(["--random-offset", str(random_offset)])

    stdout, rc = run_script(cmd)
    if rc != 0:
        return {"trigger": False, "reason": "check_trigger.py failed"}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"trigger": False, "reason": "check_trigger.py output parse error"}


def check_holiday(date_str):
    """Check holiday info for a date."""
    stdout, rc = run_script([
        sys.executable,
        os.path.join(SCRIPT_DIR, "check_trigger.py"),
        "holiday", date_str
    ])
    if rc != 0:
        return {"type": "workday", "is_workday": True, "holiday_name": None}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"type": "workday", "is_workday": True, "holiday_name": None}


def check_slug(month_json_path, slug):
    """Check if a slug is available."""
    stdout, rc = run_script([
        sys.executable,
        os.path.join(SCRIPT_DIR, "check_slug.py"),
        month_json_path, slug
    ])
    if rc != 0:
        return slug

    try:
        result = json.loads(stdout)
        return result.get("suggested_slug", slug)
    except json.JSONDecodeError:
        return slug


def get_author_nickname(author_id, authors_data):
    """Get nickname by author ID."""
    if author_id in authors_data:
        return authors_data[author_id].get("name", author_id)
    return author_id


# ==================== Image Generation ====================

# Reference image mapping for image generation. These are high-quality
# 512x512 PNG character portraits in static/images/input/, used as reference
# tiles for the CF flux-2 model. Separate from display avatars (avatar-*.webp)
# which are tiny (200x200) and only used for the website UI.
REFERENCE_IMAGE_FILES = {
    "doubao": "input/doubao.png",
    "guga": "input/guga.png",
    "doro": "input/doro.png",
    "feibi": "input/feibi.png",
    "baizi": "input/baizi.png",
    "nuonuo": "input/nuonuo.png",
}

# Romanized names for the image prompt. The flux-2-klein-4b model is
# English-trained; Chinese characters in the prompt degrade output quality.
# Reference images are matched by position (image[] order), so the names in the
# prompt only help the model understand WHO does WHAT - romanized names keep the
# whole prompt English without losing meaning.
NAME_ROMANIZATION = {
    "doubao": "Doubao",
    "guga": "Guga",
    "doro": "Doro",
    "feibi": "Feibi",
    "baizi": "Baizi",
    "nuonuo": "Nuonuo",
}

def get_reference_image_path(author_id):
    """Return absolute path to an author's 512x512 PNG reference image, or None."""
    fname = REFERENCE_IMAGE_FILES.get(author_id)
    if not fname:
        return None
    path = os.path.join(IMAGES_DIR, fname)
    return path if os.path.exists(path) else None


# Cache for pre-processed reference image PNG bytes, keyed by character id.
# Populated once at startup by preload_references() so image generation can
# reuse the cached bytes without re-reading and re-processing files.
_REFERENCE_CACHE = {}


def preload_references():
    """Load and pre-process all character reference images into _REFERENCE_CACHE.

    Converts each reference image to <=512x512 PNG bytes (the format CF flux-2
    expects), so image generation can pass cached bytes directly instead of
    re-reading files on every call. The input/*.png files are already 512x512
    so no resizing happens — they pass through at full resolution.
    """
    from ai_client import _prepare_reference_image
    for aid in REFERENCE_IMAGE_FILES:
        path = get_reference_image_path(aid)
        if path:
            png_bytes = _prepare_reference_image(path)
            if png_bytes:
                _REFERENCE_CACHE[aid] = png_bytes
                print(f"[ref] Cached {aid}: {len(png_bytes)} bytes",
                      file=sys.stderr)
            else:
                print(f"[ref] Failed to cache {aid}", file=sys.stderr)
        else:
            print(f"[ref] No reference image file for {aid}", file=sys.stderr)


def extract_mentioned_characters(content, authors_data):
    """Find which characters are mentioned in the whisper content.

    Matches both nicknames (e.g. "菲比") and ids (e.g. "feibi"). Returns a
    list of author ids found in the content.
    """
    mentioned = []
    content_lower = content.lower()
    for aid, info in authors_data.items():
        nick = info.get("name", aid)
        if nick in content or aid in content_lower:
            mentioned.append(aid)
    return mentioned


def _has_image_field(whisper_data):
    """Check if a whisper dict (full data from JSON) has images."""
    imgs = whisper_data.get("images") if isinstance(whisper_data, dict) else None
    return bool(imgs and len(imgs) > 0)


def load_recent_whispers_with_images(count=10):
    """Load recent whispers including their images field, for image-ratio stats."""
    posts = []
    if not os.path.exists(WHISPERS_DIR):
        return posts
    month_files = sorted([f for f in os.listdir(WHISPERS_DIR) if f.endswith(".json")],
                         reverse=True)
    for mf in month_files:
        if len(posts) >= count * 2:
            break
        with open(os.path.join(WHISPERS_DIR, mf), "r", encoding="utf-8") as f:
            data = json.load(f)
        for slug, w in data.items():
            if isinstance(w, dict) and w.get("date"):
                posts.append({
                    "date": w["date"],
                    "author": w.get("author", ""),
                    "title": w.get("title", ""),
                    "content": w.get("content", ""),
                    "images": w.get("images", []),
                })
    posts.sort(key=lambda x: x["date"], reverse=True)
    return posts[:count]


def should_have_image(content, character_id, authors_data, recent_with_images,
                      model_needs_image=None, model_mentioned=None):
    """Decide whether a new whisper should have an image.

    Rules (updated to use model-provided metadata):
    1. Model-provided mentioned_characters (non-empty) -> mandatory group photo
    2. Model says needs_image -> mandatory
    3. Recent pure-text ratio >= 30% -> mandatory
    4. Otherwise: 70% probability

    Returns (should_have: bool, reason: str, mentioned_chars: list).
    """
    # Rule 1: model-provided mentioned characters -> mandatory group photo
    mentioned = []
    if model_mentioned:
        mentioned = [m for m in model_mentioned if m != character_id]
    if not mentioned:
        # Fallback to keyword extraction if model didn't provide any
        mentioned = [m for m in extract_mentioned_characters(content, authors_data)
                     if m != character_id]
    if mentioned:
        return True, f"mentions other characters: {mentioned}", mentioned

    # Rule 2: model says needs_image
    if model_needs_image:
        return True, "model decided needs_image", mentioned

    # Rule 3: recent pure-text ratio >= 30% -> mandatory
    if recent_with_images:
        total = len(recent_with_images)
        with_img = sum(1 for w in recent_with_images if w.get("images"))
        pure_text_ratio = (total - with_img) / total if total > 0 else 0
        if pure_text_ratio >= PURE_TEXT_RATIO_THRESHOLD:
            return True, f"pure-text ratio {pure_text_ratio:.0%} >= {PURE_TEXT_RATIO_THRESHOLD:.0%}", mentioned

    # Rule 4: probabilistic (70%)
    if random.random() < IMAGE_PROBABILITY:
        return True, f"probabilistic (p={IMAGE_PROBABILITY})", mentioned
    return False, "skipped (no mandatory rule, probability missed)", mentioned


def build_image_prompt(text_provider, content, character_id, mentioned_chars,
                       authors_data, now_dt):
    """Build an English image generation prompt via Qwen3-8B.

    The prompt is critical for the 4B flux model - it must be concrete, visual,
    and describe the SCENE and ACTION only, using a labeled multi-line format
    (Action/Object/Expression/Setting/Time/Weather/Environment/Mood/Style/
    Palette/Lighting/Quality).

    IMPORTANT: Character appearance (hair color, features, clothing, etc.) is
    NEVER described in the prompt. The reference avatar images are the SOLE
    source of character appearance - they are passed to the model separately.
    The prompt only describes what the characters are DOING and the scene
    around them. A code-fixed hard constraint is PREPENDED to this prompt at
    generation time to lock appearance to the reference images.

    Returns a string prompt, or None on failure.
    """
    # Build character-reference mapping. Reference images are sent in the
    # same order as this list, so the AI can refer to characters by their
    # reference image number instead of by name (the image model doesn't
    # know character names — only the reference image positions matter).
    def _roman(aid):
        return NAME_ROMANIZATION.get(aid, get_author_nickname(aid, authors_data))
    char_names = [_roman(character_id)] + [_roman(a) for a in mentioned_chars]
    ref_mapping = ", ".join(
        f"reference image {i+1} = {name}" for i, name in enumerate(char_names)
    )

    system_prompt = f"""You are a prompt generator. Your task is to generate a structured character scene description based on user prompt.

Characters correspond to reference images in order: {ref_mapping}
Refer to characters as "the character in reference image 1", "the character in reference image 2", etc. Do NOT use character names in the output — the image model only knows characters by their reference image position.

Variable Fields to Fill (based on user prompt, generate values for these only):

    ===FIELDS START===
    Action:
    Object:
    Expression:
    Setting:
    Time:
    Weather:
    Environment details:
    Mood:
    Palette:
    Lighting:
    Quality:
    ===FIELDS END===


    Rules:

    1. Only generate content for the Variable Fields listed above.
    2. Keep all values concise, descriptive, and comma-separated where appropriate.
    3. Refer to characters by reference image number, never by name.
    4. Output format must be:

    Action: [your value]
    Object: [your value]
    Expression: [your value]
    Setting: [your value]
    Time: [your value]
    Weather: [your value]
    Environment details: [your value]
    Mood: [your value]
    Palette: [your value]
    Lighting: [your value]
    Quality: [your value]


    Example Output:

    Action: the character in reference image 1 is cooking, stirring a pot, holding a wooden spoon
    Object: pot of soup, wooden spoon, ingredients on counter
    Expression: focused, slightly surprised, happy
    Setting: small kitchen with counter and stove
    Time: evening, dinner time
    Weather: sunny outside, warm inside
    Environment details: wooden cabinets, hanging utensils, soft lighting from overhead lamp, steam from pot
    Mood: content, joyful, domestic
    Palette: warm oranges, soft pinks, creamy whites, earthy browns
    Lighting: soft golden light from lamp, gentle reflections on surfaces
    Quality: detailed rendering, clean lines, 4k"""

    user_prompt = f"""Whisper content (If Chinese, translate the scene to English in the prompt):
\"\"\"{content}\"\"\"

Time: {now_dt.strftime('%Y-%m-%d %H:%M')} (Beijing time)

Write the image prompt now using the labeled format above. Only the prompt, nothing else."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = text_provider.generate(messages, max_tokens=10240, temperature=0.3, enable_thinking=True)
        prompt = prompt.strip().strip('"').strip("'").strip("`")
        # Strip markdown code fence if present
        if prompt.startswith("```"):
            lines = prompt.split("\n")
            prompt = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return prompt if prompt else None
    except Exception as e:
        print(f"[image] Prompt generation failed: {e}", file=sys.stderr)
        return None


def generate_whisper_image(image_provider, rephrase_provider, content, character_id,
                           mentioned_chars, authors_data, now_dt, slug, date_str):
    """Generate one image for a whisper.

    Args:
        image_provider: WorkersAIImage instance
        rephrase_provider: text provider for prompt building/rephrasing (Qwen3-8B)
        content: whisper content text
        character_id: author id
        mentioned_chars: list of other character ids mentioned
        authors_data: authors dict
        now_dt: datetime
        slug: whisper slug
        date_str: YYYY-MM-DD string

    Returns:
        (image_filename: str or None, image_path: str or None)
        image_filename is the webp filename (e.g. "2026-06-26-foo-1.webp")
        image_path is absolute path to the saved file.
    """
    # Build prompt
    prompt = build_image_prompt(rephrase_provider, content, character_id,
                                mentioned_chars, authors_data, now_dt)
    if not prompt:
        print("[image] Failed to build prompt, skipping image", file=sys.stderr)
        return None, None
    print(f"[image] Prompt: {prompt[:120]}...")

    # Collect reference images: author + mentioned chars' reference images (max 4)
    # Use pre-cached PNG bytes if available, fall back to file paths.
    ref_chars = [character_id] + mentioned_chars
    ref_data = []
    for aid in ref_chars:
        cached = _REFERENCE_CACHE.get(aid)
        if cached:
            ref_data.append((aid, cached))
        else:
            p = get_reference_image_path(aid)
            if p:
                ref_data.append(p)
            else:
                print(f"[image] No reference image found for {aid}", file=sys.stderr)
    ref_data = ref_data[:MAX_REFERENCE_IMAGES]
    if not ref_data:
        print("[image] No reference images available, skipping image", file=sys.stderr)
        return None, None
    ref_labels = [r[0] if isinstance(r, tuple) else os.path.basename(r) for r in ref_data]
    print(f"[image] Reference images: {ref_labels}")

    # Output path: static/images/YYYY-MM-DD-{slug}-1.webp
    # CF returns PNG (base64); we save as .png then convert to .webp
    image_filename = f"{date_str}-{slug}-1.webp"
    final_path = os.path.join(IMAGES_DIR, image_filename)
    # Ensure images dir exists
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Try up to 3 rephrases if flagged by safety filter, then simplify.
    # current_prompt holds the AI-generated SCENE description only; the
    # hardcoded appearance constraint is PREPENDED fresh on every attempt so
    # it is always at the very front (highest priority position) even after
    # rephrasing.
    current_prompt = prompt
    max_retries = 3
    for attempt in range(max_retries + 1):
        # Prepend the code-fixed appearance hard constraint at the very front
        # of the final prompt (highest priority position). Reference images
        # are the sole source of character appearance.
        final_prompt = IMAGE_APPEARANCE_HARD_CONSTRAINT + "\n\n" + current_prompt
        # Use .png temp path first (CF returns PNG), convert to webp after
        temp_path = final_path + ".tmp.png"
        try:
            result_path = image_provider.generate(final_prompt, temp_path,
                                                   reference_images=ref_data)
            if result_path:
                # Convert to webp via process_image.py (handles resize + webp + compression)
                stdout, rc = run_script([
                    sys.executable, PROCESS_IMAGE_SCRIPT, temp_path, final_path
                ])
                if rc != 0 or not os.path.exists(final_path):
                    # Fallback: if process_image.py fails, just rename png to webp
                    # (Hugo can serve it, browser will handle)
                    print(f"[image] process_image.py failed, using raw PNG", file=sys.stderr)
                    import shutil
                    shutil.move(temp_path, final_path)
                else:
                    # Clean up temp
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                print(f"[image] Generated: {final_path}")
                return image_filename, final_path
        except RuntimeError as e:
            flagged = getattr(e, "flagged", False)
            if flagged and attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"[image] Flagged by safety filter (attempt {attempt+1}/{max_retries}), "
                      f"rephrasing after {wait}s...", file=sys.stderr)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    # Last retry: use a simplified minimal prompt to minimize
                    # the chance of triggering the safety filter again.
                    current_prompt = _simplify_image_prompt(current_prompt)
                else:
                    new_prompt = _rephrase_image_prompt(rephrase_provider, current_prompt)
                    if new_prompt:
                        current_prompt = new_prompt
                continue
            print(f"[image] Generation failed: {e}", file=sys.stderr)
            break
        except Exception as e:
            print(f"[image] Generation error: {e}", file=sys.stderr)
            break

    # Cleanup temp file if exists
    temp_path = final_path + ".tmp.png"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return None, None


def _rephrase_image_prompt(rephrase_provider, prompt):
    """Rephrase an image prompt to bypass CF safety filter (same scene, different wording)."""
    system_prompt = """You rephrase image generation prompts for the flux-2-klein-4b model.
Rules:
1. Keep the SAME scene/subject/objects
2. Change wording, sentence structure, and ordering substantially
3. Keep it concrete and visual (nouns > adjectives)
4. Keep the art style / lighting description
5. Output ONLY the rephrased prompt, no explanation, no quotes
6. If the original mentions a brand/person name that may trigger filters, replace with a generic but accurate description"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        new_prompt = rephrase_provider.generate(messages, max_tokens=10240, temperature=0.3, enable_thinking=True)
        new_prompt = new_prompt.strip().strip('"').strip("'").strip("`").strip()
        if new_prompt.startswith("```"):
            lines = new_prompt.split("\n")
            new_prompt = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return new_prompt if new_prompt else None
    except Exception as e:
        print(f"[image] Rephrase failed: {e}", file=sys.stderr)
        return None


def _simplify_image_prompt(prompt):
    """Strip a prompt down to minimal safe content for the final 3030 retry.

    Keeps only the core scene/action lines, drops fields that are more likely
    to trigger CF's safety filter (e.g. detailed body descriptions). If the
    labeled-format parse fails, falls back to a generic safe scene.
    """
    keep_fields = ("Action:", "Object:", "Setting:", "Mood:")
    lines = prompt.strip().split("\n")
    kept = [l for l in lines if l.strip().startswith(keep_fields)]
    if kept:
        return "\n".join(kept)
    # Fallback: ultra-minimal generic prompt
    return "Action: characters spending time together indoors\nSetting: cozy room\nMood: calm, warm"


def repack_month_images(month_str):
    """Re-run pack_images.py for the given month after adding new images."""
    stdout, rc = run_script([sys.executable, PACK_IMAGES_SCRIPT, month_str])
    if rc == 0:
        print(f"[image] Repacked {month_str}.tar")
    else:
        print(f"[image] WARNING: pack_images.py failed for {month_str}", file=sys.stderr)
    return rc == 0


# ==================== Image Replacement (Diagnostic) ====================

# KV key prefix for image-replacement requests. The diag KV namespace is
# shared by current and future diagnostic features; each feature gets its
# own prefix under "diag:" so they can be listed/processed independently.
# Examples: diag:replace: (image replacement), diag:reload: (future), etc.
_DIAG_KV_REPLACE_PREFIX = "diag:replace:"


def _get_diag_kv():
    """Build a KVClient for the diagnostic KV namespace.

    Reuses the default Cloudflare account + API token (the diag namespace
    doesn't need its own token) and reads the namespace ID from
    CF_DIAG_KV_ID. Returns None if any env var is missing (the diagnostic
    feature silently disables in environments that haven't configured it).
    """
    account_id = os.environ.get("CF_DEFAULT_ACCOUNT_ID", "")
    api_token = os.environ.get("CF_DEFAULT_API_TOKEN", "")
    namespace_id = os.environ.get("CF_DIAG_KV_ID", "")
    if not (account_id and api_token and namespace_id):
        return None
    try:
        return KVClient(account_id, namespace_id, api_token)
    except ValueError as e:
        print(f"[diag] KV client init failed: {e}", file=sys.stderr)
        return None


def _resolve_image_filename(whisper_id, whisper_data, seq):
    """从 whisper JSON 的 images 字段推导要替换/补的图片文件名。

    优先用 whisper JSON 已有的 images[seq-1]（去掉 /images/ 前缀），
    否则用默认 {whisper_id}-{seq}.webp。
    返回 (filename, images_field_present)。
    """
    images = whisper_data.get("images") or []
    idx = seq - 1
    if 0 <= idx < len(images):
        url = images[idx]
        # url 形如 "/images/2026-06-28-nuonuo-0cd4cf-1.webp"
        filename = url.split("/images/")[-1] if "/images/" in url else url.lstrip("/")
        return filename, True
    return f"{whisper_id}-{seq}.webp", False


def _ext_for_content_type(content_type):
    """根据 content_type 推导临时文件扩展名（给 process_image.py 喂输入用）。"""
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    return ".png"


def apply_image_replacements(now_dt, dry_run=False):
    """拾取 KV 中的待处理配图替换请求并逐个应用。

    流程：KV.list → 按 whisper_id 分组取最新 → 解码图片 → 非webp转webp →
    写 static/images/ → repack 当月 tar → 更新 whisper JSON（补图时加 images 字段）。
    KV key 的删除推迟到 git push 成功之后（由 main() 执行），避免 push 失败时
    本地改动丢失导致替换请求永久丢失。每次 cron 都跑（KV.list 很便宜），无 KV
    配置时静默跳过。

    Returns (changes_made, processed_kv_keys) — processed_kv_keys 是成功应用的
    KV key 列表，由调用方在 push 成功后删除。
    """
    print("\n--- Apply Image Replacements ---")
    kv = _get_diag_kv()
    if kv is None:
        print("[diag] KV not configured (CF_DEFAULT_ACCOUNT_ID/CF_DEFAULT_API_TOKEN/CF_DIAG_KV_ID), skipping")
        return False, []

    try:
        keys = kv.list_keys(prefix=_DIAG_KV_REPLACE_PREFIX)
    except Exception as e:
        print(f"[diag] KV list failed: {e}", file=sys.stderr)
        return False, []

    if not keys:
        print("No pending image replacements")
        return False, []

    print(f"Found {len(keys)} pending image replacement request(s)")

    # 取出所有请求值
    requests_by_whisper = {}  # whisper_id -> (latest_value, all_keys_for_this_whisper)
    for key in keys:
        try:
            raw = kv.get_value(key)
            req = json.loads(raw)
        except Exception as e:
            print(f"[diag] Failed to read KV key {key}: {e}", file=sys.stderr)
            continue
        wid = req.get("whisper_id", "")
        if not wid:
            continue
        # key 格式: diag:replace:{whisper_id}:{ts}，ts 越大越新
        ts_part = key.rsplit(":", 1)[-1]
        try:
            ts = int(ts_part)
        except ValueError:
            ts = 0
        prev = requests_by_whisper.get(wid)
        if prev is None or ts > prev[1]:
            requests_by_whisper[wid] = (req, ts, [])
        # 收集所有 key 以便处理完一起删
        requests_by_whisper[wid][2].append(key)

    if not requests_by_whisper:
        print("No valid image replacement requests")
        return False, []

    if dry_run:
        for wid, (req, _, _) in requests_by_whisper.items():
            print(f"  [DRY RUN] Would replace image for {wid} (seq={req.get('seq',1)})")
        return False, []

    changes_made = False
    processed_keys = []  # KV keys for successfully-applied replacements; deleted only after git push
    for wid, (req, _, all_keys) in requests_by_whisper.items():
        month_str = req.get("month_str", wid[:7])
        seq = req.get("seq", 1)
        image_b64 = req.get("image_base64", "")
        content_type = req.get("content_type", "image/webp")

        print(f"  Processing {wid} (seq={seq})")

        # 校验 whisper 存在
        whisper_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
        if not os.path.exists(whisper_json_path):
            print(f"    Skip: whisper file {month_str}.json not found")
            _delete_keys_silent(kv, all_keys)
            continue
        month_data = load_json(whisper_json_path)
        slug_part = wid[11:]
        whisper_data = month_data.get(slug_part)
        if not whisper_data:
            print(f"    Skip: whisper {wid} not found in {month_str}.json")
            _delete_keys_silent(kv, all_keys)
            continue

        # 推导图片文件名
        image_filename, had_images = _resolve_image_filename(wid, whisper_data, seq)
        final_path = os.path.join(IMAGES_DIR, image_filename)

        # 解码 base64 写临时文件
        try:
            img_bytes = base64.b64decode(image_b64)
        except Exception as e:
            print(f"    Skip: base64 decode failed: {e}")
            _delete_keys_silent(kv, all_keys)
            continue

        os.makedirs(IMAGES_DIR, exist_ok=True)
        is_webp = "webp" in (content_type or "").lower() or image_filename.lower().endswith(".webp")

        if is_webp:
            # 直接写 webp
            try:
                with open(final_path, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"    Skip: write webp failed: {e}")
                _delete_keys_silent(kv, all_keys)
                continue
        else:
            # 非 webp：先写临时文件，再用 process_image.py 转 webp
            ext = _ext_for_content_type(content_type)
            tmp_path = final_path + ".tmp" + ext
            try:
                with open(tmp_path, "wb") as f:
                    f.write(img_bytes)
                # 调 process_image.py 转换
                _, rc = run_script([sys.executable, PROCESS_IMAGE_SCRIPT, tmp_path, final_path])
                if rc != 0 or not os.path.exists(final_path):
                    print(f"    Skip: process_image.py conversion failed")
                    _delete_keys_silent(kv, all_keys)
                    continue
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        # 补图：whisper 原本无 images 字段，加上
        if not had_images:
            images_list = whisper_data.get("images") or []
            new_url = f"/images/{image_filename}"
            if new_url not in images_list:
                # 插到 seq-1 位置或末尾
                while len(images_list) < seq - 1:
                    images_list.append("")
                if seq - 1 < len(images_list):
                    images_list[seq - 1] = new_url
                else:
                    images_list.append(new_url)
                whisper_data["images"] = [x for x in images_list if x]
                month_data[slug_part] = whisper_data
                save_json(whisper_json_path, month_data)
                print(f"    Added images field to {wid}: {whisper_data['images']}")

        # 重打包当月 tar（pack_images.py 会自动从旧 tar补齐其他图 + 重新打包，
        # 且不会用旧 tar 覆盖我们刚写的新文件）
        if not repack_month_images(month_str):
            print(f"    WARNING: repack failed for {month_str}, image written but tar not updated",
                  file=sys.stderr)

        # Defer KV key deletion to main() — only delete after git push
        # succeeds, so a push failure doesn't lose the replacement request.
        processed_keys.extend(all_keys)

        changes_made = True
        print(f"    Replaced image for {wid}: {image_filename}")

    print(f"Image replacements complete: "
          f"{sum(1 for w in requests_by_whisper.values()) } processed, "
          f"changes={'yes' if changes_made else 'no'}")
    return changes_made, processed_keys


def _delete_keys_silent(kv, keys):
    """删除一组 KV key，失败只记录不中断。"""
    for k in keys:
        try:
            kv.delete_key(k)
        except Exception as e:
            print(f"[diag] KV delete failed for {k}: {e}", file=sys.stderr)


# ==================== Content Generation ====================

def load_recent_whispers(count=15):
    """Load recent whispers directly from data files (more reliable than parsing
    timeline text). Returns list of dicts sorted by date descending.
    Each: {"date", "author", "title", "content"}.
    """
    posts = []
    whispers_dir = WHISPERS_DIR
    if not os.path.exists(whispers_dir):
        return posts
    month_files = sorted([f for f in os.listdir(whispers_dir) if f.endswith(".json")],
                         reverse=True)
    for mf in month_files:
        if len(posts) >= count * 2:
            break
        with open(os.path.join(whispers_dir, mf), "r", encoding="utf-8") as f:
            data = json.load(f)
        for slug, w in data.items():
            if isinstance(w, dict) and w.get("date"):
                posts.append({
                    "date": w["date"],
                    "author": w.get("author", ""),
                    "title": w.get("title", ""),
                    "content": w.get("content", ""),
                })
    posts.sort(key=lambda x: x["date"], reverse=True)
    return posts[:count]


def build_recent_topics_summary(recent_whispers, authors_data):
    """Build a compact summary of recent topics per author, to prevent repetition.

    Format:
        ## 最近动态主题（避免重复，不要写相似内容）
        ### doro（Doro）
        - 2026-06-25 12:16 《今天和咕嘎一起晒太阳》— 和咕嘎阳台晒太阳，果汁洒裙子...
        - 2026-06-24 13:04 《周三下午的橘子和奶茶》— 下午摸鱼，橘子奶茶提神...
        ...
    """
    if not recent_whispers:
        return "（暂无历史动态）"

    lines = ["## 各角色最近动态主题（新动态严禁与这些主题雷同/复读，包括场景、事件、用词）"]
    # Group by author, keep most recent 4 per author
    by_author = {}
    for w in recent_whispers:
        by_author.setdefault(w["author"], []).append(w)
    for author, items in by_author.items():
        name = authors_data.get(author, {}).get("name", author)
        lines.append(f"\n### {author}（{name}）最近{min(len(items), 4)}条：")
        for w in items[:4]:
            date_short = w["date"][:16].replace("T", " ")
            snippet = w["content"].replace("\n", " ")[:40]
            lines.append(f"- {date_short} 《{w['title']}》— {snippet}...")
    return "\n".join(lines)


def text_similarity(a, b):
    """Character bigram Jaccard similarity. 0-1, higher = more similar."""
    if not a or not b:
        return 0.0
    def bigrams(s):
        s = s.replace(" ", "").replace("\n", "")
        return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def is_too_similar(new_content, new_author, recent_whispers, threshold=0.45):
    """Check if new content is too similar to that author's recent posts."""
    same_author = [w["content"] for w in recent_whispers if w["author"] == new_author]
    for prev in same_author[:5]:  # check against most recent 5
        sim = text_similarity(new_content, prev)
        if sim >= threshold:
            return True, sim
    return False, 0.0


# ==================== Time Context ====================

def _time_period(hour):
    """Map hour to Chinese time period label."""
    if 0 <= hour < 7:
        return "深夜"
    elif 7 <= hour < 9:
        return "早上"
    elif 9 <= hour < 12:
        return "上午"
    elif 12 <= hour < 14:
        return "中午"
    elif 14 <= hour < 18:
        return "下午"
    elif 18 <= hour < 20:
        return "傍晚"
    elif 20 <= hour < 23:
        return "晚上"
    else:
        return "深夜"


_WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]


def _build_time_context(now_dt, whisper_data=None):
    """构建时间上下文，供所有 prompt 统一使用。

    返回 (now_block, whisper_time_block)：
      - now_block: 当前时间描述（日期+星期+时段），始终返回
      - whisper_time_block: 动态发布时间+距今间隔+时态规则；若 whisper_data
        为空或解析失败则返回空字符串。

    这是时态一致性的核心：模型必须知道"现在是什么时间"和"动态是何时发的"，
    否则会沿用动态内容里的时间表述（如动态说"明天周一"，几天后回复还照抄
    "明天周一"），造成时态错乱。
    """
    weekday = _WEEKDAY_CN[now_dt.weekday()]
    period = _time_period(now_dt.hour)
    now_block = f"当前时间：{now_dt.strftime('%Y-%m-%d %H:%M')} 周{weekday}，{period}"

    if not whisper_data:
        return now_block, ""

    whisper_date_str = whisper_data.get("date", "")
    if not whisper_date_str:
        return now_block, ""

    try:
        whisper_dt = datetime.fromisoformat(whisper_date_str)
        if whisper_dt.tzinfo is None:
            whisper_dt = whisper_dt.replace(tzinfo=TZ_BEIJING)
    except Exception:
        return now_block, ""

    delta = now_dt - whisper_dt
    total_seconds = delta.total_seconds()
    if total_seconds < 0:
        return now_block, ""

    if total_seconds < 3600:
        elapsed = f"{max(1, int(total_seconds / 60))}分钟前"
    elif total_seconds < 86400:
        elapsed = f"{int(total_seconds / 3600)}小时前"
    else:
        days = int(total_seconds / 86400)
        elapsed = f"{days}天前"

    w_weekday = _WEEKDAY_CN[whisper_dt.weekday()]
    w_period = _time_period(whisper_dt.hour)

    whisper_time_block = (
        f"动态发布时间：{whisper_dt.strftime('%Y-%m-%d %H:%M')} 周{w_weekday}，{w_period}\n"
        f"距今：{elapsed}\n"
        "【时态规则——必须严格遵守】回复内容必须基于【当前时间】的时态，"
        "不要照搬动态内容里的时间表述。例如：动态写'明天周一'但当前已是周一，"
        "回复不能再说'明天周一'（应说'今天周一'或'昨晚'等）；动态写'今晚'但"
        "当前已是第二天，回复要根据当前时间调整为'昨晚'/'今早'等。动态里提到"
        "的时间只反映作者发动态那一刻的心境，回复者站在【当前时间】看这条动态。"
    )
    return now_block, whisper_time_block


def build_publish_prompt(characters_md, timeline_text, day_info, now_dt,
                         authors_data, recent_topics_summary):
    """
    Build system and user prompts for whisper generation.
    AI selects the character AND generates content in one call.
    """
    # Day type
    if day_info["type"] == "holiday":
        day_desc = f"法定节假日（{day_info.get('holiday_name', '节日')}）"
    elif day_info["type"] == "weekend":
        day_desc = "周末"
    else:
        day_desc = "工作日"

    # Unified time context (no whisper_data for new posts)
    now_block, _ = _build_time_context(now_dt)
    # Append day type (workday/weekend/holiday) which _build_time_context doesn't cover
    now_block_with_day = f"{now_block}，{day_desc}"

    # Build character list for AI to choose from
    char_list = []
    for char_id, info in authors_data.items():
        char_list.append(f"- {info.get('name', char_id)}（ID: {char_id}）: {info.get('desc', '')}")
    char_list_text = "\n".join(char_list)

    system_prompt = f"""你是"豆包和朋友们的悄悄话"小站的调度AI。这个小站是几个虚拟角色发碎碎念动态的地方，就像QQ空间说说。

角色设定：
{characters_md}

可选角色列表：
{char_list_text}

你的任务：
1. 根据当前时间、场景、最近的动态上下文，选择一个最合适的角色来发新动态
2. 以该角色的身份写一条碎碎念动态

选择角色的原则：
- 根据角色性格和当前场景，谁最自然就选谁
- 避免和最近2条动态的作者重复
- 考虑角色之间的关系和互动，内容要和最近动态逻辑一致，不能前后矛盾
- 如果最近有人在聊某个话题，可以延续或回应，但不要复读

写动态的要求：
1. 长度50-200字，短而精
2. 口语化、轻松、随意，像真人发朋友圈
3. 必须符合所选角色的性格和说话风格
4. 可以带2-5个左右emoji（这个数量不是绝对的，要根据角色设定和动态内容等综合判断），不要太多，也不要太少
5. 内容要符合当前场景（时间场景在用户消息中给出）
6. 【重要】不要和最近动态主题重复或雷同——参考下方"各角色最近动态主题"清单，新动态的场景、事件、关键用词必须与之有明显区别，但也不是绝对，要综合考虑，比如某天的某件事可能上午发个动态，下午也可能发个动态，比如上午遇到一个bug，下午解决了，再比如去旅游，完全可以连发好几个旅游玩耍相关动态，当然我只是举例，需要你综合来考虑。尤其避免复读同一角色的近期内容，但是动态从时间线上以及逻辑上也要连贯合理
7. 角色的标志性爱好是性格的一部分，但不要每次都出现，更不要每条都围绕它写。一个爱好连续出现2次后，第3次必须换别的内容
8. 不要涉及任何真实个人隐私

格式规范（必须严格遵守）：
1. content 必须用换行分段：至少有1个空行（\\n\\n）把内容分成2-4段，不能整段不换行
2. 合理使用标点符号，标点符号直接影响动态的语气感觉，你需要重视这个，比如波浪号"～"是可选语气词，但并不是每条都用，更不要每句句末都加"~"
3. title 是一句话标题，不超过15字，不带书名号

附加元数据（一并输出，用于后续处理）：
- slug: 根据动态内容生成3-5个英文单词的slug，用连字符连接，全小写，概括动态核心内容。例如 "friday-sunset-fried-chicken"
- mood: 角色发这条动态时的心情，从 happy/tired/sad/excited/content/calm/grumpy 中选一个
- topics: 1-2个话题关键词（中文），概括动态主题。例如 ["炸鸡", "周末"]
- mentioned_characters: 动态内容中提到的其他角色ID列表（不含作者自己）。如果没提及其他角色则为空数组 []
- needs_image: 这条动态是否适合配图。有具体场景、物体、动作、视觉元素的动态适合配图（true）；纯情绪/感慨/抽象思考的不适合（false）
- storyline_trigger: 如果这条动态可能触发故事线，返回 {{"type": "comfort_request或minor_conflict", "participants": ["角色ID"]}}。comfort_request: 角色情绪低落/遇到困难需要安慰；minor_conflict: 角色之间有小矛盾/冲突。否则返回 null

输出格式（严格JSON，不要输出任何其他内容、不要markdown代码块）：
{{
  "character": "角色ID（如 doro / feibi / guga / baizi / nuonuo / doubao）",
  "title": "一句话标题，不超过15字",
  "content": "碎碎念正文，50-200字，必须包含换行分段（\\n\\n）",
  "slug": "english-kebab-case-slug",
  "mood": "happy",
  "topics": ["话题1", "话题2"],
  "mentioned_characters": [],
  "needs_image": true,
  "storyline_trigger": null
}}"""

    user_prompt = f"""{now_block_with_day}

{recent_topics_summary}

最近的动态（参考上下文，不要矛盾，不要原样复读）：
{timeline_text}

请选择一个角色并写一条新的碎碎念。注意：这是一个综合性系统性的任务，要全面考虑，我们要的是真实的感觉。

只输出JSON。"""

    return system_prompt, user_prompt


def generate_whisper_content(text_provider, characters_md, timeline_text,
                             day_info, now_dt, authors_data):
    """
    Generate whisper content via AI.
    AI selects character and generates content in one call.
    Includes post-generation similarity check; regenerates once if too similar
    to the chosen character's recent posts.
    Returns {"character": "char_id", "title": "...", "content": "..."} or None.
    """
    recent_whispers = load_recent_whispers(count=15)
    recent_topics_summary = build_recent_topics_summary(recent_whispers, authors_data)

    def _generate_once():
        """Single generation attempt. Returns parsed dict or None."""
        system_prompt, user_prompt = build_publish_prompt(
            characters_md, timeline_text, day_info, now_dt, authors_data,
            recent_topics_summary
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = text_provider.generate(messages, max_tokens=10240, temperature=0.7, enable_thinking=True)
        except Exception as e:
            print(f"AI generation failed: {e}", file=sys.stderr)
            return None

        # Parse JSON from response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        try:
            data = json.loads(response)
            character = data.get("character", "").strip()
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()

            if not character or not title or not content:
                print("AI response missing fields", file=sys.stderr)
                return None
            if character not in authors_data:
                print(f"Warning: AI returned unknown character '{character}'", file=sys.stderr)
                return None

            # Parse model-provided metadata (with fallbacks)
            result = {
                "character": character,
                "title": title,
                "content": content,
                "slug": _sanitize_slug(data.get("slug", ""), character),
                "mood": data.get("mood", "").strip() or None,
                "topics": data.get("topics", []) if isinstance(data.get("topics"), list) else [],
                "mentioned_characters": [
                    c for c in data.get("mentioned_characters", [])
                    if isinstance(c, str) and c in authors_data
                ] if isinstance(data.get("mentioned_characters"), list) else [],
                "needs_image": bool(data.get("needs_image", False)),
                "storyline_trigger": data.get("storyline_trigger"),
            }
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response as JSON: {e}", file=sys.stderr)
            print(f"Response: {response[:200]}", file=sys.stderr)
            return None

    # First attempt
    result = _generate_once()
    if not result:
        return None

    # Similarity check against the chosen character's recent posts
    similar, sim = is_too_similar(result["content"], result["character"], recent_whispers)
    if similar:
        print(f"[repeat-check] Content too similar to recent (sim={sim:.2f}), regenerating once...",
              file=sys.stderr)
        # Bump temperature for the retry to encourage divergence
        retry = _generate_once()
        if retry:
            similar2, sim2 = is_too_similar(retry["content"], retry["character"], recent_whispers)
            if not similar2:
                print(f"[repeat-check] Regeneration OK (sim={sim2:.2f})", file=sys.stderr)
                return retry
            print(f"[repeat-check] Still similar after retry (sim={sim2:.2f}), using retry anyway",
                  file=sys.stderr)
            return retry
        # retry failed to parse, fall back to original
        print("[repeat-check] Retry failed to parse, using original", file=sys.stderr)
    return result


def generate_slug(character_id, title):
    """Generate a slug from character ID and title (fallback when model doesn't provide one)."""
    import hashlib
    hash_str = hashlib.md5(title.encode("utf-8")).hexdigest()[:6]
    return f"{character_id}-{hash_str}"


def _sanitize_slug(raw_slug, character_id=""):
    """Sanitize a model-provided slug to kebab-case. Falls back to None if empty."""
    if not raw_slug or not isinstance(raw_slug, str):
        return None
    import re
    slug = raw_slug.strip().lower()
    slug = re.sub(r'[^a-z0-9-]', '-', slug)
    slug = re.sub(r'-{2,}', '-', slug)
    slug = slug.strip('-')
    if len(slug) < 3:
        return None
    # Prepend character_id for namespacing
    if character_id and not slug.startswith(character_id):
        slug = f"{character_id}-{slug}"
    return slug


# ==================== Multi-character reply selection ====================

BEST_FRIENDS = {
    "guga": "doro", "doro": "guga",
    "feibi": "nuonuo", "nuonuo": "feibi",
}


def _get_friendship_weight(aid_a, aid_b, character_states):
    """Get how likely aid_a would interact with aid_b's content."""
    # Best friends get highest weight
    if BEST_FRIENDS.get(aid_a) == aid_b or BEST_FRIENDS.get(aid_b) == aid_a:
        return 1.5
    # Base weight
    return 1.0


def _pick_friend(whisper_author_id, existing_replies, authors_data, character_states):
    """Pick a friend to reply to a user comment on the author's whisper."""
    # Best friend 60%优先
    best_friend_id = BEST_FRIENDS.get(whisper_author_id)
    if best_friend_id and best_friend_id in authors_data and random.random() < 0.6:
        reply_count = sum(1 for r in existing_replies or [] if r.get("author") == best_friend_id)
        if reply_count < 3:
            return best_friend_id

    others = [aid for aid in authors_data if aid != whisper_author_id]
    weights = []
    for aid in others:
        w = _get_friendship_weight(whisper_author_id, aid, character_states)
        reply_count = sum(1 for r in existing_replies or [] if r.get("author") == aid)
        w *= (0.5 ** reply_count)
        weights.append(max(w, 0.1))

    return random.choices(others, weights=weights, k=1)[0]


# ==================== Reply Generation ====================

def generate_smart_reply(text_provider, whisper_content, whisper_author_id,
                         whisper_author_name, user_reply_content, user_nickname,
                         characters_md, authors_data, character_states,
                         existing_replies, timeline_text, now_dt, whisper_data=None):
    """Consolidated reply generation: one model call decides who replies + generates content.

    Replaces _select_reply_character (keyword-based) + generate_reply (per-character calls).
    Returns list of (char_id, char_name, role_type, content) tuples.
    """
    # Pre-select a potential friend using weighted selection (with reply decay)
    friend_id = _pick_friend(whisper_author_id, existing_replies, authors_data, character_states)
    friend_name = authors_data.get(friend_id, {}).get("name", friend_id) if friend_id else ""

    author_hint = _get_character_state_hint(whisper_author_id, whisper_author_name, character_states)
    friend_hint = ""
    if friend_id:
        friend_hint = _get_character_state_hint(friend_id, friend_name, character_states)

    now_block, whisper_time_block = _build_time_context(now_dt, whisper_data)

    system_prompt = f"""你是一个扮演角色的AI，在"豆包和朋友们的悄悄话"小站上回复评论。

角色设定：
{characters_md}

情境：{whisper_author_name}发了一条动态，用户"{user_nickname}"评论了。

{author_hint}
{friend_hint}

你需要决定谁来回复，并生成回复内容。可选方案：
1. 作者本人回复（role: "author"）
2. 朋友{friend_name}帮忙回复（role: "friend"）—— 适合调侃、帮腔、或作者不在时
3. 两人都回复 —— 适合有趣的评论、向所有人提问的情况

根据评论内容选择最自然的方式。通常只选1或2即可，选3要评论确实有趣。

要求：
1. 回复要符合角色的性格和说话风格
2. 口语化、自然，像真实朋友聊天
3. 如果是朋友回复，可以调侃作者或和用户互动
4. 回复长度10-80字
5. 不要涉及用户隐私
6. 【时态】回复站在【当前时间】的视角，不要照搬动态内容里的时间词。

输出格式（严格JSON，不要markdown代码块）：
{{"replies": [{{"character": "角色ID", "role": "author或friend", "content": "回复内容"}}]}}"""

    user_prompt = f"""{now_block}
{whisper_time_block}

动态作者：{whisper_author_name}（ID: {whisper_author_id}）
动态内容：{whisper_content}
动态时间线：{timeline_text}

用户评论：{user_reply_content}

可选回复者：{whisper_author_name}（作者），{friend_name}（朋友）
请决定谁来回复并生成回复内容。只输出JSON。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.7, enable_thinking=True)
    except Exception as e:
        print(f"Smart reply generation failed: {e}", file=sys.stderr)
        return []

    if not response:
        return []

    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            print(f"Failed to parse smart reply JSON: {response[:200]}", file=sys.stderr)
            return []

    replies_data = data.get("replies", []) if isinstance(data, dict) else []
    valid_ids = {whisper_author_id, friend_id} - {None}
    result = []
    for r in replies_data:
        if not isinstance(r, dict):
            continue
        char_id = r.get("character", "").strip()
        role = r.get("role", "author").strip()
        content = r.get("content", "").strip()
        if not char_id or not content:
            continue
        if char_id not in valid_ids:
            continue
        char_name = authors_data.get(char_id, {}).get("name", char_id)
        # Friend reply decay: skip if already replied 3+ times
        if role == "friend":
            reply_count = sum(1 for er in existing_replies or [] if er.get("author") == char_id)
            if reply_count >= 3:
                continue
        result.append((char_id, char_name, role, content))

    # Fallback: if model returned nothing valid, author replies
    if not result:
        author_name = authors_data.get(whisper_author_id, {}).get("name", whisper_author_id)
        # Try a simple generate
        try:
            fallback_response = text_provider.generate(
                [{"role": "system", "content": f"你是{whisper_author_name}，在回复动态评论。口语化回复，10-80字。只输出回复内容。"},
                 {"role": "user", "content": f"动态：{whisper_content}\n用户评论：{user_reply_content}\n请回复："}],
                max_tokens=10240, temperature=0.7, enable_thinking=True
            )
            if fallback_response:
                result.append((whisper_author_id, author_name, "author", fallback_response.strip()))
        except Exception:
            pass

    return result


# ==================== Character Interactions ====================

# ==================== Storylines (relationship arcs) ====================

STORYLINE_SAD_PATTERNS = ["好烦", "委屈", "不开心", "难过", "生气", "被抢", "被骂", "被说", "好惨", "好累"]
STORYLINE_BLAME_PATTERNS = ["都怪", "讨厌", "不想理", "太过分了", "再也不"]


def _detect_storyline_triggers(whisper_data, whisper_id, authors_data):
    """Detect if a whisper triggers or advances a storyline.
    Uses model-provided storyline_trigger if available, falls back to keyword matching.
    Returns (triggered, sl_type, participants) or (False, None, []).
    """
    content = whisper_data.get("content", "")
    author = whisper_data.get("author", "")

    # Primary: use model-provided storyline_trigger (stored at publish time)
    model_trigger = whisper_data.get("storyline_trigger")
    if model_trigger and isinstance(model_trigger, dict):
        sl_type = model_trigger.get("type", "")
        participants = model_trigger.get("participants", [])
        if sl_type and participants:
            # Validate participants are known characters
            valid_participants = [p for p in participants if p in authors_data or p == author]
            if valid_participants:
                return True, sl_type, valid_participants

    # Fallback: keyword-based detection
    if any(p in content for p in STORYLINE_SAD_PATTERNS):
        return True, "comfort_request", [author]

    for p in STORYLINE_BLAME_PATTERNS:
        if p in content:
            for aid, info in authors_data.items():
                if aid != author and info.get("name", "") in content:
                    return True, "minor_conflict", [author, aid]

    return False, None, []


def _get_storyline_context(storylines, whisper_author_id):
    """Build storyline context string for prompts. Empty if no relevant storylines."""
    active = storylines.get("active", []) if storylines else []
    if not active:
        return ""

    relevant = [sl for sl in active if whisper_author_id in sl.get("participants", [])]
    if not relevant:
        return ""

    phase_cn = {"active": "还在别扭中", "de-escalating": "关系在缓和", "resolved": "已经和好了"}
    lines = ["【进行中的故事情节】"]
    for sl in relevant:
        lines.append(f"- {sl.get('summary', '')}（{phase_cn.get(sl.get('phase', ''), '')}）")
    lines.append("回复时注意保持情绪一致，不要和 storyline 矛盾。\n")
    return "\n".join(lines)


def _evolve_storylines(storylines, now_dt):
    """Advance active storylines over time. Caps completed list to last 15."""
    active = storylines.get("active", [])
    if not active:
        return

    for sl in active:
        started_str = sl.get("started_at", "")
        if not started_str:
            continue
        try:
            started = datetime.fromisoformat(started_str)
        except (ValueError, TypeError):
            continue

        hours = (now_dt - started).total_seconds() / 3600

        if sl["phase"] == "active" and hours > 48:
            sl["phase"] = "de-escalating"
            print(f"[storyline] {sl['id']}: active → de-escalating")

        elif sl["phase"] == "de-escalating" and hours > 72:
            sl["phase"] = "resolved"
            sl["resolved_at"] = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
            storylines.setdefault("completed", []).append(sl)
            print(f"[storyline] {sl['id']}: resolved!")

    storylines["active"] = [sl for sl in active if sl["phase"] != "resolved"]

    # B2: Cap completed storylines to last 15 to prevent unbounded growth
    completed = storylines.get("completed", [])
    if len(completed) > 15:
        storylines["completed"] = completed[-15:]
        print(f"[storyline] Trimmed completed list from {len(completed)} to 15")


# B1: Cache for whisper data to avoid repeated disk reads in cross-reference
_WHISPER_CACHE = None
_WHISPER_CACHE_LOADED = False


def _load_whisper_cache():
    """Load all whisper month JSONs into memory cache. Called once per run."""
    global _WHISPER_CACHE, _WHISPER_CACHE_LOADED
    if _WHISPER_CACHE_LOADED:
        return _WHISPER_CACHE
    _WHISPER_CACHE = {}
    if os.path.exists(WHISPERS_DIR):
        for mf in sorted(os.listdir(WHISPERS_DIR), reverse=True):
            if not mf.endswith(".json"):
                continue
            month_key = mf.replace(".json", "")
            path = os.path.join(WHISPERS_DIR, mf)
            try:
                _WHISPER_CACHE[month_key] = load_json(path)
            except Exception:
                continue
    _WHISPER_CACHE_LOADED = True
    print(f"[cache] Loaded {_WHISPER_CACHE and len(_WHISPER_CACHE) or 0} month files into whisper cache")
    return _WHISPER_CACHE


def _invalidate_whisper_cache():
    """Invalidate the whisper cache (call after publishing a new whisper)."""
    global _WHISPER_CACHE_LOADED
    _WHISPER_CACHE_LOADED = False


def _get_cross_reference_context(whisper_data, whisper_author_id, authors_data,
                                 now_dt, existing_replies):
    """Build cross-reference context showing author's recent posts and reply activity.
    Uses in-memory cache (B1) instead of reading files from disk each call.
    Returns a string to inject into prompts, or empty string if nothing relevant.
    """
    author_nick = authors_data.get(whisper_author_id, {}).get("name", whisper_author_id)

    # B1: Use cached whisper data instead of reading from disk
    cache = _load_whisper_cache()
    author_posts = []
    whisper_date = whisper_data.get("date", "")
    for month_key, month_data in cache.items():
        for slug, w in month_data.items():
            if w.get("author") == whisper_author_id and w.get("date", "") < whisper_date:
                author_posts.append((slug, w, month_key))
    # Sort by date descending, take 3 most recent
    author_posts.sort(key=lambda x: x[1].get("date", ""), reverse=True)
    author_posts = author_posts[:3]

    if not author_posts:
        return ""

    lines = [f"{author_nick}的其他动态（可自然提及）："]
    for slug, w, ym in author_posts:
        snippet = w.get("content", "").replace("\n", " ")[:25]
        lines.append(f"- 《{w.get('title', '')}》{snippet}...")
    lines.append("（如果自然，可以在回复中提及以上动态，但不强制）\n")
    return "\n".join(lines)


def build_interaction_prompt(whisper_data, whisper_author_name, existing_replies,
                             characters_md, authors_data, now_dt, candidate_chars,
                             character_states=None):
    """Build prompt for generating character-to-character interactions.

    candidate_chars: list of (char_id, nickname) tuples that are ALLOWED to
    participate this round (already filtered for variety/cooldown). The
    whisper author is included in this list if they should reply to their
    own commenters this round.
    """
    whisper_author_id = whisper_data.get("author", "")

    # Build existing replies text with floor numbers
    replies_text = "（暂无回复）"
    if existing_replies:
        lines = []
        for r in existing_replies:
            floor = r.get("floor", "?")
            nick = r.get("nickname", "?")
            content = r.get("content", "")
            reply_to = r.get("reply_to", "")
            if reply_to:
                lines.append(f"#{floor} {nick}（回复@{reply_to}）: {content}")
            else:
                lines.append(f"#{floor} {nick}: {content}")
        replies_text = "\n".join(lines)

    # Character state hints for each candidate
    state_hints_lines = []
    for char_id, nick in candidate_chars:
        hint = _get_character_state_hint(char_id, nick, character_states)
        if hint:
            state_hints_lines.append(hint)

    # Build character list from the PRE-FILTERED candidates (includes author
    # if they should participate this round). This prevents "全员到齐".
    char_list = []
    for char_id, nick in candidate_chars:
        marker = "（动态作者，可下场回复评论者）" if char_id == whisper_author_id else ""
        char_list.append(f"- {nick}（ID: {char_id}）{marker}")
    char_list_text = "\n".join(char_list) if char_list else "（无）"

    # Per-character voice constraints (P4) - hard rules to prevent homogenization
    voice_rules = '''【角色口吻硬约束——必须严格遵守，否则角色声音会串味】
- 白子：极简冷萌风。每条回复≤12字，多用句号结尾，几乎不用"～"和emoji。例："嗯，不错。" / "下次一起。" / "甜的。"
- 咕嘎：必带"咕咕嘎"或"咕嘎"或"咕咕嘎嘎"或"咕咕咕嘎嘎"等这些咕嘎不同组合的口头禅（每条至少1次），语气活泼，可多用emoji。例："咕咕嘎嘎～我也要！"
- 菲比：必带"菲比啾比"口头禅（每条1次），感叹号多，元气满满。例："菲比啾比～这个超棒！"
- 豆包：大姐姐关怀口吻，温和，会用"姐姐"自称，少用"～"。例："姐姐给你留了，快来吃。"
- Doro：可用emoji，橘子/欧润吉梗可选（不必每条都提），语气软萌。例："好呀好呀🐶等我！"
- 糯糯：游戏宅口吻，比如会跑题提到打游戏，语气随意。例："刚打完一局，我也想吃！"
emoji和标点符号不是绝对的规则，你要系统性理解这些角色，合理使用。以上的例子只是举例，再次强调，只是举例，需要你全面系统性考虑这个任务'''

    # Cross-reference context: author's recent posts with reply counts
    cross_ref_lines = _get_cross_reference_context(
        whisper_data, whisper_author_id, authors_data, now_dt, existing_replies
    )

    # Storyline context for the whisper author
    storyline_block = _get_storyline_context(
        character_states.get("storylines", {}) if character_states else {},
        whisper_author_id
    )

    state_hints_block = ""
    if state_hints_lines:
        state_hints_block = "各角色当前状态：\n" + "\n".join(state_hints_lines) + "\n\n"

    system_prompt = f"""你是"豆包和朋友们的悄悄话"小站的角色互动生成器。朋友们会看彼此的动态，自然地评论互动。

角色设定：
{characters_md}

{voice_rules}

{state_hints_block}{storyline_block}互动原则（核心——决定真实感）：
1. 【生成数量】本次只生成1-3条回复。不要追求全员到齐，只从下方"可选角色"里挑人。平淡动态可能只1条，有话题的最多3条。
2. 【作者下场】动态作者本人也可以参与！如果有人评论了动态、尤其是对作者说了话/调侃/提问，作者应该回复那条评论（带reply_to+floor）。朋友来你朋友圈评论，你总得回一句。
3. 【接话链——必须】如果已有回复里有人说了有意思的话，优先接话（带reply_to+floor），而不是每条都回复动态本身。一批回复里至少1条要接话，全部回复动态本身=失败。
4. 【部分参与】不要把可选角色全部用上。从候选里挑1-3个最自然的（根据动态内容、角色关系、谁会感兴趣），其余这次不出现。
5. 【口吻差异】严格按上方"角色口吻硬约束"写，每个角色声音必须不同。比如白子不能长篇大论等，咕嘎喜欢说口头禅等等，你要全面系统性考虑。
6. 回复长度10-80字（白子除外，可短至5字），口语化、轻松
7. 不要涉及隐私
8. 【语气词】合理使用，我们的核心目标是真实的感觉，符合角色设定和整个大家庭的氛围。口头禅只归角色本人用，其他人不模仿。

【reply_to 字段规则——必须严格遵守】
- 直接回复动态本身（OP）：reply_to填空字符串""、reply_to_floor填0
- 回复某条已有评论：reply_to填该评论者昵称、reply_to_floor填楼层号
- 判断依据：对整条动态发感慨/捧场→回复动态；针对某条评论接话/调侃/追问→回复评论
- 作者回复评论者时，必须带reply_to+floor指向那条评论

输出格式（严格JSON数组，只输出JSON）：
[{{"author": "角色ID", "nickname": "角色名", "content": "回复内容", "reply_to": "回复对象昵称或空字符串", "reply_to_floor": 楼层号或0}}]"""

    now_block, whisper_time_block = _build_time_context(now_dt, whisper_data)

    user_prompt = f"""{now_block}
{whisper_time_block}

动态作者：{whisper_author_name}
动态内容：{whisper_data.get('content', '')}

已有回复：
{replies_text}

{cross_ref_lines}
本次可选角色（只从中挑1-3个，不要全用；作者可下场回复评论者）：
{char_list_text}

请生成1-3条角色互动回复。记住：至少1条接话（带reply_to），作者可参与，口吻差异要明显，时态基于【当前时间】。只输出JSON数组。"""

    return system_prompt, user_prompt


def generate_character_interactions(text_provider, whisper_data, whisper_author_name,
                                    existing_replies, characters_md, authors_data, now_dt,
                                    candidate_chars, character_states=None):
    """Generate character-to-character replies via AI.

    candidate_chars: list of (char_id, nickname) tuples allowed this round
    (pre-filtered for variety/cooldown; includes whisper author if they
    should reply to commenters this round).

    Returns list of reply dicts or None.
    """
    system_prompt, user_prompt = build_interaction_prompt(
        whisper_data, whisper_author_name, existing_replies,
        characters_md, authors_data, now_dt, candidate_chars,
        character_states
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.7, enable_thinking=True)
    except Exception as e:
        print(f"Character interaction generation failed: {e}", file=sys.stderr)
        return None

    if not response:
        return None

    response = response.strip()
    # Strip markdown code fence
    if response.startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        replies = json.loads(response)
    except json.JSONDecodeError:
        # Try stripping control chars
        repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response)
        try:
            replies = json.loads(repaired)
        except json.JSONDecodeError:
            print(f"Failed to parse interaction JSON: {response[:200]}", file=sys.stderr)
            return None

    if not isinstance(replies, list):
        return None

    # Validate and clean up replies
    whisper_author_id = whisper_data.get("author", "")
    whisper_author_nick = authors_data.get(whisper_author_id, {}).get("name", whisper_author_id)
    nick_to_id = {info.get("name", aid): aid for aid, info in authors_data.items()}
    id_to_nick = {aid: info.get("name", aid) for aid, info in authors_data.items()}
    # Map existing floors to their authors, to validate reply_to references
    floor_author_map = {}
    # Map nickname/id -> latest floor, for inferring reply_to when the AI
    # leaves it blank but the content clearly addresses an existing replier
    # by name. The AI is unreliable at filling reply_to, so we backfill it
    # deterministically: if a reply's content mentions a prior replier's
    # nickname (and that replier isn't the author themselves), treat it as a
    # reply to that person's most recent floor.
    nick_to_floor = {}
    for r in existing_replies or []:
        f = r.get("floor")
        if f is not None:
            floor_author_map[f] = r.get("author", "") or r.get("nickname", "")
            nick = r.get("nickname", "")
            if nick:
                nick_to_floor[nick] = f
            aid = r.get("author", "")
            if aid:
                nick_to_floor[aid] = f
    # Existing reply contents (normalized) for dedup
    existing_contents = set()
    for r in existing_replies or []:
        c = r.get("content", "").strip().replace(" ", "").replace("\n", "")
        if c:
            existing_contents.add(c)

    # Set of candidate char_ids allowed this round (for filtering AI output)
    candidate_ids = {cid for cid, _ in candidate_chars}

    def _normalize_reply_to(rt):
        """Normalize reply_to to the display nickname. Accepts nickname or id."""
        if not rt:
            return ""
        if rt in id_to_nick:        # e.g. "doro" -> "Doro"
            return id_to_nick[rt]
        if rt in nick_to_id:        # already a nickname
            return rt
        return rt                   # unknown (e.g. anonymous user), keep as-is

    valid_replies = []
    seen_contents = set()           # dedup within this batch too
    for idx, r in enumerate(replies):
        if not isinstance(r, dict):
            continue
        author_id = r.get("author", "")
        nickname = r.get("nickname", "")
        content = r.get("content", "").strip()
        if not author_id or not content:
            continue
        # P0: author CAN now participate. Only filter out chars NOT in this
        # round's candidate list (prevents "全员到齐" from AI over-generating).
        if author_id not in candidate_ids:
            print(f"[reply-filter] {author_id} not in this round's candidates, skipping",
                  file=sys.stderr)
            continue
        # Ensure nickname matches author_id
        if author_id in authors_data:
            nickname = authors_data[author_id].get("name", nickname)
        # Dedup
        norm_content = content.replace(" ", "").replace("\n", "")
        if norm_content in existing_contents or norm_content in seen_contents:
            print(f"[reply-dedup] Skipping duplicate reply from {nickname}: {content[:30]}...",
                  file=sys.stderr)
            continue
        seen_contents.add(norm_content)
        existing_contents.add(norm_content)

        reply_to = r.get("reply_to", "")
        reply_to_floor = r.get("reply_to_floor", 0)
        # Normalize reply_to to display nickname
        reply_to = _normalize_reply_to(reply_to)
        # Rule: replying directly to the whisper (OP) must NOT carry reply_to.
        # Only clear if AI filled reply_to with whisper author's name/id BUT
        # the referenced floor isn't actually one of OP's replies. Do NOT
        # block the author from legitimately replying to a commenter.
        if reply_to and (
            reply_to == whisper_author_nick
            or reply_to == whisper_author_id
            or reply_to.lower() == whisper_author_id.lower()
        ):
            floor_is_op_reply = (
                reply_to_floor
                and floor_author_map.get(reply_to_floor, "") == whisper_author_id
            )
            if not floor_is_op_reply:
                # AI meant to reply to the whisper itself, not an OP reply.
                reply_to = ""
                reply_to_floor = 0
        # Deterministic backfill: if the AI left reply_to blank (or it was
        # cleared above as an OP-reply), scan the content for mentions of a
        # prior replier's nickname. If exactly one prior replier is mentioned
        # and it isn't the author themselves, treat this as a reply to that
        # person's latest floor. This catches the common failure mode where
        # the model writes "糯糯，..." but forgets to fill reply_to.
        if not reply_to:
            mentioned = []
            for nick, floor in nick_to_floor.items():
                if nick == nickname:
                    continue  # don't reply to self
                if nick and nick in content:
                    mentioned.append((nick, floor))
            if len(mentioned) == 1:
                reply_to, reply_to_floor = mentioned[0]
        valid_replies.append({
            "nickname": nickname,
            "content": content,
            "author": author_id,
            "reply_to": reply_to if reply_to else "",
            "reply_to_floor": reply_to_floor if reply_to_floor else 0,
        })

    # P1: timestamps reflect ACTUAL generation time (now_dt), NOT the whisper's
    # publish time. Replies are generated by the runner NOW (cron-scheduled),
    # so their timestamps must be near now. The old logic based timestamps on
    # whisper_data["date"], which pinned all replies within minutes of publish
    # time even when the runner actually ran hours later — producing absurd
    # times (e.g. a morning whisper with replies about "evening stew" all
    # timestamped at 07:30). Constraints:
    #   - later than the latest existing reply (preserves chronological order)
    #   - no later than now (no future timestamps)
    #   - naturally staggered across the batch
    if valid_replies:
        latest_existing_ts = ""
        for r in existing_replies or []:
            ts = r.get("timestamp", "")
            if ts and ts > latest_existing_ts:
                latest_existing_ts = ts
        try:
            last_reply_dt = (
                datetime.fromisoformat(latest_existing_ts) if latest_existing_ts else None
            )
            if last_reply_dt and last_reply_dt.tzinfo is None:
                last_reply_dt = last_reply_dt.replace(tzinfo=TZ_BEIJING)
        except Exception:
            last_reply_dt = None

        # Stagger backward from now: last reply closest to now (most recent),
        # earlier replies precede it. Each gap 1-3 min (front-end displays at
        # minute precision, so second-level gaps would show identical times).
        spans = [random.randint(1, 3) for _ in valid_replies]
        total_back = sum(spans) + random.randint(0, 2)
        current_dt = now_dt - timedelta(minutes=total_back)
        # Must not be earlier than the latest existing reply (preserve order)
        if last_reply_dt and current_dt < last_reply_dt:
            current_dt = last_reply_dt + timedelta(minutes=random.randint(1, 3))
        for reply, span in zip(valid_replies, spans):
            if current_dt > now_dt:
                current_dt = now_dt
            reply["timestamp"] = current_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
            current_dt = current_dt + timedelta(minutes=span)

    return valid_replies if valid_replies else None


def _select_candidate_chars(whisper_author_id, existing_replies, authors_data, now_dt):
    """P3: Select 2-4 candidate characters for this interaction round.

    Logic:
    - Always include 2-3 OTHER characters (not the author), randomly chosen
      with a cooldown: characters who already replied in existing_replies are
      less likely to be picked again (but can be, to allow接话).
    - With ~50% probability, include the whisper author too (so they can
      reply to their commenters). Skip if there are no existing replies yet
      (author has no one to reply to).
    - Returns list of (char_id, nickname) tuples.
    """
    all_chars = [(aid, info.get("name", aid)) for aid, info in authors_data.items()]
    other_chars = [(aid, nick) for aid, nick in all_chars if aid != whisper_author_id]
    if not other_chars:
        return []

    # Count how many times each char already replied in existing_replies
    reply_counts = {}
    for r in existing_replies or []:
        aid = r.get("author", "")
        if aid:
            reply_counts[aid] = reply_counts.get(aid, 0) + 1

    # Build weighted list: chars who replied less get higher weight
    weighted = []
    for aid, nick in other_chars:
        cnt = reply_counts.get(aid, 0)
        # Weight: 1.0 for 0 replies, 0.5 for 1, 0.25 for 2, 0.1 for 3+
        w = 1.0 / (2 ** cnt) if cnt < 3 else 0.1
        weighted.append((aid, nick, w))

    # Pick 2-3 other chars using weighted random without replacement
    n_others = random.randint(2, 3)
    n_others = min(n_others, len(weighted))
    selected = []
    pool = list(weighted)
    for _ in range(n_others):
        if not pool:
            break
        weights = [w for _, _, w in pool]
        total = sum(weights)
        if total <= 0:
            pick = random.choice(pool)
        else:
            r = random.random() * total
            cum = 0
            pick = pool[-1]
            for item in pool:
                cum += item[2]
                if r <= cum:
                    pick = item
                    break
        selected.append((pick[0], pick[1]))
        pool.remove(pick)

    # With ~50% prob, add the author IF there are existing replies for them
    # to respond to (author下场回复评论者)
    if existing_replies and random.random() < 0.5:
        author_nick = authors_data.get(whisper_author_id, {}).get("name", whisper_author_id)
        # Only add author if they haven't already replied too much
        author_reply_count = reply_counts.get(whisper_author_id, 0)
        if author_reply_count < 3:
            selected.append((whisper_author_id, author_nick))

    return selected


def do_character_interactions(config, d1_client, text_provider, now_dt, dry_run=False):
    """Generate character-to-character interactions for recent whispers lacking replies.

    P5: window extended to 72h; candidates selected by age+reply-count weighting
        (not just newest 3) so older whispers still get interactions.
    P3: per-whisper candidate chars pre-filtered (2-4, including author ~50%
        of the time when there are existing replies).
    """
    print("\n--- Character Interactions ---")

    state = d1_client.get_state()
    last_run = state.get("last_run", {}).get("character_interactions", "")
    if not last_run:
        last_run = "2026-06-01T00:00:00+08:00"

    now_str = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    random_offset = state.get("next_random_offset", {}).get("character_interactions", 0)

    trigger_result = check_trigger("character_interactions", last_run, now_str, random_offset)
    print(f"Trigger check: {trigger_result.get('trigger', False)} - {trigger_result.get('reason', '')}")

    if not trigger_result.get("trigger", False):
        print("Character interactions: not triggered, skipping")
        return False

    # Load characters.md and authors data
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}
    character_states = state.get("character_states", {})

    # P5: window extended to 72h (was 48h) for cross-day accumulation
    cutoff = now_dt - timedelta(hours=72)
    min_age = now_dt - timedelta(hours=1)  # at least 1h old

    candidates = []
    # Check current and previous month
    for month_offset in [0, 1]:
        check_dt = now_dt - timedelta(days=30 * month_offset)
        month_str = check_dt.strftime("%Y-%m")
        whisper_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
        if not os.path.exists(whisper_json_path):
            continue

        month_whispers = load_json(whisper_json_path)
        for slug, w in month_whispers.items():
            date_str = w.get("date", "")
            if not date_str:
                continue
            try:
                w_dt = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                continue

            # Must be within 72h, at least 1h old
            if w_dt < cutoff or w_dt > min_age:
                continue

            whisper_id = f"{w_dt.strftime('%Y-%m-%d')}-{slug}"

            # Check existing character replies
            reply_file = os.path.join(REPLIES_DIR, f"{month_str}.json")
            existing = load_month_file(reply_file)
            existing_replies = existing.get(whisper_id, [])
            char_reply_count = sum(1 for r in existing_replies if r.get("author", ""))

            # P1: raised cap from 5 to 8 so older whispers keep accumulating
            if char_reply_count < 8:
                candidates.append({
                    "whisper_id": whisper_id,
                    "whisper_data": w,
                    "month_str": month_str,
                    "existing_replies": existing_replies,
                    "char_reply_count": char_reply_count,
                    "date": w_dt,
                })

    if not candidates:
        print("No whispers need interactions")
        state["last_run"]["character_interactions"] = now_str
        new_offset = random.randint(0, 30)
        state["next_random_offset"]["character_interactions"] = new_offset
        d1_client.save_state(state)
        return False

    # P5: weighted selection instead of "newest 3". Weight = newer + fewer replies.
    # This gives older/under-replied whispers a chance instead of starving them.
    def _candidate_weight(c):
        age_hours = (now_dt - c["date"]).total_seconds() / 3600
        # Newer whispers get higher weight, but decay slowly over 72h
        age_factor = max(0.1, 1.0 - age_hours / 96)
        # Whispers with fewer replies get higher weight
        reply_factor = 1.0 / (1 + c["char_reply_count"])
        return age_factor * reply_factor

    candidates.sort(key=lambda x: x["date"], reverse=True)
    # Take up to 3, but use weighted random among the top ~6 to add variety
    pool = candidates[:6]
    if len(pool) > 3:
        weights = [_candidate_weight(c) for c in pool]
        total = sum(weights)
        if total > 0:
            selected_indices = []
            for _ in range(3):
                remaining = [i for i in range(len(pool)) if i not in selected_indices]
                if not remaining:
                    break
                w = [weights[i] for i in remaining]
                s = sum(w)
                if s <= 0:
                    selected_indices.append(random.choice(remaining))
                else:
                    r = random.random() * s
                    cum = 0
                    for idx in remaining:
                        cum += weights[idx]
                        if r <= cum:
                            selected_indices.append(idx)
                            break
            candidates = [pool[i] for i in selected_indices]
        else:
            candidates = pool[:3]
    else:
        candidates = pool

    print(f"Found {len(candidates)} whispers needing interactions")

    if dry_run:
        for c in candidates:
            print(f"  [DRY RUN] Would interact: {c['whisper_id']} ({c['char_reply_count']} char replies)")
        return False

    total_new_replies = 0
    for c in candidates:
        whisper_id = c["whisper_id"]
        w_data = c["whisper_data"]
        author_id = w_data.get("author", "")
        author_name = get_author_nickname(author_id, authors_data)
        existing_replies = c["existing_replies"]

        print(f"  Processing {whisper_id} ({c['char_reply_count']} existing char replies)")

        # P3: pre-filter candidate chars for this round (2-4, maybe incl. author)
        candidate_chars = _select_candidate_chars(
            author_id, existing_replies, authors_data, now_dt
        )
        if not candidate_chars:
            print(f"    No candidate chars available, skipping")
            continue
        cand_names = [nick for _, nick in candidate_chars]
        print(f"    Candidates this round: {cand_names}")

        new_replies = generate_character_interactions(
            text_provider, w_data, author_name, existing_replies,
            characters_md, authors_data, now_dt, candidate_chars,
            character_states
        )

        if not new_replies:
            print(f"    No valid interactions generated")
            continue

        # Write to data/replies/*.json
        reply_file = os.path.join(REPLIES_DIR, f"{c['month_str']}.json")
        add_replies(reply_file, whisper_id, new_replies)
        total_new_replies += len(new_replies)
        for r in new_replies:
            reply_to_info = f" (reply to {r['reply_to']}#{r['reply_to_floor']})" if r.get("reply_to") else ""
            print(f"    + {r['nickname']}: {r['content'][:40]}...{reply_to_info}")

        # Storyline detection for this whisper
        triggered, sl_type, participants = _detect_storyline_triggers(w_data, whisper_id, authors_data)
        if triggered:
            storylines = state.setdefault("storylines", {"active": [], "completed": []})
            # Check if already active
            existing = [sl for sl in storylines["active"] if sl.get("trigger_whisper_id") == whisper_id]
            if not existing:
                new_sl = {
                    "id": f"{sl_type}-{whisper_id}",
                    "type": sl_type,
                    "phase": "active",
                    "participants": participants,
                    "trigger_whisper_id": whisper_id,
                    "started_at": now_str,
                    "summary": w_data.get("content", "")[:40],
                    "escalation_level": 1,
                }
                storylines["active"].append(new_sl)
                print(f"[storyline] New {sl_type} started: {new_sl['id']}")

    # Update state
    state["last_run"]["character_interactions"] = now_str
    new_offset = random.randint(0, 30)
    state["next_random_offset"]["character_interactions"] = new_offset
    state["stats"]["total_tasks_executed"] = state["stats"].get("total_tasks_executed", 0) + 1
    d1_client.save_state(state)

    print(f"Character interactions complete: {total_new_replies} replies generated")
    return total_new_replies > 0


# ==================== Main Tasks ====================

def do_publish_whisper(config, d1_client, text_provider, now_dt, dry_run=False,
                       image_provider=None, prompt_provider=None):
    """Execute the publish whisper task."""
    print("\n--- Publish Whisper ---")

    state = d1_client.get_state()
    state.setdefault("character_states", {})
    character_states = state["character_states"]
    last_run = state.get("last_run", {}).get("whispers_publish", "")
    random_offset = state.get("next_random_offset", {}).get("whispers_publish", 0)

    if not last_run:
        last_run = "2026-06-01T00:00:00+08:00"

    now_str = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    # Check trigger
    trigger_result = check_trigger("publish_whisper", last_run, now_str, random_offset)
    print(f"Trigger check: {trigger_result.get('trigger', False)} - {trigger_result.get('reason', '')}")

    if not trigger_result.get("trigger", False):
        print("Publish whisper: not triggered, skipping")
        return False

    # Get timeline
    timeline_text = get_timeline_text(15)
    if not timeline_text:
        print("Warning: failed to get timeline, using empty context")
        timeline_text = "(no recent whispers)"

    # Check holiday
    date_str = now_dt.strftime("%Y-%m-%d")
    day_info = check_holiday(date_str)
    print(f"Day info: {day_info.get('type')} {day_info.get('holiday_name', '')}")

    # Load characters.md and authors data
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}

    # Generate content: AI selects character + generates content in one call
    content_data = generate_whisper_content(
        text_provider, characters_md, timeline_text, day_info, now_dt, authors_data
    )

    if not content_data:
        # Fallback: use weighted random character selector + retry AI generation
        print("AI content generation failed, falling back to character_selector...")
        character_id = select_character(WHISPERS_DIR)
        character_name = get_author_nickname(character_id, authors_data)
        print(f"Fallback character: {character_name} ({character_id})")

        # Retry with a simpler prompt for the selected character
        content_data = _fallback_generate(text_provider, character_id, character_name,
                                          characters_md, timeline_text, day_info, now_dt)
        if not content_data:
            print("Fallback generation also failed, skipping")
            return False

    character_id = content_data["character"]
    character_name = get_author_nickname(character_id, authors_data)
    print(f"Selected character: {character_name} ({character_id})")
    print(f"Generated: {content_data['title']}")

    if dry_run:
        print(f"[DRY RUN] Would publish: {content_data['title']}")
        print(f"Content: {content_data['content']}")
        return False

    # Use model-provided slug, or fallback to hash-based slug
    slug = content_data.get("slug") or generate_slug(character_id, content_data["title"])
    month_str = now_dt.strftime("%Y-%m")
    month_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
    slug = check_slug(month_json_path, slug)
    print(f"Slug: {slug}")

    # Load month JSON
    if os.path.exists(month_json_path):
        month_data = load_json(month_json_path)
    else:
        month_data = {}

    # Add new whisper (store model-provided metadata for downstream use)
    model_mentioned = content_data.get("mentioned_characters", [])
    model_topics = content_data.get("topics", [])
    model_mood = content_data.get("mood")
    model_storyline_trigger = content_data.get("storyline_trigger")
    month_data[slug] = {
        "title": content_data["title"],
        "date": now_str,
        "author": character_id,
        "content": content_data["content"],
        "tags": [],
        "topics": model_topics,
        "mood": model_mood,
    }
    if model_storyline_trigger:
        month_data[slug]["storyline_trigger"] = model_storyline_trigger

    # ---- Image generation ----
    # Use model-provided needs_image + mentioned_characters, with rules 3/4 fallback
    image_generated = False
    if image_provider and prompt_provider and not dry_run:
        recent_with_images = load_recent_whispers_with_images(RECENT_K_FOR_IMAGE_STATS)
        should_img, reason, mentioned = should_have_image(
            content_data["content"], character_id, authors_data, recent_with_images,
            model_needs_image=content_data.get("needs_image", False),
            model_mentioned=model_mentioned
        )
        print(f"[image] Decision: {should_img} ({reason})")
        if should_img:
            date_str = now_dt.strftime("%Y-%m-%d")
            img_filename, img_path = generate_whisper_image(
                image_provider, prompt_provider,
                content_data["content"], character_id, mentioned,
                authors_data, now_dt, slug, date_str
            )
            if img_filename and img_path and os.path.exists(img_path):
                month_data[slug]["images"] = [f"/images/{img_filename}"]
                # Re-save whisper JSON with images field
                save_json(month_json_path, month_data)
                # Repack the month's tar so the image ships to the repo
                repack_month_images(month_str)
                image_generated = True
                print(f"[image] Attached image: /images/{img_filename}")
            else:
                print(f"[image] Generation failed, whisper will be text-only")
    elif not image_provider:
        print("[image] No image provider configured, skipping image generation")

    # Save
    save_json(month_json_path, month_data)
    _invalidate_whisper_cache()  # B1: invalidate cache after publishing
    print(f"Saved whisper to {month_json_path}")

    # Update character state (use model-provided mood/topics with keyword fallback)
    cs = character_states.setdefault(character_id, {})
    cs["mood"] = model_mood or _infer_mood(content_data["content"], character_id)
    cs["energy"] = max(20, cs.get("energy", 50) - 5)  # expends some energy
    cs["last_active"] = now_str
    topics = model_topics if model_topics else _extract_topics(content_data["content"])
    if topics:
        existing = cs.get("recent_topics", [])
        cs["recent_topics"] = (topics + existing)[:3]

    # Update state
    state["last_run"]["whispers_publish"] = now_str
    new_offset = random.randint(0, 90)
    state["next_random_offset"]["whispers_publish"] = new_offset
    state["stats"]["total_tasks_executed"] = state["stats"].get("total_tasks_executed", 0) + 1
    d1_client.save_state(state)

    print(f"Updated D1 state: last_run={now_str}, next_offset={new_offset}")
    return True


def _fallback_generate(text_provider, character_id, character_name,
                       characters_md, timeline_text, day_info, now_dt):
    """Fallback: generate content for a pre-selected character."""
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now_dt.weekday()]

    if day_info["type"] == "holiday":
        day_desc = f"法定节假日（{day_info.get('holiday_name', '节日')}）"
    elif day_info["type"] == "weekend":
        day_desc = "周末"
    else:
        day_desc = "工作日"

    hour = now_dt.hour
    if 7 <= hour < 9:
        period = "早上"
    elif 9 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 20:
        period = "傍晚"
    else:
        period = "晚上"

    system_prompt = """你是一个扮演角色的AI，在"豆包和朋友们的悄悄话"小站上发动态。

角色设定：
{characters_md}

要求：
1. 长度50-200字，短而精
2. 口语化、轻松、随意，像真人发朋友圈
3. 必须符合所扮演角色的性格和说话风格
4. 可以带1-2个emoji
5. 内容要符合当前场景（时间场景在用户消息中给出）
6. 不要和最近动态主题完全重复
7. 只输出JSON格式

输出格式（严格JSON）：
{"character": "角色ID", "title": "一句话标题", "content": "碎碎念正文"}""".replace("{characters_md}", characters_md)

    user_prompt = f"""你扮演的角色：{character_name}（角色ID: {character_id}）
当前时间：{now_str} {weekday_cn}，{day_desc}，{period}

最近的动态：
{timeline_text}

请以{character_name}的身份写一条新的碎碎念。只输出JSON。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.9, enable_thinking=True)
    except Exception as e:
        print(f"Fallback AI generation failed: {e}", file=sys.stderr)
        return None

    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        data = json.loads(response)
        return {
            "character": data.get("character", character_id).strip(),
            "title": data.get("title", "").strip(),
            "content": data.get("content", "").strip(),
            "slug": None,  # fallback will use generate_slug()
            "mood": None,
            "topics": [],
            "mentioned_characters": [],
            "needs_image": None,  # None = let rules 3/4 decide
            "storyline_trigger": None,
        }
    except json.JSONDecodeError:
        return None


def do_check_replies(config, d1_client, text_provider, now_dt, dry_run=False):
    """Check and reply to user comments from D1."""
    print("\n--- Check Replies ---")

    state = d1_client.get_state()
    last_run = state.get("last_run", {}).get("whispers_check_replies", "")

    if not last_run:
        last_run = "2026-06-01T00:00:00+08:00"

    now_str = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    random_offset = state.get("next_random_offset", {}).get("whispers_check_replies", 0)

    # Check trigger
    trigger_result = check_trigger("check_replies", last_run, now_str, random_offset)
    print(f"Trigger check: {trigger_result.get('trigger', False)} - {trigger_result.get('reason', '')}")

    if not trigger_result.get("trigger", False):
        print("Check replies: not triggered, skipping")
        return False, []

    # Get pending replies from D1 (is_doubao = 0)
    replies = d1_client.get_pending_replies()
    if not replies:
        print("No new replies to process")
        state["last_run"]["whispers_check_replies"] = now_str
        new_offset = random.randint(0, 5)
        state["next_random_offset"]["whispers_check_replies"] = new_offset
        d1_client.save_state(state)
        return False, []

    print(f"Found {len(replies)} pending replies to process")

    # Load characters.md and authors data
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}

    # Cache timeline text once (avoids subprocess per user reply)
    timeline_text = get_timeline_text(15)

    # Group replies by whisper_id
    replies_by_whisper = {}
    reply_ids_to_delete = []

    for reply in replies:
        whisper_id = reply.get("whisper_id", "")
        if whisper_id:
            if whisper_id not in replies_by_whisper:
                replies_by_whisper[whisper_id] = []
            replies_by_whisper[whisper_id].append(reply)
            if "id" in reply:
                reply_ids_to_delete.append(reply["id"])

    if dry_run:
        print(f"[DRY RUN] Would process {len(replies)} replies across {len(replies_by_whisper)} whispers")
        return False, []

    # Process each whisper's replies
    new_replies_added = 0

    # Load existing replies from repo files for context
    existing_replies_cache = {}
    character_states = state.get("character_states", {})

    for whisper_id, whisper_replies in replies_by_whisper.items():
        # Extract year-month from whisper_id (format: YYYY-MM-DD-slug)
        month_str = whisper_id[:7]  # YYYY-MM

        # Find the whisper content
        whisper_data = None
        whisper_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
        if os.path.exists(whisper_json_path):
            month_whispers = load_json(whisper_json_path)
            slug_part = whisper_id[11:]  # slug after the date-
            if slug_part in month_whispers:
                whisper_data = month_whispers[slug_part]

        if not whisper_data:
            print(f"Warning: whisper {whisper_id} not found, skipping replies")
            continue

        whisper_author_id = whisper_data.get("author", "")
        whisper_author_name = get_author_nickname(whisper_author_id, authors_data)
        whisper_content = whisper_data.get("content", "")

        # Load existing replies from repo for context (for friend selection)
        if month_str not in existing_replies_cache:
            reply_file_path = os.path.join(REPLIES_DIR, f"{month_str}.json")
            existing_replies_cache[month_str] = load_month_file(reply_file_path) if os.path.exists(reply_file_path) else {}
        existing_replies = existing_replies_cache[month_str].get(whisper_id, [])

        # Collect ALL new replies (user replies + character replies) for this
        # whisper, to be written to the repo json file in one batch at the end.
        # D1 is ONLY a transient cache for pending user replies; the canonical
        # store for both user and character replies is data/replies/*.json.
        # Character replies are NEVER written to D1.
        new_replies_for_whisper = []

        for user_reply in whisper_replies:
            user_content = user_reply.get("content", "")
            user_nickname = user_reply.get("nickname", "匿名")

            # Add the user reply itself to the repo json (author="" marks it
            # as a user reply, matching the existing data format).
            user_reply_ts = user_reply.get("timestamp", "")
            new_replies_for_whisper.append({
                "nickname": user_nickname,
                "content": user_content,
                "timestamp": user_reply_ts,
                "author": "",
                "reply_to": user_reply.get("reply_to", "") or "",
                "reply_to_floor": user_reply.get("reply_to_floor") or 0,
            })

            # Consolidated: one model call decides who replies + generates content
            smart_replies = generate_smart_reply(
                text_provider, whisper_content, whisper_author_id,
                whisper_author_name, user_content, user_nickname,
                characters_md, authors_data, character_states,
                existing_replies, timeline_text, now_dt, whisper_data
            )

            # Compute a safe base timestamp for character replies:
            # MUST be later than the user reply it responds to (否则会出现角色回复
            # 早于被回复的用户评论，破坏时序)，且不晚于当前真实时间。
            try:
                user_reply_dt = datetime.fromisoformat(user_reply_ts)
                if user_reply_dt.tzinfo is None:
                    user_reply_dt = user_reply_dt.replace(tzinfo=TZ_BEIJING)
            except Exception:
                user_reply_dt = now_dt

            # earliest = 用户回复后 1-2 分钟；同时允许最多比 now 早 1-3 分钟（自然抖动）；
            # 二者取较晚者，确保既晚于用户回复又不超 now。
            # 前端按分钟显示，所以抖动单位用分钟，避免多条回复显示同一时间。
            earliest = user_reply_dt + timedelta(minutes=random.randint(1, 2))
            base_dt = max(earliest, now_dt - timedelta(minutes=random.randint(0, 3)))
            base_dt = min(base_dt, now_dt)  # 不能晚于 now

            reply_dt = base_dt
            for char_id, char_name, role_type, ai_reply in smart_replies:
                if ai_reply:
                    reply_time = reply_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
                    # 下一条角色回复稍晚于此条（连续回复自然递增），但不超过 now
                    reply_dt = min(reply_dt + timedelta(minutes=random.randint(1, 3)),
                                   now_dt)

                    new_replies_for_whisper.append({
                        "nickname": char_name,
                        "content": ai_reply,
                        "timestamp": reply_time,
                        "author": char_id,
                        # Both author and friend are replying to the user's
                        # comment, so both must carry reply_to pointing at it.
                        # The old code cleared reply_to for role_type=="author",
                        # which dropped the @ on the author's own replies — a
                        # basic interaction rule violation. Deterministic rule,
                        # not AI-dependent.
                        "reply_to": user_nickname,
                        "reply_to_floor": user_reply.get("floor") or 0,
                    })
                    new_replies_added += 1
                    role_tag = f"[{role_type}]"
                    print(f"  {role_tag} {char_name} replied to {whisper_id}: {ai_reply[:50]}...")

        # Write all new replies (user + character) to repo json in one batch.
        # add_replies() merges with existing replies and recalculates floors
        # by timestamp, so ordering stays consistent.
        if new_replies_for_whisper:
            reply_file = os.path.join(REPLIES_DIR, f"{month_str}.json")
            add_replies(reply_file, whisper_id, new_replies_for_whisper)
            # Invalidate the cache so subsequent whispers in the same run see
            # the updated replies (avoids stale floor/timestamp context).
            if month_str in existing_replies_cache:
                existing_replies_cache[month_str] = load_month_file(reply_file)

    # NOTE: D1 reply deletion is deferred to main(), after git push succeeds.
    # If push fails, the local repo changes are lost (next checkout is fresh),
    # so keeping the D1 rows lets the next run re-process them — no data loss.

    # Update state
    state["last_run"]["whispers_check_replies"] = now_str
    new_offset = random.randint(0, 5)
    state["next_random_offset"]["whispers_check_replies"] = new_offset
    state["stats"]["total_tasks_executed"] = state["stats"].get("total_tasks_executed", 0) + 1
    d1_client.save_state(state)

    print(f"Reply processing complete: {new_replies_added} AI replies generated")
    return new_replies_added > 0, reply_ids_to_delete


# ==================== Git Operations ====================

def git_commit_and_push(changes_made, dry_run=False):
    """Commit and push changes. Returns True on success, False on failure.

    If the push is rejected (remote moved ahead — common when a concurrent
    run or manual push landed between checkout and push), pulls --rebase to
    replay our commit on top and retries. Gives up after a few attempts so
    we don't loop forever on a real conflict.
    """
    if not changes_made:
        print("\nNo changes to commit")
        return True

    if dry_run:
        print(f"\n[DRY RUN] Would commit and push changes")
        return True

    # Configure git
    run_script(["git", "config", "user.name", "Fox"])
    run_script(["git", "config", "user.email", "fox@example.com"])

    # Add and commit
    run_script(["git", "add", "-A"])
    run_script(["git", "commit", "-m", "feat: update whispers via automated runner"])

    # Push with rebase-retry. The remote often moves ahead between checkout
    # and push (concurrent runner, manual push). A plain push fails
    # non-fast-forward in that case; pull --rebase replays our commit on top
    # and we retry. Give up after 3 attempts on a real conflict.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        _, rc = run_script(["git", "push"])
        if rc == 0:
            print("Pushed changes to remote")
            return True
        if attempt >= max_attempts:
            print(f"Failed to push after {max_attempts} attempts, giving up", file=sys.stderr)
            return False
        print(f"Push rejected (attempt {attempt}/{max_attempts}), pulling --rebase...", file=sys.stderr)
        _, pull_rc = run_script(["git", "pull", "--rebase", "origin", "main"])
        if pull_rc != 0:
            print("git pull --rebase failed (likely a real conflict), giving up", file=sys.stderr)
            run_script(["git", "rebase", "--abort"])
            return False

    return False


# ==================== Main Entry ====================

def main():
    parser = argparse.ArgumentParser(description="Whisper runner")
    parser.add_argument("--dry-run", action="store_true", help="Dry run, no actual changes")
    parser.add_argument("--force-publish", action="store_true", help="Force publish regardless of trigger")
    args = parser.parse_args()

    now = now_beijing()
    print(f"=== Whisper Runner started at {now.strftime('%Y-%m-%d %H:%M:%S')} Beijing time ===")

    # Check active hours
    if not (ACTIVE_HOUR_START <= now.hour < ACTIVE_HOUR_END):
        print(f"Outside active hours ({ACTIVE_HOUR_START}:00-{ACTIVE_HOUR_END}:00), exiting")
        return 0

    # Load config
    config = load_json(CONFIG_PATH)
    print(f"Config loaded: {len(config.get('operations', {}))} operations")

    # Initialize clients
    try:
        d1_client = D1Client()
    except ValueError as e:
        print(f"Failed to initialize D1 client: {e}", file=sys.stderr)
        return 1

    ai_text_config = config.get("ai", {}).get("text", {})
    if not ai_text_config:
        print("No AI text provider configured", file=sys.stderr)
        return 1

    # Support both named-profile format and legacy flat format
    if "default" not in ai_text_config and "provider" in ai_text_config:
        ai_text_config = {"default": ai_text_config}

    # Create one text provider per profile
    text_providers = {}
    for name, prof_cfg in ai_text_config.items():
        try:
            text_providers[name] = create_text_provider(prof_cfg)
            print(f"AI text provider [{name}]: {prof_cfg.get('model', 'unknown')}")
        except Exception as e:
            print(f"Failed to init provider [{name}]: {e}", file=sys.stderr)

    if not text_providers:
        print("No text provider could be initialized", file=sys.stderr)
        return 1

    def get_provider(op_name):
        """Get the text provider for a given operation based on its text_profile."""
        op_cfg = config.get("operations", {}).get(op_name, {})
        profile = op_cfg.get("text_profile", "default")
        return text_providers.get(profile, text_providers.get("default"))

    # Initialize image provider (CF Workers AI flux-2-klein-4b) for whisper images.
    # Optional: if not configured or init fails, whispers will be text-only.
    image_provider = None
    ai_image_config = config.get("ai", {}).get("image", {})
    if ai_image_config:
        try:
            image_provider = create_image_provider(ai_image_config)
            print(f"AI image provider: {ai_image_config.get('model', 'unknown')}")
        except Exception as e:
            print(f"Failed to init image provider (whispers will be text-only): {e}", file=sys.stderr)

    # Pre-load and cache reference image PNG bytes so image generation doesn't
    # re-read files on every call.
    if image_provider:
        preload_references()

    # Prompt provider for image prompt building/rephrasing: use the "oc"
    # text profile if available, else fall back to default.
    prompt_provider = text_providers.get("oc") or text_providers.get("default")

    # Update heartbeat count
    state = d1_client.get_state()
    state.setdefault("character_states", {})
    state.setdefault("storylines", {"active": [], "completed": []})
    state["stats"]["total_heartbeats"] = state["stats"].get("total_heartbeats", 0) + 1
    # Evolve character states and storylines based on time
    _evolve_character_states(state["character_states"], now)
    _evolve_storylines(state["storylines"], now)
    d1_client.save_state(state)

    changes_made = False

    # Task 1: Publish whisper
    if args.force_publish:
        print("Force publish mode, bypassing trigger check")
        # Temporarily set last_run to far past to force trigger
        state["last_run"]["whispers_publish"] = "2026-06-01T00:00:00+08:00"
        d1_client.save_state(state)

    published = do_publish_whisper(
        config, d1_client, get_provider("publish_whisper"), now, args.dry_run,
        image_provider=image_provider, prompt_provider=prompt_provider
    )
    if published:
        changes_made = True

    # Task 2: Check replies
    replied, reply_ids_to_delete = do_check_replies(config, d1_client, get_provider("check_replies"), now, args.dry_run)
    if replied:
        changes_made = True

    # Task 3: Character interactions (generate character-to-character replies)
    interacted = do_character_interactions(config, d1_client, get_provider("character_interactions"), now, args.dry_run)
    if interacted:
        changes_made = True

    # Task 4: Apply pending image replacements from the diagnostic KV queue.
    # Runs every cron (no trigger check) — KV.list is cheap and replacements
    # are on-demand. Silently skips if KV env vars not configured.
    replaced, kv_keys_to_delete = apply_image_replacements(now, args.dry_run)
    if replaced:
        changes_made = True

    # Record token usage stats into D1 state (sum across all providers)
    combined_usage = {"prompt": 0, "completion": 0, "total": 0, "cache_hit": 0}
    for p in text_providers.values():
        if hasattr(p, "usage_total"):
            for k in combined_usage:
                combined_usage[k] += p.usage_total.get(k, 0)

    if combined_usage["total"] > 0:
        state = d1_client.get_state()
        merge_usage_into_state(state, combined_usage,
                               now.strftime("%Y-%m-%dT%H:%M:%S+08:00"))
        d1_client.save_state(state)
        print(f"Token usage this run: prompt={combined_usage['prompt']} "
              f"completion={combined_usage['completion']} "
              f"total={combined_usage['total']} "
              f"cache_hit={combined_usage['cache_hit']}")

    # Git commit and push. Destructive cleanups (D1 reply deletion, KV key
    # deletion) are deferred until AFTER a successful push — if the push
    # fails, the local file changes are lost (next checkout is fresh), so
    # keeping the D1 rows and KV keys lets the next run re-process them
    # instead of silently dropping user data.
    push_ok = git_commit_and_push(changes_made, args.dry_run)

    if push_ok and changes_made and not args.dry_run:
        if reply_ids_to_delete:
            print(f"Deleting {len(reply_ids_to_delete)} synced user replies from D1")
            d1_client.delete_replies(reply_ids_to_delete)
        if kv_keys_to_delete:
            kv = _get_diag_kv()
            if kv:
                print(f"Deleting {len(kv_keys_to_delete)} processed image-replacement KV keys")
                _delete_keys_silent(kv, kv_keys_to_delete)

    print(f"\n=== Whisper Runner finished ===")
    if not push_ok:
        print("WARNING: git push failed — D1 replies and KV keys were NOT deleted, "
              "next run will retry", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
