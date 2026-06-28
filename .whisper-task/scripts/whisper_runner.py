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
from datetime import datetime, timezone, timedelta

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ai_client import create_text_provider, create_image_provider, merge_usage_into_state
from d1_client import D1Client
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
    "Reference image(s) show the character(s).\nFor single reference image: apply all fixed attributes to the sole character.\nFor multiple reference images (input_image_0, input_image_1, ...): each image represents a distinct character. All characters must appear together in the rendered scene. Apply fixed attributes to their respective referenced character.\nKeep the reference image's clothing style. No exposed skin.\nStyle anchor: kawaii chibi anime illustration, thick clean black outlines, soft cel shading, pastel warm colors, rounded cute proportions, big glossy eyes, gentle warm ambient lighting, cozy daily scene, soft blush, clean flat coloring.\nFixed attributes: face shape, hairstyle, skin tone, eye color, body build, clothing style, hand details."
)

# Beijing timezone
TZ_BEIJING = timezone(timedelta(hours=8))

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

# Avatar filename mapping. avatar.webp is doubao (no suffix).
AVATAR_FILES = {
    "doubao": "avatar.webp",
    "guga": "avatar-guga.webp",
    "doro": "avatar-doro.webp",
    "feibi": "avatar-feibi.webp",
    "baizi": "avatar-baizi.webp",
    "nuonuo": "avatar-nuonuo.webp",
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

def get_avatar_path(author_id):
    """Return absolute path to an author's avatar file, or None if missing."""
    fname = AVATAR_FILES.get(author_id)
    if not fname:
        return None
    path = os.path.join(IMAGES_DIR, fname)
    return path if os.path.exists(path) else None


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


def should_have_image(content, character_id, authors_data, recent_with_images):
    """Decide whether a new whisper should have an image.

    Rules (per instructions.md §2.3.0):
    1. Mentions other characters -> mandatory group photo
    2. Has concrete scene/object/photo-related words -> mandatory
    3. Otherwise: if recent pure-text ratio >= 30% -> mandatory
    4. Otherwise: 70% probability

    Returns (should_have: bool, reason: str, mentioned_chars: list).
    """
    # Rule 1: mentions other characters -> mandatory group photo
    mentioned = [m for m in extract_mentioned_characters(content, authors_data)
                 if m != character_id]
    if mentioned:
        return True, f"mentions other characters: {mentioned}", mentioned

    # Rule 2: concrete scene / object / photo words
    photo_words = ["拍", "给你看", "偷拍", "晒"]
    scene_patterns = ["吃了", "去了", "在", "买了", "做了", "收到", "发现", "刚到", "新买的"]
    if any(w in content for w in photo_words) or any(p in content for p in scene_patterns):
        return True, "concrete scene/object/photo-related", mentioned

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
    # Build character NAME list only (no appearance description).
    # Use romanized names so the whole prompt stays English (flux is
    # English-trained; Chinese in the prompt degrades image quality).
    def _roman(aid):
        return NAME_ROMANIZATION.get(aid, get_author_nickname(aid, authors_data))
    char_names = [_roman(character_id)] + [_roman(a) for a in mentioned_chars]
    char_names_text = ", ".join(char_names)

    #system_prompt = f"""You write image generation prompts for the flux-2-klein-4b model. These images illustrate short social-media posts ("whispers") from a group of friends.
#
#CRITICAL RULES (the prompt is the single most important factor for output quality):
#1. Output ONLY English. No Chinese characters anywhere - translate the entire scene (objects, places, actions, mood) to English. Use the romanized character names provided below. No explanation, no quotes, no markdown.
#2. Be CONCRETE and VISUAL: use specific nouns (objects, places, body language, facial expressions). Avoid abstract adjectives.
#3. Describe the SCENE and ACTION matching the post content. Never describe character appearance (hair color, eye color, clothing, body type, etc.) - character appearance is locked to reference avatar images passed separately, the prompt only describes what characters are DOING and the scene around them.
#4. Refer to characters only by their romanized names (e.g. "Doro and Guga eating together"), never by appearance traits.
#5. If multiple characters are mentioned, describe them interacting naturally in the scene.
#
#OUTPUT FORMAT - use this labeled multi-line structure with blank-line grouping (do NOT describe character appearance in any line, only actions/scene/environment):
#Action: <what the character(s) are doing, specific verbs and body language>
#Object: <key objects/props in the scene, specific nouns>
#Expression: <facial expression only, no appearance traits>
#
#Setting: <location/scene description>
#Time: <time of day>
#Weather: <weather if relevant>
#Environment details: <concrete background elements: furniture, plants, objects, textures>
#
#Mood: <emotional atmosphere>
#Style: digital illustration, soft painterly textures
#Palette: <2-4 dominant colors matching the scene mood>
#Lighting: <specific light source and how it falls on the scene>
#Quality: detailed rendering, shallow depth of field, 8k
#
#Do NOT add any other lines. Do NOT describe hair, face shape, clothing, or body type - those come from reference images.
#
#Characters appearing in this image (refer to them by name only): {char_names_text}
#
#The model output is 1024x768 (landscape). Compose accordingly."""

    system_prompt = f"""You are a prompt generator. Your task is to generate a structured character scene description based on user prompt.

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
    3. Output format must be:

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

    Action: they are cooking, stirring a pot, holding a wooden spoon
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
        prompt = text_provider.generate(messages, max_tokens=10240, temperature=0.3)
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

    # Collect reference images: author avatar + mentioned chars' avatars (max 4)
    ref_chars = [character_id] + mentioned_chars
    ref_paths = []
    for aid in ref_chars:
        p = get_avatar_path(aid)
        if p:
            ref_paths.append(p)
        else:
            print(f"[image] No avatar found for {aid}", file=sys.stderr)
    ref_paths = ref_paths[:MAX_REFERENCE_IMAGES]
    if not ref_paths:
        print("[image] No reference avatars available, skipping image", file=sys.stderr)
        return None, None
    print(f"[image] Reference avatars: {[os.path.basename(p) for p in ref_paths]}")

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
                                                   reference_images=ref_paths)
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
                print(f"[image] Flagged by safety filter (attempt {attempt+1}), rephrasing...", file=sys.stderr)
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
        new_prompt = rephrase_provider.generate(messages, max_tokens=10240, temperature=0.3)
        new_prompt = new_prompt.strip().strip('"').strip("'").strip("`").strip()
        if new_prompt.startswith("```"):
            lines = new_prompt.split("\n")
            new_prompt = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return new_prompt if new_prompt else None
    except Exception as e:
        print(f"[image] Rephrase failed: {e}", file=sys.stderr)
        return None


def repack_month_images(month_str):
    """Re-run pack_images.py for the given month after adding new images."""
    stdout, rc = run_script([sys.executable, PACK_IMAGES_SCRIPT, month_str])
    if rc == 0:
        print(f"[image] Repacked {month_str}.tar")
    else:
        print(f"[image] WARNING: pack_images.py failed for {month_str}", file=sys.stderr)
    return rc == 0


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


def build_publish_prompt(characters_md, timeline_text, day_info, now_dt,
                         authors_data, recent_topics_summary):
    """
    Build system and user prompts for whisper generation.
    AI selects the character AND generates content in one call.
    """
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now_dt.weekday()]

    # Day type
    if day_info["type"] == "holiday":
        day_desc = f"法定节假日（{day_info.get('holiday_name', '节日')}）"
    elif day_info["type"] == "weekend":
        day_desc = "周末"
    else:
        day_desc = "工作日"

    # Time period
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

输出格式（严格JSON，不要输出任何其他内容、不要markdown代码块）：
{{
  "character": "角色ID（如 doro / feibi / guga / baizi / nuonuo / doubao）",
  "title": "一句话标题，不超过15字",
  "content": "碎碎念正文，50-200字，必须包含换行分段（\\n\\n）"
}}"""

    user_prompt = f"""当前时间：{now_str} {weekday_cn}，{day_desc}，{period}

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
            response = text_provider.generate(messages, max_tokens=10240, temperature=0.7)
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
            return {"character": character, "title": title, "content": content}
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
    """Generate a slug from character ID and title."""
    # Simple: use character_id + a short hash of title
    import hashlib
    hash_str = hashlib.md5(title.encode("utf-8")).hexdigest()[:6]
    return f"{character_id}-{hash_str}"


# ==================== Reply Generation ====================

def build_reply_prompt(whisper_content, whisper_author_name, user_reply_content,
                       characters_md, character_id, character_name, timeline_text):
    """Build prompt for generating a reply to a user comment."""
    system_prompt = f"""你是一个扮演角色的AI，在"豆包和朋友们的悄悄话"小站上回复用户的评论。

角色设定：
{characters_md}

要求：
1. 回复要符合所扮演角色的性格和说话风格，合理使用emoji和标点符号（非常重要，直接影响回复的感觉和语气等真实感）
2. 口语化、自然，像真实朋友聊天
3. 回复长度10-80字
4. 不要涉及用户隐私
5. 可以追问、调侃、分享看法等等，这只是举例，不是局限于这些
6. 只输出回复内容，不要输出其他内容"""

    user_prompt = f"""你扮演的角色：{character_name}（角色ID: {character_id}）
动态作者：{whisper_author_name}
动态内容：{whisper_content}
动态时间线：{timeline_text}

用户评论：{user_reply_content}

请以{character_name}的身份回复这条评论。只输出回复内容。"""

    return system_prompt, user_prompt


def generate_reply(text_provider, whisper_content, whisper_author_name,
                   user_reply_content, characters_md, character_id, character_name):
    """Generate a reply to a user comment."""
    system_prompt, user_prompt = build_reply_prompt(
        whisper_content, whisper_author_name, user_reply_content,
        characters_md, character_id, character_name, get_timeline_text(15)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.9)
        return response.strip() if response else None
    except Exception as e:
        print(f"Reply generation failed: {e}", file=sys.stderr)
        return None


# ==================== Character Interactions ====================

def build_interaction_prompt(whisper_data, whisper_author_name, existing_replies,
                             characters_md, authors_data, now_dt, candidate_chars):
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

    system_prompt = f"""你是"豆包和朋友们的悄悄话"小站的角色互动生成器。朋友们会看彼此的动态，自然地评论互动。

角色设定：
{characters_md}

{voice_rules}

互动原则（核心——决定真实感）：
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

    user_prompt = f"""动态作者：{whisper_author_name}
动态内容：{whisper_data.get('content', '')}

已有回复：
{replies_text}

本次可选角色（只从中挑1-3个，不要全用；作者可下场回复评论者）：
{char_list_text}

当前时间：{now_dt.strftime('%Y-%m-%d %H:%M')}

请生成1-3条角色互动回复。记住：至少1条接话（带reply_to），作者可参与，口吻差异要明显。只输出JSON数组。"""

    return system_prompt, user_prompt


def generate_character_interactions(text_provider, whisper_data, whisper_author_name,
                                    existing_replies, characters_md, authors_data, now_dt,
                                    candidate_chars):
    """Generate character-to-character replies via AI.

    candidate_chars: list of (char_id, nickname) tuples allowed this round
    (pre-filtered for variety/cooldown; includes whisper author if they
    should reply to commenters this round).

    Returns list of reply dicts or None.
    """
    system_prompt, user_prompt = build_interaction_prompt(
        whisper_data, whisper_author_name, existing_replies,
        characters_md, authors_data, now_dt, candidate_chars
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.7)
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
    for r in existing_replies or []:
        f = r.get("floor")
        if f is not None:
            floor_author_map[f] = r.get("author", "") or r.get("nickname", "")
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
        valid_replies.append({
            "nickname": nickname,
            "content": content,
            "author": author_id,
            "reply_to": reply_to if reply_to else "",
            "reply_to_floor": reply_to_floor if reply_to_floor else 0,
        })

    # P1: timestamps use ACTUAL run time (now_dt), with small backdated jitter
    # so replies don't all share the exact same minute. This produces real
    # cross-hour/cross-day distribution across multiple cron runs, instead of
    # the old "all挤在3-12分钟窗口" fake-spread.
    if valid_replies:
        # Assign each reply a timestamp = now - (1..10 min per reply, staggered)
        # so the most recent reply is ~1-3 min ago, earlier ones further back
        # but all within the last ~30 min (this round's natural window).
        current_dt = now_dt - timedelta(minutes=random.randint(1, 3))
        for reply in reversed(valid_replies):
            reply["timestamp"] = current_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
            current_dt = current_dt - timedelta(minutes=random.randint(2, 8))

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
            characters_md, authors_data, now_dt, candidate_chars
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

    # Generate slug and check uniqueness
    slug = generate_slug(character_id, content_data["title"])
    month_str = now_dt.strftime("%Y-%m")
    month_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
    slug = check_slug(month_json_path, slug)
    print(f"Slug: {slug}")

    # Load month JSON
    if os.path.exists(month_json_path):
        month_data = load_json(month_json_path)
    else:
        month_data = {}

    # Add new whisper
    month_data[slug] = {
        "title": content_data["title"],
        "date": now_str,
        "author": character_id,
        "content": content_data["content"],
        "tags": [],
    }

    # ---- Image generation ----
    # Decide whether this whisper should have an image, then generate one
    # using CF Workers AI (flux-2-klein-4b) with character avatars as
    # reference images. See instructions.md §2.3 for rules.
    image_generated = False
    if image_provider and prompt_provider and not dry_run:
        recent_with_images = load_recent_whispers_with_images(RECENT_K_FOR_IMAGE_STATS)
        should_img, reason, mentioned = should_have_image(
            content_data["content"], character_id, authors_data, recent_with_images
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
    print(f"Saved whisper to {month_json_path}")

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
        response = text_provider.generate(messages, max_tokens=10240, temperature=0.9)
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
        return False

    # Get pending replies from D1 (is_doubao = 0)
    replies = d1_client.get_pending_replies()
    if not replies:
        print("No new replies to process")
        state["last_run"]["whispers_check_replies"] = now_str
        new_offset = random.randint(0, 5)
        state["next_random_offset"]["whispers_check_replies"] = new_offset
        d1_client.save_state(state)
        return False

    print(f"Found {len(replies)} pending replies to process")

    # Load characters.md and authors data
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}

    # Group replies by whisper_id
    replies_by_whisper = {}
    reply_ids_to_mark = []

    for reply in replies:
        whisper_id = reply.get("whisper_id", "")
        if whisper_id:
            if whisper_id not in replies_by_whisper:
                replies_by_whisper[whisper_id] = []
            replies_by_whisper[whisper_id].append(reply)
            if "id" in reply:
                reply_ids_to_mark.append(reply["id"])

    if dry_run:
        print(f"[DRY RUN] Would process {len(replies)} replies across {len(replies_by_whisper)} whispers")
        return False

    # Process each whisper's replies
    new_replies_added = 0

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

        for user_reply in whisper_replies:
            user_content = user_reply.get("content", "")
            user_nickname = user_reply.get("nickname", "匿名")

            # Generate a character reply (whisper author replies)
            ai_reply = generate_reply(
                text_provider, whisper_content, whisper_author_name,
                user_content, characters_md, whisper_author_id, whisper_author_name
            )

            if ai_reply:
                reply_time = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
                # Get next floor number
                max_floor = d1_client.get_max_floor(whisper_id)
                new_floor = max_floor + 1

                # Insert character reply into D1
                d1_client.add_character_reply(
                    whisper_id, whisper_author_name, ai_reply,
                    reply_time, new_floor
                )
                new_replies_added += 1
                print(f"  Generated reply for {whisper_id}: {ai_reply[:50]}...")

    # Mark user replies as processed (is_doubao = 2)
    if reply_ids_to_mark:
        print(f"Marking {len(reply_ids_to_mark)} user replies as processed")
        d1_client.mark_replies_processed(reply_ids_to_mark)

    # Update state
    state["last_run"]["whispers_check_replies"] = now_str
    new_offset = random.randint(0, 5)
    state["next_random_offset"]["whispers_check_replies"] = new_offset
    state["stats"]["total_tasks_executed"] = state["stats"].get("total_tasks_executed", 0) + 1
    d1_client.save_state(state)

    print(f"Reply processing complete: {new_replies_added} AI replies generated")
    return new_replies_added > 0


# ==================== Git Operations ====================

def git_commit_and_push(changes_made, dry_run=False):
    """Commit and push changes if any."""
    if not changes_made:
        print("\nNo changes to commit")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would commit and push changes")
        return

    # Configure git
    run_script(["git", "config", "user.name", "Fox"])
    run_script(["git", "config", "user.email", "fox@example.com"])

    # Add changes
    run_script(["git", "add", "-A"])

    # Commit
    commit_msg = "feat: update whispers via automated runner"
    run_script(["git", "commit", "-m", commit_msg])

    # Push
    stdout, rc = run_script(["git", "push"])
    if rc == 0:
        print("Pushed changes to remote")
    else:
        print("Failed to push changes", file=sys.stderr)


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

    # Prompt provider for image prompt building/rephrasing: use the "free"
    # text profile if available, else fall back to default.
    prompt_provider = text_providers.get("free") or text_providers.get("default")

    # Update heartbeat count
    state = d1_client.get_state()
    state["stats"]["total_heartbeats"] = state["stats"].get("total_heartbeats", 0) + 1
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
    replied = do_check_replies(config, d1_client, get_provider("check_replies"), now, args.dry_run)
    if replied:
        changes_made = True

    # Task 3: Character interactions (generate character-to-character replies)
    interacted = do_character_interactions(config, d1_client, get_provider("character_interactions"), now, args.dry_run)
    if interacted:
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

    # Git commit and push
    git_commit_and_push(changes_made, args.dry_run)

    print(f"\n=== Whisper Runner finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
