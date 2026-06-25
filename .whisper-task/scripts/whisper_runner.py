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
from datetime import datetime, timezone, timedelta

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ai_client import create_text_provider, merge_usage_into_state
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


# ==================== Content Generation ====================

def build_publish_prompt(characters_md, timeline_text, day_info, now_dt,
                         authors_data):
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
- 如果最近有人在聊某个话题，可以延续或回应

写动态的要求：
1. 长度50-200字，短而精
2. 口语化、轻松、随意，像真人发朋友圈
3. 必须符合所选角色的性格和说话风格
4. 可以带1-2个emoji，不要太多
5. 内容要符合当前场景（时间场景在用户消息中给出）
6. 不要和最近动态主题完全重复
7. 不要涉及任何真实个人隐私

输出格式（严格JSON，不要输出其他内容）：
{{"character": "角色ID", "title": "一句话标题", "content": "碎碎念正文"}}"""

    user_prompt = f"""当前时间：{now_str} {weekday_cn}，{day_desc}，{period}

最近的动态（参考上下文，不要矛盾，不要完全重复主题）：
{timeline_text}

请选择一个角色并写一条新的碎碎念。只输出JSON。"""

    return system_prompt, user_prompt


def generate_whisper_content(text_provider, characters_md, timeline_text,
                             day_info, now_dt, authors_data):
    """
    Generate whisper content via AI.
    AI selects character and generates content in one call.
    Returns {"character": "char_id", "title": "...", "content": "..."} or None.
    """
    system_prompt, user_prompt = build_publish_prompt(
        characters_md, timeline_text, day_info, now_dt, authors_data
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=512, temperature=0.9)
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

        # Validate character ID
        if character not in authors_data:
            print(f"Warning: AI returned unknown character '{character}'", file=sys.stderr)
            return None

        return {"character": character, "title": title, "content": content}
    except json.JSONDecodeError as e:
        print(f"Failed to parse AI response as JSON: {e}", file=sys.stderr)
        print(f"Response: {response[:200]}", file=sys.stderr)
        return None


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
1. 回复要符合所扮演角色的性格和说话风格
2. 口语化、自然，像真实朋友聊天
3. 回复长度10-80字
4. 不要涉及用户隐私
5. 可以追问、调侃、分享看法
6. 只输出回复内容，不要输出其他内容"""

    user_prompt = f"""你扮演的角色：{character_name}（角色ID: {character_id}）
动态作者：{whisper_author_name}
动态内容：{whisper_content}

用户评论：{user_reply_content}

请以{character_name}的身份回复这条评论。只输出回复内容。"""

    return system_prompt, user_prompt


def generate_reply(text_provider, whisper_content, whisper_author_name,
                   user_reply_content, characters_md, character_id, character_name):
    """Generate a reply to a user comment."""
    system_prompt, user_prompt = build_reply_prompt(
        whisper_content, whisper_author_name, user_reply_content,
        characters_md, character_id, character_name
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=256, temperature=0.85)
        return response.strip() if response else None
    except Exception as e:
        print(f"Reply generation failed: {e}", file=sys.stderr)
        return None


# ==================== Character Interactions ====================

def build_interaction_prompt(whisper_data, whisper_author_name, existing_replies,
                             characters_md, authors_data, now_dt):
    """Build prompt for generating character-to-character interactions."""
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

    # Build character list (excluding whisper author)
    char_list = []
    for char_id, info in authors_data.items():
        if char_id != whisper_author_id:
            char_list.append(f"- {info.get('name', char_id)}（ID: {char_id}）")
    char_list_text = "\n".join(char_list)

    system_prompt = f"""你是"豆包和朋友们的悄悄话"小站的角色互动生成器。朋友们会看彼此的动态，自然地评论互动。

角色设定：
{characters_md}

互动原则：
1. 为动态生成2-5条角色间的互动回复，像朋友刷朋友圈看到动态后自然评论
2. 回复数量看内容：平淡动态可能只有2个朋友评论，有话题性的动态可能有4-5个朋友参与；不要每次都固定数量
3. 可以直接回复动态，也可以回复已有评论（形成对话链）
4. 互动要自然，符合角色性格和说话风格
5. 不要每个人都只回复动态，看到有意思的评论可以接话
6. 回复长度10-80字，口语化、轻松
7. 不要涉及隐私
8. 动态作者不参与回复（是别人来评论TA的动态）

输出格式（严格JSON数组，不要输出其他内容）：
[{{"author": "角色ID", "nickname": "角色名", "content": "回复内容", "reply_to": "回复对象昵称或空字符串", "reply_to_floor": 楼层号或0}}]

字段说明：
- reply_to: 回复某条已有评论时填该评论者的昵称；直接回复动态填空字符串""
- reply_to_floor: 回复某条已有评论时填该评论的楼层号；直接回复动态填0
- 只能回复已有评论（不能回复其他新生成的回复）"""

    user_prompt = f"""动态作者：{whisper_author_name}
动态内容：{whisper_data.get('content', '')}

已有回复：
{replies_text}

可选角色（不要用动态作者"{whisper_author_name}"）：
{char_list_text}

当前时间：{now_dt.strftime('%Y-%m-%d %H:%M')}

请生成2-5条角色互动回复，数量自然即可。只输出JSON数组。"""

    return system_prompt, user_prompt


def generate_character_interactions(text_provider, whisper_data, whisper_author_name,
                                    existing_replies, characters_md, authors_data, now_dt):
    """Generate character-to-character replies via AI. Returns list of reply dicts or None."""
    system_prompt, user_prompt = build_interaction_prompt(
        whisper_data, whisper_author_name, existing_replies,
        characters_md, authors_data, now_dt
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = text_provider.generate(messages, max_tokens=512, temperature=0.9)
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
    valid_replies = []
    for idx, r in enumerate(replies):
        if not isinstance(r, dict):
            continue
        author_id = r.get("author", "")
        nickname = r.get("nickname", "")
        content = r.get("content", "").strip()
        if not author_id or not content:
            continue
        # Don't allow whisper author to reply to their own whisper
        if author_id == whisper_author_id:
            continue
        # Ensure nickname matches author_id
        if author_id in authors_data:
            nickname = authors_data[author_id].get("name", nickname)
        reply_to = r.get("reply_to", "")
        reply_to_floor = r.get("reply_to_floor", 0)
        valid_replies.append({
            "nickname": nickname,
            "content": content,
            "author": author_id,
            "reply_to": reply_to if reply_to else "",
            "reply_to_floor": reply_to_floor if reply_to_floor else 0,
        })

    # Stagger timestamps: replies should be spread out over time, like real
    # friends commenting at different moments. Each reply a few minutes apart,
    # all in the past relative to now.
    if valid_replies:
        # Calculate the earliest timestamp: work backwards from now
        # Last reply is 2-6 min ago, each earlier reply 3-12 min before that
        current_dt = now_dt - timedelta(minutes=random.randint(2, 6))
        for _ in range(len(valid_replies) - 1):
            current_dt = current_dt - timedelta(minutes=random.randint(3, 12))
        # Now assign timestamps going forward, each a few minutes apart
        for reply in valid_replies:
            reply["timestamp"] = current_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
            current_dt = current_dt + timedelta(minutes=random.randint(3, 12))
            # Ensure not in the future
            if current_dt > now_dt:
                current_dt = now_dt - timedelta(minutes=1)

    return valid_replies if valid_replies else None


def do_character_interactions(config, d1_client, text_provider, now_dt, dry_run=False):
    """Generate character-to-character interactions for recent whispers lacking replies."""
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

    # Find recent whispers (last 48h) that need interactions
    cutoff = now_dt - timedelta(hours=48)
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

            # Must be within 48h, at least 1h old
            if w_dt < cutoff or w_dt > min_age:
                continue

            whisper_id = f"{w_dt.strftime('%Y-%m-%d')}-{slug}"

            # Check existing character replies
            reply_file = os.path.join(REPLIES_DIR, f"{month_str}.json")
            existing = load_month_file(reply_file)
            existing_replies = existing.get(whisper_id, [])
            char_reply_count = sum(1 for r in existing_replies if r.get("author", ""))

            if char_reply_count < 5:
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

    # Sort by date descending (newest first), take up to 3
    candidates.sort(key=lambda x: x["date"], reverse=True)
    candidates = candidates[:3]

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

        new_replies = generate_character_interactions(
            text_provider, w_data, author_name, existing_replies,
            characters_md, authors_data, now_dt
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

def do_publish_whisper(config, d1_client, text_provider, now_dt, dry_run=False):
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
        response = text_provider.generate(messages, max_tokens=512, temperature=0.9)
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

    ai_config = config.get("ai", {}).get("text", {})
    if not ai_config:
        print("No AI text provider configured", file=sys.stderr)
        return 1

    try:
        text_provider = create_text_provider(ai_config)
        print(f"AI text provider: {ai_config.get('provider', 'unknown')}")
    except Exception as e:
        print(f"Failed to initialize AI provider: {e}", file=sys.stderr)
        return 1

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

    published = do_publish_whisper(config, d1_client, text_provider, now, args.dry_run)
    if published:
        changes_made = True

    # Task 2: Check replies
    replied = do_check_replies(config, d1_client, text_provider, now, args.dry_run)
    if replied:
        changes_made = True

    # Task 3: Character interactions (generate character-to-character replies)
    interacted = do_character_interactions(config, d1_client, text_provider, now, args.dry_run)
    if interacted:
        changes_made = True

    # Record token usage stats into D1 state
    if text_provider.usage_total["total"] > 0:
        state = d1_client.get_state()
        merge_usage_into_state(state, text_provider.usage_total,
                               now.strftime("%Y-%m-%dT%H:%M:%S+08:00"))
        d1_client.save_state(state)
        print(f"Token usage this run: prompt={text_provider.usage_total['prompt']} "
              f"completion={text_provider.usage_total['completion']} "
              f"total={text_provider.usage_total['total']} "
              f"cache_hit={text_provider.usage_total['cache_hit']}")

    # Git commit and push
    git_commit_and_push(changes_made, args.dry_run)

    print(f"\n=== Whisper Runner finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
