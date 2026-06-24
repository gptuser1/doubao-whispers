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

from ai_client import create_text_provider
from d1_client import D1Client
from character_selector import select_character, CHARACTER_WEIGHTS

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

def build_publish_prompt(character_id, character_name, characters_md, timeline_text,
                         day_info, now_dt, recent_authors):
    """Build the system and user prompts for whisper generation."""
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now_dt.weekday()]

    # Determine day type description
    if day_info["type"] == "holiday":
        day_desc = f"法定节假日（{day_info.get('holiday_name', '节日')}）"
    elif day_info["type"] == "weekend":
        day_desc = "周末"
    else:
        day_desc = "工作日"

    # Time period description
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

    system_prompt = f"""你是一个扮演"{character_name}"角色的AI，在一个叫"豆包和朋友们的悄悄话"的小站上发动态。

你的任务：以{character_name}的身份写一条碎碎念动态，就像发QQ空间说说一样。

角色设定：
{characters_md}

你是：{character_name}（角色ID: {character_id}）

要求：
1. 长度50-200字，短而精
2. 口语化、轻松、随意，像真人发朋友圈
3. 必须符合{character_name}的性格和说话风格
4. 可以带1-2个emoji，不要太多
5. 内容要符合当前场景：{day_desc}的{period}
6. 不要和最近动态主题重复
7. 不要涉及任何真实个人隐私
8. 只输出JSON格式，不要输出其他内容

输出格式（严格JSON）：
{{"title": "一句话标题", "content": "碎碎念正文"}}"""

    user_prompt = f"""当前时间：{now_str} {weekday_cn}，{day_desc}，{period}

最近的动态（不要和这些主题重复）：
{timeline_text}

请以{character_name}的身份写一条新的碎碎念。只输出JSON。"""

    return system_prompt, user_prompt


def generate_whisper_content(text_provider, character_id, character_name,
                             characters_md, timeline_text, day_info, now_dt,
                             recent_authors):
    """Generate whisper content via AI."""
    system_prompt, user_prompt = build_publish_prompt(
        character_id, character_name, characters_md, timeline_text,
        day_info, now_dt, recent_authors
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
    # Try to extract JSON from possible markdown code blocks
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
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()

        if not title or not content:
            print("AI response missing title or content", file=sys.stderr)
            return None

        return {"title": title, "content": content}
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
    system_prompt = f"""你是一个扮演"{character_name}"角色的AI，在"豆包和朋友们的悄悄话"小站上回复用户的评论。

角色设定：
{characters_md}

你是：{character_name}（角色ID: {character_id}）

要求：
1. 回复要符合{character_name}的性格和说话风格
2. 口语化、自然，像真实朋友聊天
3. 回复长度10-80字
4. 不要涉及用户隐私
5. 可以追问、调侃、分享看法
6. 只输出回复内容，不要输出其他内容"""

    user_prompt = f"""动态作者：{whisper_author_name}
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

    # Select character
    character_id = select_character(WHISPERS_DIR)
    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}
    character_name = get_author_nickname(character_id, authors_data)
    print(f"Selected character: {character_name} ({character_id})")

    # Load characters.md
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    # Get recent authors for context
    from character_selector import get_recent_authors
    recent_authors = get_recent_authors(WHISPERS_DIR)

    # Generate content
    content_data = generate_whisper_content(
        text_provider, character_id, character_name,
        characters_md, timeline_text, day_info, now_dt, recent_authors
    )

    if not content_data:
        print("Failed to generate whisper content, skipping")
        return False

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
    whisper_id = f"{now_dt.strftime('%Y-%m-%d')}-{slug}"
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
    # Generate new random offset for next time
    new_offset = random.randint(0, 90)
    state["next_random_offset"]["whispers_publish"] = new_offset
    state["stats"]["total_tasks_executed"] = state["stats"].get("total_tasks_executed", 0) + 1
    d1_client.save_state(state)

    print(f"Updated D1 state: last_run={now_str}, next_offset={new_offset}")
    return True


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

    # Get replies from D1
    replies = d1_client.get_replies()
    if not replies:
        print("No new replies to process")
        # Update state even if no replies
        state["last_run"]["whispers_check_replies"] = now_str
        new_offset = random.randint(0, 5)
        state["next_random_offset"]["whispers_check_replies"] = new_offset
        d1_client.save_state(state)
        return False

    print(f"Found {len(replies)} replies to process")

    # Load characters.md
    characters_md = ""
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            characters_md = f.read()

    # Load authors data
    authors_data = load_json(AUTHORS_PATH) if os.path.exists(AUTHORS_PATH) else {}

    # Group replies by whisper_id
    replies_by_whisper = {}
    reply_ids_to_delete = []

    for reply in replies:
        whisper_id = reply.get("whisper_id", "")
        if whisper_id:
            if whisper_id not in replies_by_whisper:
                replies_by_whisper[whisper_id] = []
            replies_by_whisper[whisper_id].append(reply)
            # Collect reply ID for deletion
            if "id" in reply:
                reply_ids_to_delete.append(reply["id"])

    if dry_run:
        print(f"[DRY RUN] Would process {len(replies)} replies across {len(replies_by_whisper)} whispers")
        return False

    # Process each whisper's replies
    new_replies_added = 0

    for whisper_id, whisper_replies in replies_by_whisper.items():
        # Extract year-month from whisper_id (format: YYYY-MM-DD-slug)
        month_str = whisper_id[:7]  # YYYY-MM
        reply_file_path = os.path.join(REPLIES_DIR, f"{month_str}.json")

        # Find the whisper content
        whisper_data = None
        whisper_json_path = os.path.join(WHISPERS_DIR, f"{month_str}.json")
        if os.path.exists(whisper_json_path):
            month_whispers = load_json(whisper_json_path)
            # whisper_id = YYYY-MM-DD-slug, slug is everything after date
            date_part = whisper_id[:10]  # YYYY-MM-DD
            slug_part = whisper_id[11:]  # slug after the date-
            if slug_part in month_whispers:
                whisper_data = month_whispers[slug_part]

        if not whisper_data:
            print(f"Warning: whisper {whisper_id} not found, skipping replies")
            continue

        whisper_author_id = whisper_data.get("author", "")
        whisper_author_name = get_author_nickname(whisper_author_id, authors_data)
        whisper_content = whisper_data.get("content", "")

        # Build replies to add
        replies_to_add = []

        for user_reply in whisper_replies:
            user_content = user_reply.get("content", "")
            user_nickname = user_reply.get("nickname", "匿名")
            user_timestamp = user_reply.get("timestamp", now_str)

            # Add user reply to the list
            replies_to_add.append({
                "nickname": user_nickname,
                "content": user_content,
                "timestamp": user_timestamp,
                "author": "",
            })

            # Generate a character reply
            # Choose which character replies: prefer the whisper author
            reply_char_id = whisper_author_id
            reply_char_name = whisper_author_name

            ai_reply = generate_reply(
                text_provider, whisper_content, whisper_author_name,
                user_content, characters_md, reply_char_id, reply_char_name
            )

            if ai_reply:
                reply_time = now_dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
                replies_to_add.append({
                    "nickname": reply_char_name,
                    "content": ai_reply,
                    "timestamp": reply_time,
                    "author": reply_char_id,
                    "reply_to": user_nickname,
                })
                new_replies_added += 1
                print(f"  Generated reply for {whisper_id}: {ai_reply[:50]}...")

        # Add replies to the reply file using reply_utils.py
        if replies_to_add:
            replies_json = json.dumps(replies_to_add, ensure_ascii=False)
            stdout, rc = run_script([
                sys.executable,
                os.path.join(SCRIPT_DIR, "reply_utils.py"),
                "add", reply_file_path, whisper_id, replies_json
            ])
            if rc == 0:
                print(f"  Added {len(replies_to_add)} replies to {whisper_id}")
            else:
                print(f"  Error adding replies to {whisper_id}", file=sys.stderr)

    # Delete processed replies from D1
    if reply_ids_to_delete:
        print(f"Deleting {len(reply_ids_to_delete)} processed replies from D1")
        d1_client.delete_replies(reply_ids_to_delete)

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

    # Git commit and push
    git_commit_and_push(changes_made, args.dry_run)

    print(f"\n=== Whisper Runner finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
