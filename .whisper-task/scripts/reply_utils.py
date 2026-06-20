#!/usr/bin/env python3
"""
月回复文件操作工具

用法：
    # 追加回复
    python reply_utils.py add <month_file> <whisper_id> <replies_json>

    # 获取某条动态的回复
    python reply_utils.py get <month_file> <whisper_id>

    # 列出所有动态的回复数量
    python reply_utils.py list <month_file>

回复 JSON 格式：
    [
      {
        "nickname": "咕嘎",
        "content": "回复内容……",
        "timestamp": "2026-06-19T15:30:00+08:00",
        "author": "guga",
        "reply_to": "Doro",
        "reply_to_floor": 3
      },
      ...
    ]
"""

import argparse
import json
import os
import sys


def load_month_file(month_file):
    """读取月回复文件，不存在则返回空字典"""
    if not os.path.exists(month_file):
        return {}
    with open(month_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_month_file(month_file, data):
    """保存月回复文件"""
    # 确保目录存在
    dirname = os.path.dirname(month_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(month_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recalculate_floors(replies):
    """重新计算楼层号，按 timestamp 升序排序后从 1 开始编号"""
    # 按 timestamp 升序排序
    sorted_replies = sorted(replies, key=lambda r: r.get('timestamp', ''))

    # 重新编号
    for i, reply in enumerate(sorted_replies, 1):
        reply['floor'] = i

    return sorted_replies


def add_replies(month_file, whisper_id, new_replies):
    """
    追加回复到月文件

    Args:
        month_file: 月回复文件路径
        whisper_id: 动态 ID
        new_replies: 新回复数组
    """
    data = load_month_file(month_file)

    # 获取现有回复，没有则创建空数组
    existing_replies = data.get(whisper_id, [])

    # 合并回复
    all_replies = existing_replies + new_replies

    # 重新排序并计算楼层号
    all_replies = recalculate_floors(all_replies)

    # 更新数据
    data[whisper_id] = all_replies

    # 保存
    save_month_file(month_file, data)

    print(f"已追加 {len(new_replies)} 条回复到 {whisper_id}")
    print(f"  该动态总回复数：{len(all_replies)} 条")
    print(f"  月文件：{month_file}")


def get_replies(month_file, whisper_id):
    """获取某条动态的所有回复"""
    data = load_month_file(month_file)
    replies = data.get(whisper_id, [])

    if not replies:
        print(f"动态 {whisper_id} 暂无回复")
        return

    print(f"动态 {whisper_id} 共有 {len(replies)} 条回复：")
    print()
    for reply in replies:
        floor = reply.get('floor', '?')
        nickname = reply.get('nickname', '匿名')
        content = reply.get('content', '')
        reply_to = reply.get('reply_to', '')
        reply_to_floor = reply.get('reply_to_floor', '')

        prefix = f"#{floor}F {nickname}"
        if reply_to:
            prefix += f" → 回复 @{reply_to}"
            if reply_to_floor:
                prefix += f" #{reply_to_floor}F"

        print(f"{prefix}")
        print(f"  {content}")
        print()


def list_replies(month_file):
    """列出所有动态的回复数量"""
    data = load_month_file(month_file)

    if not data:
        print(f"月文件 {month_file} 暂无数据")
        return

    print(f"月文件 {month_file} 共有 {len(data)} 条动态：")
    print()
    for whisper_id, replies in sorted(data.items()):
        print(f"  {whisper_id}: {len(replies)} 条回复")


def main():
    parser = argparse.ArgumentParser(description='月回复文件操作工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # add 子命令
    add_parser = subparsers.add_parser('add', help='追加回复')
    add_parser.add_argument('month_file', help='月回复文件路径')
    add_parser.add_argument('whisper_id', help='动态 ID')
    add_parser.add_argument('replies_json', help='回复数组的 JSON 字符串')

    # get 子命令
    get_parser = subparsers.add_parser('get', help='获取某条动态的回复')
    get_parser.add_argument('month_file', help='月回复文件路径')
    get_parser.add_argument('whisper_id', help='动态 ID')

    # list 子命令
    list_parser = subparsers.add_parser('list', help='列出所有动态的回复数量')
    list_parser.add_argument('month_file', help='月回复文件路径')

    args = parser.parse_args()

    if args.command == 'add':
        try:
            new_replies = json.loads(args.replies_json)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败：{e}")
            return 1

        if not isinstance(new_replies, list):
            print("错误：replies_json 必须是数组")
            return 1

        add_replies(args.month_file, args.whisper_id, new_replies)

    elif args.command == 'get':
        get_replies(args.month_file, args.whisper_id)

    elif args.command == 'list':
        list_replies(args.month_file)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
