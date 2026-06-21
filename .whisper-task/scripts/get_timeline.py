#!/usr/bin/env python3
"""
Timeline 查看工具
按时间倒序输出最近 N 条动态及其回复，用于生成上下文。
"""

import os
import sys
import json
import argparse
from datetime import datetime


def parse_front_matter(filepath):
    """解析 Markdown 文件的 front matter（简单 YAML 解析，无需额外依赖）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.startswith('---\n'):
        return {}, content
    
    end_pos = content.find('\n---\n', 4)
    if end_pos == -1:
        return {}, content
    
    fm_text = content[4:end_pos]
    body = content[end_pos + 5:]
    
    fm = {}
    current_key = None
    
    for line in fm_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        
        # 处理 key: value 格式
        if ':' in stripped and not stripped.startswith('-'):
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 去掉引号
            if len(value) >= 2:
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
            
            fm[key] = value
            current_key = key
        # 处理数组（简单处理，我们用不到太复杂的）
        elif stripped.startswith('- ') and current_key:
            if current_key not in fm:
                fm[current_key] = []
            elif not isinstance(fm[current_key], list):
                fm[current_key] = [fm[current_key]]
            item = stripped[2:].strip()
            if (item.startswith('"') and item.endswith('"')) or \
               (item.startswith("'") and item.endswith("'")):
                item = item[1:-1]
            fm[current_key].append(item)
    
    return fm, body


def get_author_nickname(authors, author_id):
    """根据作者 ID 获取昵称"""
    if not authors:
        return author_id
    
    if author_id in authors:
        return authors[author_id].get('name', author_id)
    
    return author_id


def format_time(iso_str):
    """格式化 ISO 时间为可读格式"""
    try:
        # 处理带时区的格式，比如 2026-06-21T12:15:00+08:00
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return iso_str


def format_time_short(iso_str):
    """只显示时分"""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime('%H:%M')
    except:
        return ''


def main():
    parser = argparse.ArgumentParser(description='查看最近动态 Timeline')
    parser.add_argument('--count', type=int, default=15, help='最近多少条动态，默认 15')
    parser.add_argument('--content-dir', default='content/whispers', help='动态目录')
    parser.add_argument('--replies-dir', default='data/replies', help='回复目录')
    parser.add_argument('--authors-file', default='data/authors.json', help='作者信息文件')
    args = parser.parse_args()
    
    content_dir = args.content_dir
    if not os.path.exists(content_dir):
        print(f'错误：动态目录 {content_dir} 不存在')
        sys.exit(1)
    
    # 1. 收集所有动态文件（按年月倒序遍历，凑够就停）
    posts = []
    stopped = False
    
    # 找所有年份目录，倒序排列
    years = []
    for name in os.listdir(content_dir):
        path = os.path.join(content_dir, name)
        if os.path.isdir(path) and name.isdigit():
            years.append(name)
    years.sort(reverse=True)
    
    for year in years:
        if stopped:
            break
        
        year_path = os.path.join(content_dir, year)
        
        # 找所有月份目录，倒序排列
        months = []
        for name in os.listdir(year_path):
            path = os.path.join(year_path, name)
            if os.path.isdir(path) and name.isdigit():
                months.append(name)
        months.sort(reverse=True)
        
        for month in months:
            if stopped:
                break
            
            month_path = os.path.join(year_path, month)
            
            # 找所有 .md 文件，按文件名倒序（日期新的在前）
            files = []
            for name in os.listdir(month_path):
                if name.endswith('.md') and name != '_index.md':
                    files.append(name)
            files.sort(reverse=True)
            
            for filename in files:
                if stopped:
                    break
                
                filepath = os.path.join(month_path, filename)
                whisper_id = filename[:-3]  # 去掉 .md
                
                fm, body = parse_front_matter(filepath)
                
                date_str = fm.get('date', '')
                if not date_str:
                    continue
                
                title = fm.get('title', '')
                author = fm.get('author', '')
                
                posts.append({
                    'whisper_id': whisper_id,
                    'date': date_str,
                    'title': title,
                    'author': author,
                    'body': body.strip(),
                    'filepath': filepath,
                    'year_month': f'{year}-{month}'
                })
                
                if len(posts) >= args.count * 2:  # 多读一些，因为还要按精确时间排序
                    stopped = True
                    break
    
    if not posts:
        print('没有找到动态')
        return
    
    # 2. 按精确时间倒序排序，取前 N 条
    posts.sort(key=lambda x: x['date'], reverse=True)
    posts = posts[:args.count]
    
    # 3. 读取作者信息
    authors = {}
    if os.path.exists(args.authors_file):
        with open(args.authors_file, 'r', encoding='utf-8') as f:
            authors = json.load(f)
    
    # 4. 读取回复文件（按月缓存）
    reply_cache = {}
    
    def get_replies(whisper_id, year_month):
        if year_month not in reply_cache:
            reply_file = os.path.join(args.replies_dir, f'{year_month}.json')
            if os.path.exists(reply_file):
                with open(reply_file, 'r', encoding='utf-8') as f:
                    reply_cache[year_month] = json.load(f)
            else:
                reply_cache[year_month] = {}
        
        return reply_cache[year_month].get(whisper_id, [])
    
    # 5. 统计配图情况
    posts_with_images = 0
    posts_without_images = 0
    
    for post in posts:
        # 重新解析 front matter 来判断有没有图（或者直接从之前的解析结果里拿）
        fm, _ = parse_front_matter(post['filepath'])
        has_image = False
        
        if 'images' in fm and fm['images']:
            if isinstance(fm['images'], list) and len(fm['images']) > 0:
                has_image = True
            elif isinstance(fm['images'], str) and fm['images'].strip():
                has_image = True
        
        if 'image' in fm and fm['image'] and fm['image'].strip():
            has_image = True
        
        if has_image:
            posts_with_images += 1
        else:
            posts_without_images += 1
    
    total = len(posts)
    text_only_ratio = posts_without_images / total * 100 if total > 0 else 0
    
    # 6. 格式化输出
    print(f'=== 最近 {len(posts)} 条动态 Timeline ===')
    print('（按时间倒序，最新在前）')
    print()
    
    for i, post in enumerate(posts, 1):
        nickname = get_author_nickname(authors, post['author'])
        time_str = format_time(post['date'])
        
        print('━' * 50)
        print(f'#{i}  {time_str}  {nickname}  《{post["title"]}》')
        print('━' * 50)
        print()
        print(post['body'])
        print()
        
        replies = get_replies(post['whisper_id'], post['year_month'])
        
        if replies:
            print(f'回复（{len(replies)}条）：')
            print()
            
            for reply in replies:
                reply_nick = reply.get('nickname', '匿名')
                reply_time = format_time_short(reply.get('timestamp', ''))
                floor = reply.get('floor', '')
                reply_to = reply.get('reply_to', '')
                reply_to_floor = reply.get('reply_to_floor', '')
                
                floor_str = f'#{floor}F' if floor else ''
                
                reply_header = f'  {floor_str}  {reply_nick}'
                if reply_time:
                    reply_header += f'  {reply_time}'
                
                if reply_to and reply_to_floor:
                    reply_header += f'  → 回复 @{reply_to} #{reply_to_floor}F'
                elif reply_to:
                    reply_header += f'  → 回复 @{reply_to}'
                
                print(reply_header)
                
                content = reply.get('content', '')
                # 缩进显示
                for line in content.split('\n'):
                    print(f'      {line}')
                
                print()
        else:
            print('暂无回复')
            print()
        
        print()
    
    # 7. 输出配图统计
    print('━' * 50)
    print(f'配图统计（最近 {total} 条）：')
    print(f'  有配图：{posts_with_images} 条')
    print(f'  纯文字：{posts_without_images} 条')
    print(f'  纯文字占比：{text_only_ratio:.1f}%')
    print()
    if text_only_ratio >= 30:
        print(f'  状态：纯文字比例超过 30%，新动态必须配图')
    else:
        print(f'  状态：纯文字比例在 30% 以内，正常')
    print()


if __name__ == '__main__':
    main()
