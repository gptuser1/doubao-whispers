#!/usr/bin/env python3
"""
Timeline 查看工具
按时间倒序输出最近 N 条动态及其回复，用于生成上下文。
直接从 data/whispers/ 的按月 JSON 文件读取。
"""
import os
import sys
import json
import argparse
from datetime import datetime


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
    parser.add_argument('--whispers-dir', default='data/whispers', help='动态数据目录（JSON）')
    parser.add_argument('--replies-dir', default='data/replies', help='回复目录')
    parser.add_argument('--authors-file', default='data/authors.json', help='作者信息文件')
    args = parser.parse_args()
    
    whispers_dir = args.whispers_dir
    if not os.path.exists(whispers_dir):
        print(f'错误：动态数据目录 {whispers_dir} 不存在')
        sys.exit(1)
    
    # 1. 收集所有动态（按月份倒序读取，凑够就停）
    posts = []
    
    # 找所有月份文件，倒序排列（最新的月份在前）
    month_files = []
    for name in os.listdir(whispers_dir):
        if name.endswith('.json'):
            month_files.append(name)
    month_files.sort(reverse=True)
    
    for month_file in month_files:
        if len(posts) >= args.count * 2:  # 多读一些，按精确时间排序
            break
        
        month = month_file[:-5]  # 去掉 .json
        filepath = os.path.join(whispers_dir, month_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            whispers = json.load(f)
        
        for slug, whisper in whispers.items():
            date_str = whisper.get('date', '')
            if not date_str:
                continue
            
            whisper_id = f'{date_str[:10]}-{slug}'  # 用于匹配回复
            
            posts.append({
                'whisper_id': whisper_id,
                'slug': slug,
                'date': date_str,
                'title': whisper.get('title', ''),
                'author': whisper.get('author', ''),
                'body': whisper.get('content', '').strip(),
                'images': whisper.get('images', []),
                'year_month': month
            })
    
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
        images = post.get('images', [])
        if images and len(images) > 0:
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
