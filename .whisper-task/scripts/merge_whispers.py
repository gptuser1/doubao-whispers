#!/usr/bin/env python3
"""
合并动态文件：把 content/whispers/ 下的 md 文件按月合并成 JSON 文件。
用于一次性迁移现有数据。
"""

import os
import re
import json
import sys
from pathlib import Path


def parse_front_matter(content):
    """解析 Markdown 文件的 front matter"""
    if not content.startswith('---'):
        return {}, content
    
    # 找到第二个 ---
    end = content.find('---', 3)
    if end == -1:
        return {}, content
    
    fm_text = content[3:end].strip()
    body = content[end+3:].strip()
    
    # 简单解析 YAML（只处理我们需要的字段，不用 PyYAML）
    fm = {}
    current_key = None
    current_list = None
    
    for line in fm_text.split('\n'):
        line = line.rstrip()
        if not line:
            continue
        
        # 列表项
        if line.startswith('  - ') or line.startswith('  - "'):
            if current_list is not None:
                val = line[4:].strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                current_list.append(val)
            continue
        
        # 普通键值对
        match = re.match(r'^(\w+):\s*(.*)$', line)
        if match:
            key = match.group(1)
            val = match.group(2).strip()
            
            # 去掉外层引号
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            
            # 内联列表：[item1, item2] 或 ["item1", "item2"]
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1].strip()
                if inner:
                    items = []
                    for item in inner.split(','):
                        item = item.strip()
                        if item.startswith('"') and item.endswith('"'):
                            item = item[1:-1]
                        items.append(item)
                    fm[key] = items
                else:
                    fm[key] = []
                current_key = key
                current_list = fm[key]
            elif val == '':
                # 多行列表，下一行开始
                fm[key] = []
                current_key = key
                current_list = fm[key]
            else:
                fm[key] = val
                current_key = key
                current_list = None
    
    return fm, body


def merge_whispers(content_dir='content/whispers', output_dir='data/whispers'):
    """合并所有动态文件"""
    content_path = Path(content_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按月份分组
    months = {}
    
    # 遍历所有 md 文件
    for md_file in sorted(content_path.rglob('*.md')):
        if md_file.name == '_index.md':
            continue
        
        # 从文件名提取日期和 slug
        # 格式：YYYY-MM-DD-slug.md
        name = md_file.stem
        match = re.match(r'^(\d{4}-\d{2})-\d{2}-(.+)$', name)
        if not match:
            print(f"跳过（文件名格式不对）：{md_file.name}")
            continue
        
        month = match.group(1)  # YYYY-MM
        slug = match.group(2)
        
        # 读取文件内容
        content = md_file.read_text(encoding='utf-8')
        fm, body = parse_front_matter(content)
        
        # 构建动态对象
        whisper = {
            'title': fm.get('title', ''),
            'date': fm.get('date', ''),
            'author': fm.get('author', ''),
            'content': body,
        }
        
        # 处理图片
        if 'images' in fm and fm['images']:
            whisper['images'] = fm['images']
        elif 'image' in fm and fm['image']:
            whisper['images'] = [fm['image']]
        
        # 处理标签
        if 'tags' in fm and fm['tags']:
            whisper['tags'] = fm['tags']
        
        # 加入对应月份
        if month not in months:
            months[month] = {}
        months[month][slug] = whisper
    
    # 写入 JSON 文件
    for month, whispers in sorted(months.items()):
        output_file = output_path / f'{month}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(whispers, f, ensure_ascii=False, indent=2)
        
        count = len(whispers)
        print(f"✓ {month}.json：{count} 条动态")
    
    total = sum(len(v) for v in months.values())
    print(f"\n合计：{len(months)} 个月，{total} 条动态")


if __name__ == '__main__':
    content_dir = sys.argv[1] if len(sys.argv) > 1 else 'content/whispers'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/whispers'
    merge_whispers(content_dir, output_dir)
