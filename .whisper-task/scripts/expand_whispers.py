#!/usr/bin/env python3
"""
拆分动态文件：把 data/whispers/ 下的按月 JSON 文件，
拆分成 content/whispers/ 下的独立 md 文件。
用于 Hugo 构建前的预处理。
"""

import os
import json
import sys
from pathlib import Path


def expand_whispers(data_dir='data/whispers', output_dir='content/whispers'):
    """拆分所有动态文件"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"错误：数据目录不存在：{data_dir}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    total = 0
    
    # 遍历所有 JSON 文件
    for json_file in sorted(data_path.glob('*.json')):
        month = json_file.stem  # YYYY-MM
        year, month_num = month.split('-')
        
        # 读取 JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            whispers = json.load(f)
        
        # 按月创建子目录
        month_dir = output_path / year / month_num
        month_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成每个 md 文件
        for slug, whisper in whispers.items():
            # 从 date 字段提取日期（YYYY-MM-DD）
            date = whisper.get('date', '')
            if date:
                date_part = date[:10]  # YYYY-MM-DD
            else:
                date_part = f"{year}-{month_num}-01"
            
            # 构建 front matter
            fm_lines = ['---']
            fm_lines.append(f'title: "{whisper.get("title", "")}"')
            fm_lines.append(f'date: {whisper.get("date", "")}')
            fm_lines.append(f'slug: "{slug}"')
            fm_lines.append(f'author: "{whisper.get("author", "")}"')
            
            # 图片
            images = whisper.get('images', [])
            if images:
                if len(images) == 1:
                    fm_lines.append(f'image: "{images[0]}"')
                else:
                    fm_lines.append('images:')
                    for img in images:
                        fm_lines.append(f'  - "{img}"')
            
            # 标签
            tags = whisper.get('tags', [])
            if tags:
                tags_str = ', '.join(f'"{t}"' for t in tags)
                fm_lines.append(f'tags: [{tags_str}]')
            
            fm_lines.append('---')
            fm_lines.append('')
            fm_lines.append(whisper.get('content', ''))
            
            # 写入文件
            filename = f'{date_part}-{slug}.md'
            filepath = month_dir / filename
            filepath.write_text('\n'.join(fm_lines), encoding='utf-8')
            
            total += 1
        
        print(f"✓ {month}：{len(whispers)} 条动态")
    
    print(f"\n合计：{total} 条动态已生成到 {output_dir}/")


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/whispers'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'content/whispers'
    expand_whispers(data_dir, output_dir)
