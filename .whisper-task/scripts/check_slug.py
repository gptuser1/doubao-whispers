#!/usr/bin/env python3
"""
检查 slug 在指定月份的动态 JSON 文件中是否唯一。
如果已存在，自动建议下一个可用的 slug（加序号）。

用法：
    python check_slug.py <month_json_file> <slug>

示例：
    python check_slug.py data/whispers/2026-06.json doro-orange

输出（JSON 格式）：
    - available: 是否可用（true/false）
    - slug: 建议使用的 slug（如果原 slug 可用就是原 slug，否则是自动生成的新 slug）
    - exists: 原 slug 是否存在
    - suggestion: 建议说明
"""

import json
import sys
import os
import re


def check_slug_unique(month_file, slug):
    """
    检查 slug 是否在指定月份的 JSON 文件中唯一。
    如果已存在，自动生成下一个可用的序号 slug。
    
    返回 dict：
        - available: 原 slug 是否可用
        - suggested_slug: 建议使用的 slug
        - exists: 原 slug 是否存在
        - message: 说明信息
    """
    # 如果文件不存在，slug 肯定可用
    if not os.path.exists(month_file):
        return {
            'available': True,
            'suggested_slug': slug,
            'exists': False,
            'message': f'文件 {month_file} 不存在，slug "{slug}" 可用'
        }
    
    # 读取文件
    try:
        with open(month_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {
            'available': False,
            'suggested_slug': slug,
            'exists': False,
            'message': f'读取文件失败: {e}'
        }
    
    # 检查 slug 是否存在
    if slug not in data:
        return {
            'available': True,
            'suggested_slug': slug,
            'exists': False,
            'message': f'slug "{slug}" 可用'
        }
    
    # slug 已存在，找下一个可用的序号
    # 先看看有没有类似 slug-2, slug-3 这样的
    base_slug = slug
    counter = 2
    
    # 如果原 slug 本身就带数字后缀（比如 doro-orange-2），从那个数字开始
    match = re.match(r'^(.+)-(\d+)$', slug)
    if match:
        base_slug = match.group(1)
        counter = int(match.group(2)) + 1
    
    # 找下一个可用的
    while f'{base_slug}-{counter}' in data:
        counter += 1
    
    suggested = f'{base_slug}-{counter}'
    
    return {
        'available': False,
        'suggested_slug': suggested,
        'exists': True,
        'message': f'slug "{slug}" 已存在，建议使用 "{suggested}"'
    }


def main():
    if len(sys.argv) < 3:
        print('用法: python check_slug.py <month_json_file> <slug>')
        print('示例: python check_slug.py data/whispers/2026-06.json doro-orange')
        sys.exit(1)
    
    month_file = sys.argv[1]
    slug = sys.argv[2]
    
    result = check_slug_unique(month_file, slug)
    
    # 输出 JSON 格式
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 如果 slug 已存在，返回非 0 退出码，方便脚本判断
    if result['exists']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
