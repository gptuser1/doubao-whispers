#!/usr/bin/env python3
"""
按月打包配图为 tar 文件。
扫描 static/images/ 下的配图，按月份分别打包到 data/images/ 目录。

用法：
    python pack_images.py [月份]
    python .whisper-task/scripts/pack_images.py

示例：
    # 打包所有月份
    python pack_images.py
    
    # 只打包 2026 年 6 月
    python pack_images.py 2026-06

说明：
- 只打包配图（文件名格式：YYYY-MM-DD-*.webp）
- 头像文件（avatar-*.webp）不打包
- 每个月生成一个 YYYY-MM.tar 文件
- 打包后的文件放在 data/images/ 目录
"""

import os
import sys
import tarfile
import re
from collections import defaultdict

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

STATIC_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'static', 'images')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')

# 配图文件名格式：YYYY-MM-DD-xxx.webp
IMAGE_PATTERN = re.compile(r'^(\d{4}-\d{2})-\d{2}-.+\.webp$')


def get_images_by_month():
    """
    扫描 static/images/ 下的配图，按月份分组。
    
    返回 dict: { '2026-06': ['file1.webp', 'file2.webp', ...], ... }
    """
    if not os.path.exists(STATIC_IMAGES_DIR):
        print(f'错误: 目录不存在: {STATIC_IMAGES_DIR}')
        return {}
    
    images = defaultdict(list)
    
    for filename in os.listdir(STATIC_IMAGES_DIR):
        match = IMAGE_PATTERN.match(filename)
        if match:
            month = match.group(1)
            images[month].append(filename)
    
    return images


def pack_month(month, files):
    """
    打包指定月份的图片为 tar 文件。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tar_path = os.path.join(OUTPUT_DIR, f'{month}.tar')
    
    try:
        with tarfile.open(tar_path, 'w') as tar:
            for filename in sorted(files):
                file_path = os.path.join(STATIC_IMAGES_DIR, filename)
                tar.add(file_path, arcname=filename)
        
        size_kb = os.path.getsize(tar_path) / 1024
        print(f'  ✓ {month}.tar ({len(files)} 张图, {size_kb:.1f} KB)')
        return True
        
    except Exception as e:
        print(f'  ✗ 打包失败 {month}: {e}')
        return False


def main():
    # 检查是否指定了月份
    target_month = sys.argv[1] if len(sys.argv) > 1 else None
    
    print('=' * 50)
    print('按月打包配图')
    print('=' * 50)
    print()
    
    images_by_month = get_images_by_month()
    
    if not images_by_month:
        print('没有找到配图文件')
        return 0
    
    if target_month:
        # 只打包指定月份
        if target_month not in images_by_month:
            print(f'错误: 没有找到 {target_month} 的配图')
            return 1
        
        print(f'打包 {target_month}...')
        success = pack_month(target_month, images_by_month[target_month])
    else:
        # 打包所有月份
        print(f'找到 {len(images_by_month)} 个月的配图，开始打包...')
        print()
        
        success = True
        for month in sorted(images_by_month.keys()):
            if not pack_month(month, images_by_month[month]):
                success = False
    
    print()
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 50)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
