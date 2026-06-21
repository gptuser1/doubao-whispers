#!/usr/bin/env python3
"""
构建前统一预处理脚本。
所有构建前需要执行的步骤都放在这里，CF Pages 构建命令只需要调用这一个脚本。

目前包含的预处理步骤：
1. 解压按月打包的配图 tar 文件到 static/images/
2. 拆分按月的动态 JSON 文件为独立的 Markdown 文件（供 Hugo 读取）

以后新增预处理步骤直接往这个脚本里加，不需要改构建命令。

用法：
    python preprocess.py
    python .whisper-task/scripts/preprocess.py
"""

import os
import sys
import subprocess
import tarfile

# 项目根目录（脚本在 .whisper-task/scripts/ 下，往上两级就是根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 路径配置
IMAGES_TAR_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')
STATIC_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'static', 'images')
EXPAND_SCRIPT = os.path.join(SCRIPT_DIR, 'expand_whispers.py')


def extract_image_tars():
    """
    解压 data/images/ 下所有 .tar 文件到 static/images/。
    头像文件不打包，直接放在 static/images/ 里。
    """
    if not os.path.exists(IMAGES_TAR_DIR):
        print(f'[images] 目录不存在: {IMAGES_TAR_DIR}，跳过图片解压')
        return True
    
    # 确保输出目录存在
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    
    tar_files = sorted([
        f for f in os.listdir(IMAGES_TAR_DIR)
        if f.endswith('.tar')
    ])
    
    if not tar_files:
        print(f'[images] 没有找到 tar 文件，跳过')
        return True
    
    print(f'[images] 找到 {len(tar_files)} 个图片 tar 文件，开始解压...')
    
    for tar_file in tar_files:
        tar_path = os.path.join(IMAGES_TAR_DIR, tar_file)
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(STATIC_IMAGES_DIR)
            print(f'  ✓ 已解压: {tar_file}')
        except Exception as e:
            print(f'  ✗ 解压失败 {tar_file}: {e}')
            return False
    
    print(f'[images] 图片解压完成')
    return True


def expand_whispers():
    """
    调用 expand_whispers.py 拆分动态 JSON 文件。
    """
    print(f'[whispers] 开始拆分动态 JSON 文件...')
    
    try:
        result = subprocess.run(
            [sys.executable, EXPAND_SCRIPT],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f'  ✗ expand_whispers 失败')
            print(result.stderr)
            return False
        
        # 统计生成了多少文件
        content_dir = os.path.join(PROJECT_ROOT, 'content', 'whispers')
        count = 0
        for root, dirs, files in os.walk(content_dir):
            count += len([f for f in files if f.endswith('.md')])
        
        print(f'  ✓ 已生成 {count} 个动态文件')
        return True
        
    except Exception as e:
        print(f'  ✗ 执行 expand_whispers 出错: {e}')
        return False


def main():
    print('=' * 50)
    print('开始构建前预处理')
    print('=' * 50)
    print()
    
    success = True
    
    # 步骤 1：解压图片
    if not extract_image_tars():
        success = False
    
    print()
    
    # 步骤 2：拆分动态 JSON
    if not expand_whispers():
        success = False
    
    print()
    print('=' * 50)
    if success:
        print('预处理完成 ✓')
    else:
        print('预处理部分失败 ✗')
    print('=' * 50)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
