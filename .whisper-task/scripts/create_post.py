#!/usr/bin/env python3
"""
动态文件创建工具

用法：
    python create_post.py --date "2026-06-20T14:30:00+08:00" \\
                          --author guga \\
                          --title "今天吃了好吃的" \\
                          --content "今天吃了超好吃的包子！咕咕嘎嘎～" \\
                          [--images "/images/xxx-1.webp" "/images/xxx-2.webp"] \\
                          [--tags "美食" "日常"] \\
                          [--slug "custom-slug"] \\
                          [--output-dir "content/whispers"]

输出：会在 content/whispers/YYYY/MM/ 目录下创建 YYYY-MM-DD-{slug}.md 文件
"""

import argparse
import os
import re
from datetime import datetime


def slugify(title):
    """从标题生成 slug（简单版，英文数字和短横线）"""
    # 简单处理：转小写，替换非字母数字为短横线，去重，去首尾
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\u4e00-\u9fa5]+', '-', slug)
    slug = slug.strip('-')
    # 如果全是中文，就用拼音或者直接用日期？
    # 这里简单处理，如果结果为空就返回 post
    if not slug:
        slug = 'post'
    return slug


def create_post(date_str, author, title, content, images=None, tags=None, slug=None, output_dir='content/whispers'):
    """
    创建动态文件

    Args:
        date_str: 发布时间（ISO 格式）
        author: 作者 ID
        title: 标题
        content: 正文内容
        images: 配图路径数组
        tags: 标签数组
        slug: 自定义 slug，不填则从标题生成
        output_dir: 输出根目录

    Returns:
        创建的文件路径
    """
    # 解析日期
    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    year = date.strftime('%Y')
    month = date.strftime('%m')
    date_only = date.strftime('%Y-%m-%d')

    # 生成 slug
    if not slug:
        slug = slugify(title)

    # 文件名
    filename = f"{date_only}-{slug}.md"

    # 目录路径
    dir_path = os.path.join(output_dir, year, month)
    file_path = os.path.join(dir_path, filename)

    # 确保目录存在
    os.makedirs(dir_path, exist_ok=True)

    # 生成 front matter
    front_matter = []
    front_matter.append('---')
    front_matter.append(f'title: "{title}"')
    front_matter.append(f'date: {date_str}')
    front_matter.append(f'author: "{author}"')

    if images:
        if len(images) == 1:
            front_matter.append(f'image: "{images[0]}"')
        else:
            front_matter.append('images:')
            for img in images:
                front_matter.append(f'  - "{img}"')

    if tags:
        front_matter.append('tags: [' + ', '.join(f'"{t}"' for t in tags) + ']')

    front_matter.append('---')
    front_matter.append('')

    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(front_matter))
        f.write(content)
        if not content.endswith('\n'):
            f.write('\n')

    print(f"动态文件已创建：{file_path}")
    print(f"  标题：{title}")
    print(f"  作者：{author}")
    print(f"  日期：{date_str}")
    if images:
        print(f"  配图：{len(images)} 张")
    if tags:
        print(f"  标签：{', '.join(tags)}")

    return file_path


def main():
    parser = argparse.ArgumentParser(description='动态文件创建工具')
    parser.add_argument('--date', required=True, help='发布时间（ISO 格式）')
    parser.add_argument('--author', required=True, help='作者 ID')
    parser.add_argument('--title', required=True, help='标题')
    parser.add_argument('--content', required=True, help='正文内容')
    parser.add_argument('--images', nargs='*', help='配图路径（可多个）')
    parser.add_argument('--tags', nargs='*', help='标签（可多个）')
    parser.add_argument('--slug', help='自定义 slug，不填则从标题生成')
    parser.add_argument('--output-dir', default='content/whispers', help='输出根目录，默认 content/whispers')

    args = parser.parse_args()

    create_post(
        date_str=args.date,
        author=args.author,
        title=args.title,
        content=args.content,
        images=args.images,
        tags=args.tags,
        slug=args.slug,
        output_dir=args.output_dir
    )

    return 0


if __name__ == '__main__':
    exit(main())
