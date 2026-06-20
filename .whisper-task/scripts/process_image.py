#!/usr/bin/env python3
"""
图片处理工具：缩放、转 WebP、压缩

用法：
    python process_image.py <input_path> <output_path> [options]

选项：
    --max-size INT    最大边像素，默认 1200
    --quality INT     WebP 质量，默认 80
    --max-kb INT      最大文件大小(KB)，默认 500，超过自动降质量
    --min-size INT    最小边像素，默认 600，低于这个值会警告
"""

import argparse
import os
from PIL import Image


def process_image(input_path, output_path, max_size=1200, quality=80, max_kb=500, min_size=600):
    """
    处理图片：缩放 + 转 WebP + 压缩

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        max_size: 最大边像素
        quality: 初始质量
        max_kb: 最大文件大小(KB)
        min_size: 最小边像素（低于会警告）
    """
    # 打开图片
    img = Image.open(input_path)

    # 计算缩放比例（按最大边等比缩放）
    if img.width > img.height:
        ratio = max_size / img.width
    else:
        ratio = max_size / img.height

    # 只有比目标大才缩放，小图不放大
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # 检查最小边
    min_side = min(img.width, img.height)
    if min_side < min_size:
        print(f"警告：图片最短边只有 {min_side}px，低于建议的 {min_size}px，可能会模糊")

    # 保存为 WebP，逐步降质量直到满足大小限制
    current_quality = quality
    while current_quality >= 50:
        img.save(output_path, 'WEBP', quality=current_quality)
        file_size_kb = os.path.getsize(output_path) / 1024

        if file_size_kb <= max_kb:
            break

        current_quality -= 10

    # 输出结果信息
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"处理完成：{output_path}")
    print(f"  尺寸：{img.width} x {img.height}")
    print(f"  质量：{current_quality}")
    print(f"  大小：{file_size_kb:.1f} KB")

    if file_size_kb > max_kb and current_quality <= 50:
        print(f"警告：即使质量降到 50，文件大小仍有 {file_size_kb:.1f} KB，超过 {max_kb} KB 限制")


def main():
    parser = argparse.ArgumentParser(description='图片处理工具：缩放、转 WebP、压缩')
    parser.add_argument('input', help='输入图片路径')
    parser.add_argument('output', help='输出图片路径（.webp）')
    parser.add_argument('--max-size', type=int, default=1200, help='最大边像素，默认 1200')
    parser.add_argument('--quality', type=int, default=80, help='WebP 质量，默认 80')
    parser.add_argument('--max-kb', type=int, default=500, help='最大文件大小(KB)，默认 500')
    parser.add_argument('--min-size', type=int, default=600, help='最小边像素警告阈值，默认 600')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在：{args.input}")
        return 1

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_image(
        args.input,
        args.output,
        max_size=args.max_size,
        quality=args.quality,
        max_kb=args.max_kb,
        min_size=args.min_size
    )

    return 0


if __name__ == '__main__':
    exit(main())
