#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试图片预处理效果
对比原始图片和去干扰后的图片
"""

import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from caocrvfy.core import utils


def test_preprocessing(image_path):
    """测试单张图片的预处理效果"""
    # 加载原始图片
    img_original = Image.open(image_path)
    
    # 不使用预处理
    img_array_raw = utils.load_image(image_path, use_preprocessing=False)
    
    # 使用预处理
    img_array_processed = utils.load_image(image_path, use_preprocessing=True)
    
    return img_original, img_array_raw, img_array_processed


def visualize_comparison(image_paths):
    """可视化对比多张图片的预处理效果"""
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        try:
            img_original, img_raw, img_processed = test_preprocessing(img_path)
            
            # 原始图片
            axes[i, 0].imshow(img_original)
            axes[i, 0].set_title(f'原始图片\n{filename}', fontsize=10)
            axes[i, 0].axis('off')
            
            # 不预处理（仅归一化）
            axes[i, 1].imshow(img_raw)
            axes[i, 1].set_title('训练输入（无预处理）\n带干扰线和噪点', fontsize=10)
            axes[i, 1].axis('off')
            
            # 预处理后
            axes[i, 2].imshow(img_processed)
            axes[i, 2].set_title('训练输入（预处理后）\n去除干扰，突出字符', fontsize=10)
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: preprocessing_comparison.png")
    plt.show()


def main():
    print("=" * 80)
    print(" " * 20 + "图片预处理效果测试")
    print("=" * 80)
    print()
    
    # 获取captcha/img目录下的图片
    captcha_dir = os.path.join(os.path.dirname(__file__), 'captcha', 'img')
    
    if not os.path.exists(captcha_dir):
        print(f"❌ 错误: 目录不存在 {captcha_dir}")
        print("请先运行: python captcha/generate_captcha.py")
        return
    
    image_files = [f for f in os.listdir(captcha_dir) if f.endswith('.png')]
    
    if len(image_files) == 0:
        print(f"❌ 错误: 目录中没有图片 {captcha_dir}")
        print("请先运行: python captcha/generate_captcha.py")
        return
    
    print(f"找到 {len(image_files)} 张验证码图片")
    print()
    
    # 选择几张不同类型的图片进行测试
    test_images = []
    
    # 选择数字类型
    for f in image_files:
        label = utils.parse_filename(f)
        if label.isdigit() and len(label) >= 4:
            test_images.append(os.path.join(captcha_dir, f))
            break
    
    # 选择字母类型
    for f in image_files:
        label = utils.parse_filename(f)
        if label.isalpha():
            test_images.append(os.path.join(captcha_dir, f))
            break
    
    # 选择混合类型
    for f in image_files:
        label = utils.parse_filename(f)
        if not label.isdigit() and not label.isalpha() and len(label) >= 4:
            test_images.append(os.path.join(captcha_dir, f))
            break
    
    if len(test_images) == 0:
        # 如果没有找到，随机选择前3张
        test_images = [os.path.join(captcha_dir, f) for f in image_files[:3]]
    
    print(f"测试图片数量: {len(test_images)}")
    for img_path in test_images:
        filename = os.path.basename(img_path)
        label = utils.parse_filename(filename)
        print(f"  • {filename} → 标签: {label}")
    print()
    
    print("正在生成对比图...")
    print()
    
    visualize_comparison(test_images)
    
    print()
    print("=" * 80)
    print("📊 预处理效果说明:")
    print("=" * 80)
    print("  • 左列: 原始图片（包含强干扰线和噪点）")
    print("  • 中列: 不预处理（当前训练使用）→ 学习困难")
    print("  • 右列: 预处理后（建议使用）→ 字符清晰，易于学习")
    print()
    print("💡 建议:")
    print("  1. 安装 opencv-python: pip install opencv-python")
    print("  2. 在 utils.load_image() 中默认启用预处理")
    print("  3. 预期准确率提升: +5-10%")
    print("=" * 80)


if __name__ == '__main__':
    main()
