#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证训练数据中是否包含数学题类型
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from caocrvfy.core.data_loader import CaptchaDataLoader
import os

def main():
    print("=" * 80)
    print(" " * 25 + "训练数据验证")
    print("=" * 80)
    print()
    
    # 使用本地测试目录
    test_captcha_dir = os.path.join(os.path.dirname(__file__), 'captcha', 'img')
    print(f"验证码目录: {test_captcha_dir}")
    print()
    
    # 加载数据
    loader = CaptchaDataLoader(captcha_dir=test_captcha_dir)
    count = loader.load_data()
    
    print()
    print("-" * 80)
    print("数据分析：")
    print("-" * 80)
    
    # 统计短标签（可能是数学题答案）
    short_labels = [l for l in loader.labels if len(l) <= 3]
    
    print(f"总样本数: {count}")
    print(f"短标签数量: {len(short_labels)} ({len(short_labels)/count*100:.1f}%)")
    
    if len(short_labels) > 0:
        print(f"\n⚠️ 警告：发现{len(short_labels)}个短标签，可能是数学题类型！")
        print(f"短标签示例（前20个）: {short_labels[:20]}")
        
        # 分析标签长度分布
        length_dist = {}
        for label in loader.labels:
            length = len(label)
            length_dist[length] = length_dist.get(length, 0) + 1
        
        print("\n标签长度分布:")
        for length in sorted(length_dist.keys()):
            count_len = length_dist[length]
            percentage = count_len / len(loader.labels) * 100
            print(f"  长度 {length}: {count_len} ({percentage:.1f}%)")
    else:
        print("\n✓ 未发现短标签，数据集正常")
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
