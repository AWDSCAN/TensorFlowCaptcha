#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学题验证码训练演示
展示新命名方式下的完整训练流程
"""

import os
import sys
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from caocrvfy.core.data_loader import CaptchaDataLoader
from caocrvfy.core import utils, config


def demonstrate_math_training():
    """演示数学题验证码的训练流程"""
    print("=" * 80)
    print(" " * 15 + "数学题验证码训练流程演示")
    print("=" * 80)
    print()
    
    # 使用测试数据
    test_captcha_dir = os.path.join(os.path.dirname(__file__), 'captcha', 'img')
    
    print("步骤 1: 加载数据")
    print("-" * 80)
    loader = CaptchaDataLoader(captcha_dir=test_captcha_dir)
    count = loader.load_data()
    
    print(f"✓ 加载了 {count} 张验证码")
    print()
    
    # 分析数据类型
    print("步骤 2: 分析数据类型")
    print("-" * 80)
    
    math_samples = []
    normal_samples = []
    
    for image_path, label_text in zip(loader.image_paths, loader.labels):
        filename = os.path.basename(image_path)
        
        # 检查是否为数学题类型（包含运算符）
        has_math_ops = any(op in label_text for op in ['+', '-', '*', '=', '?'])
        
        if has_math_ops:
            math_samples.append((filename, label_text))
        else:
            normal_samples.append((filename, label_text))
    
    print(f"✓ 数学题类型: {len(math_samples)} 张")
    print(f"✓ 普通类型: {len(normal_samples)} 张")
    print()
    
    # 展示数学题样本
    if len(math_samples) > 0:
        print("数学题样本:")
        for filename, label in math_samples:
            print(f"  • 文件名: {filename}")
            print(f"    标签: {label}")
            
            # 验证字符是否都在字符集中
            invalid_chars = [c for c in label if c not in config.CHAR_SET]
            if len(invalid_chars) > 0:
                print(f"    ✗ 包含非法字符: {invalid_chars}")
            else:
                print(f"    ✓ 所有字符都在字符集中")
            print()
    
    print()
    
    print("步骤 3: 标签向量化")
    print("-" * 80)
    
    # 展示一个数学题的标签向量化过程
    if len(math_samples) > 0:
        filename, label_text = math_samples[0]
        print(f"示例: {label_text}")
        print()
        
        # 转换为向量
        label_vector = utils.text_to_vector(label_text)
        
        print(f"标签长度: {len(label_text)}")
        print(f"填充后长度: {config.MAX_CAPTCHA}")
        print(f"向量维度: {label_vector.shape} = ({config.MAX_CAPTCHA} × {config.CHAR_SET_LEN})")
        print()
        
        # 显示每个字符的编码
        print("字符编码:")
        padded_text = label_text.ljust(config.MAX_CAPTCHA, config.PADDING_CHAR)
        for i, char in enumerate(padded_text):
            if char in config.CHAR_SET:
                char_idx = config.CHAR_SET.index(char)
                vector_idx = i * config.CHAR_SET_LEN + char_idx
                print(f"  位置 {i}: '{char}' → 字符集索引 {char_idx} → 向量位置 {vector_idx}")
            else:
                print(f"  位置 {i}: '{char}' → ✗ 不在字符集中")
        print()
        
        # 验证向量转回文本
        recovered_text = utils.vector_to_text(label_vector)
        print(f"向量转回文本: {recovered_text}")
        if recovered_text == label_text:
            print("✓ 向量化和反向量化正确")
        else:
            print(f"✗ 向量化错误: 期望 '{label_text}', 得到 '{recovered_text}'")
    
    print()
    print()
    
    print("步骤 4: 训练目标")
    print("-" * 80)
    print("对于数学题验证码:")
    print("  • 输入: 图片 (如显示 '3*5=?')")
    print("  • 标签: 题目文本 '3*5=?' (不是答案 '15')")
    print("  • 目标: 模型学习识别数学运算题本身")
    print()
    print("这样训练后，模型可以:")
    print("  1. 识别数学题的内容（包括运算符）")
    print("  2. 后续可以通过eval()计算答案")
    print("  3. 或者作为OCR使用，识别任意数学表达式")
    print()
    
    print("=" * 80)
    print("✅ 演示完成")
    print()
    print("📊 字符集统计:")
    print(f"  • 总字符数: {config.CHAR_SET_LEN}")
    print(f"  • 数字: 10 (0-9)")
    print(f"  • 字母: 52 (A-Z, a-z)")
    print(f"  • 空格: 1")
    print(f"  • 数学运算符: 5 (+, -, *, =, ?)")
    print()
    print("🎯 训练建议:")
    print("  1. 如果只需要答案，使用 generate_captcha_fixed.py (移除数学题)")
    print("  2. 如果需要识别题目，使用当前新命名方式")
    print("  3. 建议分开训练：数学题模型 + 普通验证码模型")
    print("=" * 80)


def show_file_format_examples():
    """展示文件格式示例"""
    print()
    print("=" * 80)
    print(" " * 20 + "文件命名格式对比")
    print("=" * 80)
    print()
    
    import base64
    
    examples = [
        {
            'type': '数学题（旧格式-错误）',
            'question': '19+3=?',
            'answer': '22',
            'old_filename': '22-abc123def456.png',
            'problem': '文件名是答案，图片是题目 → 无法训练'
        },
        {
            'type': '数学题（新格式-正确）',
            'question': '19+3=?',
            'answer': '22',
            'new_filename': base64.b64encode('19+3=?'.encode()).decode() + '_22_abc123def456.png',
            'benefit': '文件名包含题目(base64)和答案 → 可以训练'
        },
        {
            'type': '普通验证码',
            'text': 'abc123',
            'filename': 'abc123-xyz789.png',
            'note': '保持原有格式不变'
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['type']}")
        print("-" * 80)
        
        if 'question' in ex:
            print(f"   题目: {ex['question']}")
            print(f"   答案: {ex['answer']}")
        if 'text' in ex:
            print(f"   内容: {ex['text']}")
        
        if 'old_filename' in ex:
            print(f"   旧文件名: {ex['old_filename']}")
            print(f"   ✗ 问题: {ex['problem']}")
        
        if 'new_filename' in ex:
            print(f"   新文件名: {ex['new_filename']}")
            print(f"   ✓ 优势: {ex['benefit']}")
            
            # 解析演示
            parts = ex['new_filename'].replace('.png', '').split('_')
            if len(parts) == 3:
                b64, ans, hash_val = parts
                decoded = base64.b64decode(b64.encode()).decode()
                print(f"   解析:")
                print(f"     - base64部分: {b64} → 解码: {decoded}")
                print(f"     - 答案部分: {ans}")
                print(f"     - hash部分: {hash_val}")
        
        if 'filename' in ex:
            print(f"   文件名: {ex['filename']}")
            if 'note' in ex:
                print(f"   说明: {ex['note']}")
        
        print()
    
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_math_training()
    show_file_format_examples()
