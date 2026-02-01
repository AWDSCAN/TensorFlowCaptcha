#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数学题验证码的新命名方式
验证base64编码/解码是否正常工作
"""

import os
import sys
import base64

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from captcha.generate_captcha import CaptchaGenerator
from caocrvfy.core import utils


def test_math_captcha_naming():
    """测试数学题验证码的命名和解析"""
    print("=" * 80)
    print(" " * 20 + "数学题验证码命名测试")
    print("=" * 80)
    print()
    
    # 创建数学题生成器
    generator = CaptchaGenerator(
        width=200,
        height=60,
        mode='pil',
        captcha_type='math'
    )
    
    print("测试1: 生成数学题验证码")
    print("-" * 80)
    
    # 生成多个数学题验证码测试
    test_cases = []
    for i in range(5):
        image, text, answer, filename = generator.generate_captcha()
        test_cases.append((text, answer, filename))
        
        print(f"\n第 {i+1} 个验证码:")
        print(f"  题目: {text}")
        print(f"  答案: {answer}")
        print(f"  文件名: {filename}")
        
        # 验证文件名格式
        if '_' in filename:
            parts = filename.replace('.png', '').split('_')
            if len(parts) == 3:
                base64_part, answer_part, hash_part = parts
                print(f"    ✓ 格式正确: base64={base64_part[:20]}... / 答案={answer_part} / hash={hash_part}")
                
                # 验证base64解码
                try:
                    decoded = base64.b64decode(base64_part.encode('utf-8')).decode('utf-8')
                    if decoded == text:
                        print(f"    ✓ base64解码正确: {decoded}")
                    else:
                        print(f"    ✗ base64解码错误: 期望 {text}, 得到 {decoded}")
                except Exception as e:
                    print(f"    ✗ base64解码失败: {e}")
            else:
                print(f"    ✗ 文件名格式错误: 应该有3部分，实际有{len(parts)}部分")
        else:
            print(f"    ✗ 文件名格式错误: 缺少下划线分隔符")
    
    print()
    print()
    
    print("测试2: 解析文件名")
    print("-" * 80)
    
    for text, answer, filename in test_cases:
        parsed_text = utils.parse_filename(filename)
        
        print(f"\n文件名: {filename}")
        print(f"  原始题目: {text}")
        print(f"  解析结果: {parsed_text}")
        
        if parsed_text == text:
            print(f"  ✓ 解析正确")
        else:
            print(f"  ✗ 解析错误")
    
    print()
    print()
    
    print("测试3: 对比普通类型")
    print("-" * 80)
    
    # 生成普通类型验证码对比
    normal_generator = CaptchaGenerator(
        width=200,
        height=60,
        mode='pil',
        captcha_type='mixed'
    )
    
    image, text, answer, filename = normal_generator.generate_captcha()
    
    print(f"\n普通验证码:")
    print(f"  内容: {text}")
    print(f"  文件名: {filename}")
    
    parsed_text = utils.parse_filename(filename)
    print(f"  解析结果: {parsed_text}")
    
    if parsed_text == text:
        print(f"  ✓ 解析正确")
    else:
        print(f"  ✗ 解析错误")
    
    print()
    print("=" * 80)
    print("✅ 测试完成")
    print()
    print("💡 新命名方式说明:")
    print("  • 数学题: base64(题目)_答案_hash.png")
    print("  • 普通类型: 内容-hash.png")
    print()
    print("🎯 训练时行为:")
    print("  • 数学题: 识别 '19+3=?' 图片 → 输出 '19+3=?' 文本")
    print("  • 普通类型: 识别 'abc123' 图片 → 输出 'abc123' 文本")
    print("=" * 80)


def test_character_set():
    """测试字符集是否包含数学运算符"""
    print()
    print("=" * 80)
    print(" " * 20 + "字符集测试")
    print("=" * 80)
    print()
    
    from caocrvfy.core import config
    
    print(f"字符集大小: {config.CHAR_SET_LEN}")
    print(f"字符集内容: {repr(config.CHAR_SET)}")
    print()
    
    # 检查数学运算符
    math_chars = ['+', '-', '*', '=', '?']
    print("数学运算符检查:")
    for char in math_chars:
        if char in config.CHAR_SET:
            idx = config.CHAR_SET.index(char)
            print(f"  ✓ '{char}' 在字符集中 (索引: {idx})")
        else:
            print(f"  ✗ '{char}' 不在字符集中")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    test_math_captcha_naming()
    test_character_set()
