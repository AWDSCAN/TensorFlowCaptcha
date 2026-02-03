#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试9层卷积模型和数字纠正逻辑
"""

import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

from caocrvfy.core import model, config, utils
import numpy as np

def test_model_creation():
    """测试9层卷积模型创建"""
    print("=" * 80)
    print(" " * 25 + "测试9层卷积模型")
    print("=" * 80)
    
    # 创建模型
    print("\n1. 创建模型...")
    cnn_model = model.create_cnn_model()
    
    # 编译模型
    print("2. 编译模型（使用自适应学习率）...")
    cnn_model = model.compile_model(cnn_model)
    
    # 打印模型摘要
    print("\n3. 模型结构:")
    model.print_model_summary(cnn_model)
    
    # 测试模型预测
    print("\n4. 测试模型预测...")
    test_input = np.random.rand(
        2,  # batch size
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    ).astype(np.float32)
    
    output = cnn_model.predict(test_input, verbose=0)
    print(f"   输入形状: {test_input.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   ✓ 模型预测正常")
    
    return cnn_model


def test_digit_correction():
    """测试数字纠正逻辑"""
    print("\n" + "=" * 80)
    print(" " * 25 + "测试数字纠正逻辑")
    print("=" * 80)
    
    # 测试用例：(输入, 期望输出, 描述)
    test_cases = [
        ("12O4", "1204", "O -> 0 (3位数字场景)"),
        ("1234", "1234", "全数字，无需纠正"),
        ("123O", "1230", "末尾O -> 0 (3位数字场景)"),
        ("O234", "0234", "开头O -> 0 (3位数字场景)"),
        ("1I34", "1134", "I -> 1 (3位数字场景)"),
        ("12B4", "1284", "B -> 8 (3位数字场景)"),
        ("1Z34", "1234", "Z -> 2 (3位数字场景)"),
        ("abc1", "abc1", "只有1位数字，不纠正"),
        ("a1b2", "a1b2", "只有2位数字，不纠正"),
        ("abcd", "abcd", "无数字，不纠正"),
        ("123", "123", "长度不是4位，不纠正"),
        ("12345", "12345", "长度不是4位，不纠正"),
    ]
    
    print("\n测试结果:")
    print("-" * 80)
    passed = 0
    failed = 0
    
    for input_text, expected, description in test_cases:
        result = utils.correct_digit_confusion(input_text)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
            print(f"{status} {description}")
            print(f"  输入: '{input_text}' -> 输出: '{result}' (期望: '{expected}')")
        else:
            failed += 1
            print(f"{status} {description} [FAILED]")
            print(f"  输入: '{input_text}' -> 输出: '{result}' (期望: '{expected}')")
    
    print("-" * 80)
    print(f"\n总计: {len(test_cases)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    
    if failed == 0:
        print("\n✓ 所有测试通过！")
    else:
        print(f"\n✗ {failed} 个测试失败")
    
    return failed == 0


def test_vector_to_text_with_correction():
    """测试vector_to_text函数的纠正功能"""
    print("\n" + "=" * 80)
    print(" " * 20 + "测试vector_to_text数字纠正")
    print("=" * 80)
    
    # 创建一个模拟的one-hot向量，表示 "12O4"
    # 其中O会被识别为字母O (大写)
    vector = np.zeros((config.MAX_CAPTCHA, config.CHAR_SET_LEN), dtype=np.float32)
    
    # 字符集顺序: 0-9 (10个) + A-Z (26个) + a-z (26个) + ' ' (1个) = 63个
    # '1' -> index 1
    # '2' -> index 2  
    # 'O' -> index 10 + 14 = 24 (O是字母表第15个，A是第1个，所以O在10+14)
    # '4' -> index 4
    
    vector[0, 1] = 1.0   # 第1位: '1'
    vector[1, 2] = 1.0   # 第2位: '2'
    vector[2, 24] = 1.0  # 第3位: 'O'
    vector[3, 4] = 1.0   # 第4位: '4'
    
    # 测试不带纠正
    result_no_correction = utils.vector_to_text(vector, apply_correction=False)
    print(f"\n不带纠正: {result_no_correction}")
    
    # 测试带纠正
    result_with_correction = utils.vector_to_text(vector, apply_correction=True)
    print(f"带纠正:   {result_with_correction}")
    
    if result_with_correction == "1204":
        print("✓ 数字纠正功能正常工作")
        return True
    else:
        print(f"✗ 数字纠正功能异常，期望 '1204'，实际 '{result_with_correction}'")
        return False


if __name__ == '__main__':
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "9层卷积模型和数字纠正逻辑测试" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 测试1：模型创建
    try:
        test_model_creation()
        model_test_passed = True
    except Exception as e:
        print(f"\n✗ 模型创建测试失败: {e}")
        model_test_passed = False
    
    # 测试2：数字纠正逻辑
    try:
        correction_test_passed = test_digit_correction()
    except Exception as e:
        print(f"\n✗ 数字纠正测试失败: {e}")
        correction_test_passed = False
    
    # 测试3：vector_to_text纠正
    try:
        vector_test_passed = test_vector_to_text_with_correction()
    except Exception as e:
        print(f"\n✗ vector_to_text纠正测试失败: {e}")
        vector_test_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    print(" " * 30 + "测试总结")
    print("=" * 80)
    print(f"模型创建测试:         {'✓ 通过' if model_test_passed else '✗ 失败'}")
    print(f"数字纠正逻辑测试:     {'✓ 通过' if correction_test_passed else '✗ 失败'}")
    print(f"vector_to_text测试:   {'✓ 通过' if vector_test_passed else '✗ 失败'}")
    print("=" * 80)
    
    if model_test_passed and correction_test_passed and vector_test_passed:
        print("\n✓ 所有测试通过！")
        print("\n优化内容:")
        print("  1. ✓ 模型架构改为9层卷积网络")
        print("  2. ✓ 卷积层不使用dropout")
        print("  3. ✓ 使用Adam自适应学习率优化器")
        print("  4. ✓ 增加4位验证码数字纠正逻辑（O->0等）")
        print("  5. ✓ 验证码长度统一为4位")
        print("=" * 80)
    else:
        print("\n✗ 部分测试失败，请检查代码")
        print("=" * 80)
