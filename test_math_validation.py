#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学题三步验证流程测试脚本

测试内容:
1. 验证utils中的数学题工具函数
2. 测试从文件名提取预期答案
3. 测试数学表达式计算
4. 测试完整三步验证流程

使用方法:
    cd tensorflow_cnn_captcha
    python test_math_validation.py
"""

import os
import sys

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

from core import utils


def test_is_math_expression():
    """测试数学表达式识别"""
    print("\n" + "="*80)
    print("测试1: 数学表达式识别 (is_math_expression)")
    print("="*80)
    
    test_cases = [
        ("19+3=?", True, "标准数学题"),
        ("5*6=?", True, "乘法题"),
        ("100-50=?", True, "减法题"),
        ("ABCD", False, "纯字母"),
        ("1234", False, "纯数字"),
        ("AB+CD", True, "包含运算符"),
    ]
    
    passed = 0
    for text, expected, description in test_cases:
        result = utils.is_math_expression(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: '{text}' -> {result} (预期: {expected})")
        if result == expected:
            passed += 1
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_extract_answer_from_filename():
    """测试从文件名提取答案"""
    print("\n" + "="*80)
    print("测试2: 从文件名提取答案 (extract_answer_from_filename)")
    print("="*80)
    
    test_cases = [
        ("MTkrMz0/_22_abc123.png", "22", "新格式数学题"),
        ("base64str_100_xyz789.png", "100", "大数字答案"),
        ("oldformat.png", None, "旧格式文件名"),
        ("22-hash.png", None, "旧数学题格式"),
    ]
    
    passed = 0
    for filename, expected, description in test_cases:
        result = utils.extract_answer_from_filename(filename)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: '{filename}' -> {result} (预期: {expected})")
        if result == expected:
            passed += 1
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_evaluate_math_expression():
    """测试数学表达式计算"""
    print("\n" + "="*80)
    print("测试3: 数学表达式计算 (evaluate_math_expression)")
    print("="*80)
    
    test_cases = [
        ("19+3=?", "22", "加法题"),
        ("19+3", "22", "无等号和问号"),
        ("100-50=?", "50", "减法题"),
        ("5*6=?", "30", "乘法题"),
        ("10/2=?", "5", "除法题"),
        ("2+3*4=?", "14", "运算顺序"),
        ("(2+3)*4=?", "20", "括号运算"),
        ("ABCD", None, "非数学表达式"),
        ("1; drop table", None, "SQL注入测试"),
    ]
    
    passed = 0
    for expression, expected, description in test_cases:
        result = utils.evaluate_math_expression(expression)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: '{expression}' -> {result} (预期: {expected})")
        if result == expected:
            passed += 1
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_validate_math_captcha():
    """测试完整三步验证流程"""
    print("\n" + "="*80)
    print("测试4: 完整三步验证流程 (validate_math_captcha)")
    print("="*80)
    
    test_cases = [
        # (识别结果, 预期答案, 预期最终结果, 描述)
        ("19+3=?", "22", True, "完全正确"),
        ("19+3", "22", True, "无等号问号但答案正确"),
        ("19+4=?", "22", False, "识别错误（题目错误）"),
        ("ABCD", "22", False, "非数学表达式"),
        ("5*6=?", "30", True, "乘法题正确"),
        ("100-50=?", "51", False, "答案不匹配"),
    ]
    
    passed = 0
    for predicted_text, expected_answer, expected_result, description in test_cases:
        result = utils.validate_math_captcha(predicted_text, expected_answer)
        
        is_correct = result['is_correct']
        status = "✓" if is_correct == expected_result else "✗"
        
        print(f"\n{status} {description}:")
        print(f"  识别: '{predicted_text}'")
        print(f"  预期答案: {expected_answer}")
        print(f"  步骤1 - 识别为数学题: {result['step1_recognized']}")
        print(f"  步骤2 - 计算结果: {result['step2_calculated']}")
        print(f"  步骤3 - 答案匹配: {result['step3_matched']}")
        print(f"  最终结果: {is_correct} (预期: {expected_result})")
        
        if is_correct == expected_result:
            passed += 1
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_integration_flow():
    """测试完整集成流程"""
    print("\n" + "="*80)
    print("测试5: 完整集成流程（模拟真实场景）")
    print("="*80)
    
    # 模拟场景: 从文件名到三步验证
    scenarios = [
        {
            'filename': 'MTkrMz0/_22_abc123.png',
            'predicted_text': '19+3=?',
            'description': '场景1: 数学题完全识别正确'
        },
        {
            'filename': 'MTkrMz0/_22_abc123.png',
            'predicted_text': '19+4=?',
            'description': '场景2: 数学题识别错误（题目识别错）'
        },
        {
            'filename': 'MTkrMz0/_22_abc123.png',
            'predicted_text': 'ABCD1234',
            'description': '场景3: 完全没有识别为数学题'
        },
        {
            'filename': 'NSpHPT0/_35_xyz789.png',
            'predicted_text': '5*7=?',
            'description': '场景4: 乘法题正确'
        },
    ]
    
    passed = 0
    total = len(scenarios)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*80}")
        print(f"{scenario['description']}")
        print(f"{'-'*80}")
        
        filename = scenario['filename']
        predicted_text = scenario['predicted_text']
        
        # 步骤1: 从文件名提取预期答案
        expected_answer = utils.extract_answer_from_filename(filename)
        print(f"文件名: {filename}")
        print(f"提取预期答案: {expected_answer}")
        
        if expected_answer is None:
            print("⚠️  无法从文件名提取答案，跳过验证")
            continue
        
        # 步骤2: 模型识别结果
        print(f"模型识别结果: '{predicted_text}'")
        
        # 步骤3: 三步验证
        validation = utils.validate_math_captcha(predicted_text, expected_answer)
        
        print(f"\n三步验证结果:")
        print(f"  步骤1 - 识别为数学题: {'✓' if validation['step1_recognized'] else '✗'} ({validation['step1_recognized']})")
        print(f"  步骤2 - 计算结果: {validation['step2_calculated'] or 'N/A'}")
        print(f"  步骤3 - 答案匹配: {'✓' if validation['step3_matched'] else '✗'} ({validation['step3_matched']})")
        print(f"  最终判定: {'✓ 正确' if validation['is_correct'] else '✗ 错误'}")
        
        # 判断是否符合预期（场景1应该正确，其他应该错误）
        expected_correct = (i == 1 or i == 4)
        if validation['is_correct'] == expected_correct:
            passed += 1
    
    print(f"\n通过: {passed}/{total}")
    return passed == total


def main():
    """运行所有测试"""
    print("\n" + "█"*80)
    print("█" + " "*30 + "数学题三步验证测试" + " "*29 + "█")
    print("█"*80)
    
    all_tests = [
        ("数学表达式识别", test_is_math_expression),
        ("文件名答案提取", test_extract_answer_from_filename),
        ("数学表达式计算", test_evaluate_math_expression),
        ("三步验证流程", test_validate_math_captcha),
        ("完整集成流程", test_integration_flow),
    ]
    
    results = []
    for test_name, test_func in all_tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "█"*80)
    print("█" + " "*33 + "测试总结" + " "*35 + "█")
    print("█"*80)
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！三步验证流程实现正确。")
        print("\n下一步:")
        print("  1. 生成新格式的数学题训练数据")
        print("  2. 运行 caocrvfy/train_v4.py 开始训练")
        print("  3. 训练完成后查看数学题三步验证准确率")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查实现。")
        return 1


if __name__ == '__main__':
    exit(main())
