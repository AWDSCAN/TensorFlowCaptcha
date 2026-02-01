#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keras模型加载和测试脚本
用于本地验证训练效果

功能：
1. 加载已训练的Keras模型
2. 在验证集/测试集上评估性能
3. 显示预测示例和错误分析
4. 生成详细的评估报告

使用方法：
    # 测试final_model
    python test_model.py
    
    # 测试指定模型
    python test_model.py --model models/best_model.keras
    
    # 测试GPU服务器模型
    python test_model.py --model /data/coding/caocrvfy/core/models/final_model.keras
    
    # 显示更多示例
    python test_model.py --samples 50
    
    # 只显示错误示例
    python test_model.py --only-errors
"""

import os
import sys
import argparse
import numpy as np
from tensorflow import keras

from core import config
from core.data_loader import CaptchaDataLoader
from core import utils


def load_model(model_path):
    """加载Keras模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
    
    print(f"📥 加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    print("   ✓ 模型加载成功")
    
    # 显示模型信息
    print(f"\n📊 模型信息:")
    print(f"   输入形状: {model.input_shape}")
    print(f"   输出形状: {model.output_shape}")
    print(f"   参数量: {model.count_params():,}")
    
    # 计算模型大小
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   文件大小: {model_size:.2f} MB")
    
    return model


def evaluate_model(model, val_images, val_labels):
    """评估模型性能"""
    print("\n" + "=" * 70)
    print("📈 模型评估")
    print("=" * 70)
    
    # 1. 计算损失和指标
    print(f"\n计算验证集损失和指标...")
    results = model.evaluate(val_images, val_labels, verbose=0)
    
    # 获取指标名称
    metric_names = model.metrics_names
    
    print(f"\n验证集性能:")
    for name, value in zip(metric_names, results):
        if 'loss' in name:
            print(f"   {name}: {value:.6f}")
        else:
            print(f"   {name}: {value:.4f}")
    
    # 2. 计算完整匹配准确率
    print(f"\n计算完整验证码匹配准确率...")
    predictions = model.predict(val_images, verbose=0)
    
    pred_texts = [utils.vector_to_text(pred) for pred in predictions]
    true_texts = [utils.vector_to_text(label) for label in val_labels]
    
    full_match_accuracy = utils.calculate_accuracy(true_texts, pred_texts)
    
    print(f"\n✨ 完整匹配准确率: {full_match_accuracy:.4f} ({full_match_accuracy*100:.2f}%)")
    
    return {
        'loss': results[0],
        'binary_accuracy': results[1] if len(results) > 1 else None,
        'full_match_accuracy': full_match_accuracy,
        'predictions': pred_texts,
        'ground_truth': true_texts
    }


def analyze_errors(predictions, ground_truth, max_show=20):
    """分析预测错误"""
    print("\n" + "=" * 70)
    print("🔍 错误分析")
    print("=" * 70)
    
    # 统计错误
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
        if pred != true:
            errors.append((i, pred, true))
    
    error_rate = len(errors) / len(predictions) * 100
    
    print(f"\n错误统计:")
    print(f"   总样本数: {len(predictions)}")
    print(f"   错误数量: {len(errors)}")
    print(f"   错误率: {error_rate:.2f}%")
    
    if not errors:
        print("\n🎉 完美！所有预测都正确！")
        return
    
    # 错误类型分析
    error_types = {
        '字符混淆': 0,
        '空格问题': 0,
        '字符丢失': 0,
        '字符增加': 0,
        '完全错误': 0
    }
    
    for _, pred, true in errors:
        pred_clean = pred.replace(' ', '')
        true_clean = true.replace(' ', '')
        
        if ' ' in pred and ' ' not in true:
            error_types['空格问题'] += 1
        elif len(pred_clean) < len(true_clean):
            error_types['字符丢失'] += 1
        elif len(pred_clean) > len(true_clean):
            error_types['字符增加'] += 1
        elif abs(len(pred) - len(true)) <= 2:
            error_types['字符混淆'] += 1
        else:
            error_types['完全错误'] += 1
    
    print(f"\n错误类型分布:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = count / len(errors) * 100
            print(f"   {error_type}: {count} ({percentage:.1f}%)")
    
    # 显示错误示例
    print(f"\n错误示例（前{min(max_show, len(errors))}个）:")
    print("-" * 70)
    print(f"{'索引':<8}{'真实值':<20}{'预测值':<20}{'错误类型':<15}")
    print("-" * 70)
    
    for i, (idx, pred, true) in enumerate(errors[:max_show]):
        # 判断错误类型
        pred_clean = pred.replace(' ', '')
        true_clean = true.replace(' ', '')
        
        if ' ' in pred and ' ' not in true:
            error_type = '空格问题'
        elif len(pred_clean) < len(true_clean):
            error_type = '字符丢失'
        elif len(pred_clean) > len(true_clean):
            error_type = '字符增加'
        elif abs(len(pred) - len(true)) <= 2:
            error_type = '字符混淆'
        else:
            error_type = '完全错误'
        
        print(f"{idx:<8}{true:<20}{pred:<20}{error_type:<15}")
    
    if len(errors) > max_show:
        print(f"\n... 还有 {len(errors) - max_show} 个错误未显示")


def show_predictions(predictions, ground_truth, max_show=20, only_errors=False):
    """显示预测示例"""
    print("\n" + "=" * 70)
    print("📝 预测示例" + (" (仅错误)" if only_errors else ""))
    print("=" * 70)
    print(f"{'真实值':<20}{'预测值':<20}{'匹配':<10}")
    print("-" * 70)
    
    shown = 0
    for pred, true in zip(predictions, ground_truth):
        match = pred == true
        
        if only_errors and match:
            continue
        
        match_symbol = "✓" if match else "✗"
        print(f"{true:<20}{pred:<20}{match_symbol:<10}")
        
        shown += 1
        if shown >= max_show:
            break
    
    if shown < len(predictions):
        remaining = len(predictions) - shown
        print(f"\n... 还有 {remaining} 个样本未显示")


def generate_report(model_path, results, output_file=None):
    """生成评估报告"""
    print("\n" + "=" * 70)
    print("📄 生成评估报告")
    print("=" * 70)
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("模型评估报告")
    report_lines.append("=" * 70)
    report_lines.append(f"\n模型路径: {model_path}")
    report_lines.append(f"\n验证集大小: {len(results['ground_truth'])}")
    
    report_lines.append(f"\n性能指标:")
    report_lines.append(f"  验证集损失: {results['loss']:.6f}")
    if results['binary_accuracy']:
        report_lines.append(f"  二进制准确率: {results['binary_accuracy']:.4f}")
    report_lines.append(f"  完整匹配准确率: {results['full_match_accuracy']:.4f} ({results['full_match_accuracy']*100:.2f}%)")
    
    # 错误统计
    errors = sum(1 for p, t in zip(results['predictions'], results['ground_truth']) if p != t)
    report_lines.append(f"\n错误统计:")
    report_lines.append(f"  正确预测: {len(results['ground_truth']) - errors}")
    report_lines.append(f"  错误预测: {errors}")
    report_lines.append(f"  错误率: {errors / len(results['ground_truth']) * 100:.2f}%")
    
    report_lines.append("\n" + "=" * 70)
    
    # 打印到控制台
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 报告已保存: {output_file}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Keras模型测试脚本')
    parser.add_argument('--model', type=str, 
                        default='core/models/final_model.keras',
                        help='模型路径 (默认: core/models/final_model.keras)')
    parser.add_argument('--samples', type=int, default=20,
                        help='显示的示例数量 (默认: 20)')
    parser.add_argument('--only-errors', action='store_true',
                        help='只显示错误预测')
    parser.add_argument('--report', type=str, default=None,
                        help='保存评估报告到文件')
    parser.add_argument('--analyze-errors', action='store_true',
                        help='详细分析错误')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 Keras模型测试")
    print("=" * 70)
    
    try:
        # 1. 加载模型
        print("\n步骤 1/4: 加载模型")
        print("-" * 70)
        model = load_model(args.model)
        
        # 2. 加载验证数据
        print("\n步骤 2/4: 加载验证数据")
        print("-" * 70)
        loader = CaptchaDataLoader()
        loader.load_data()
        _, _, val_images, val_labels = loader.prepare_dataset()
        print(f"   验证集大小: {len(val_images)}")
        
        # 3. 评估模型
        print("\n步骤 3/4: 评估模型")
        print("-" * 70)
        results = evaluate_model(model, val_images, val_labels)
        
        # 4. 显示结果
        print("\n步骤 4/4: 显示结果")
        print("-" * 70)
        
        # 显示预测示例
        show_predictions(
            results['predictions'],
            results['ground_truth'],
            max_show=args.samples,
            only_errors=args.only_errors
        )
        
        # 错误分析
        if args.analyze_errors or args.only_errors:
            analyze_errors(
                results['predictions'],
                results['ground_truth'],
                max_show=args.samples
            )
        
        # 生成报告
        if args.report or True:  # 总是生成简要报告
            output_file = args.report or 'evaluation_report.txt'
            generate_report(args.model, results, output_file if args.report else None)
        
        print("\n" + "=" * 70)
        print("✅ 测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
