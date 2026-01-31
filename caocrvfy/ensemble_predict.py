#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成预测 - 使用多个checkpoint模型投票提升准确率

适用场景：当单模型达到82%左右时，使用此脚本可提升1-2%
使用方法：python ensemble_predict.py
"""

import os
import numpy as np
from tensorflow import keras
from core import config
from core.data_loader import CaptchaDataLoader
from core import utils


def load_models(checkpoint_dir, checkpoint_steps):
    """
    加载多个checkpoint模型
    
    参数:
        checkpoint_dir: checkpoint目录
        checkpoint_steps: 步数列表，如[145000, 148000, 150000]
    
    返回:
        模型列表
    """
    models = []
    
    for step in checkpoint_steps:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.keras')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  警告：{checkpoint_path} 不存在，跳过")
            continue
        
        print(f"加载模型: checkpoint_step_{step}.keras")
        model = keras.models.load_model(checkpoint_path)
        models.append(model)
    
    print(f"\n✓ 成功加载 {len(models)} 个模型")
    return models


def ensemble_predict(models, images, method='average'):
    """
    集成预测：多模型投票
    
    参数:
        models: 模型列表
        images: 输入图像
        method: 'average' - 平均概率, 'voting' - 硬投票
    
    返回:
        集成预测结果
    """
    all_predictions = []
    
    print(f"\n执行集成预测（方法：{method}）...")
    for i, model in enumerate(models):
        print(f"  模型{i+1}/{len(models)} 预测中...")
        pred = model.predict(images, verbose=0)
        all_predictions.append(pred)
    
    # 转为numpy数组：(n_models, batch_size, 504)
    all_predictions = np.array(all_predictions)
    
    if method == 'average':
        # 平均概率
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'voting':
        # 硬投票：>0.5为1
        binary_preds = (all_predictions > 0.5).astype(int)
        ensemble_pred = (np.sum(binary_preds, axis=0) > len(models) / 2).astype(float)
    else:
        raise ValueError(f"未知方法: {method}")
    
    return ensemble_pred


def evaluate_ensemble(models, val_images, val_labels, method='average'):
    """
    评估集成模型
    
    参数:
        models: 模型列表
        val_images: 验证图像
        val_labels: 验证标签
        method: 集成方法
    
    返回:
        准确率
    """
    # 集成预测
    ensemble_pred = ensemble_predict(models, val_images, method=method)
    
    # 计算完整匹配准确率
    pred_texts = [utils.vector_to_text(pred) for pred in ensemble_pred]
    true_texts = [utils.vector_to_text(label) for label in val_labels]
    accuracy = utils.calculate_accuracy(true_texts, pred_texts)
    
    # 显示前10个示例
    print("\n" + "=" * 80)
    print("集成预测示例（前10个）:")
    print("-" * 80)
    print(f"{'真实值':<20}{'预测值':<20}{'匹配':<10}")
    print("-" * 80)
    
    for i in range(min(10, len(true_texts))):
        match = "✓" if true_texts[i] == pred_texts[i] else "✗"
        print(f"{true_texts[i]:<20}{pred_texts[i]:<20}{match:<10}")
    
    print("=" * 80)
    
    return accuracy


def compare_methods(models, val_images, val_labels):
    """
    比较不同集成方法的效果
    """
    print("\n" + "=" * 80)
    print("比较不同集成方法")
    print("=" * 80)
    
    methods = ['average', 'voting']
    results = {}
    
    for method in methods:
        print(f"\n测试方法: {method}")
        accuracy = evaluate_ensemble(models, val_images, val_labels, method=method)
        results[method] = accuracy
        print(f"  完整匹配准确率: {accuracy:.2%}")
    
    # 找出最佳方法
    best_method = max(results, key=results.get)
    print("\n" + "=" * 80)
    print(f"最佳方法: {best_method} ({results[best_method]:.2%})")
    print("=" * 80)
    
    return results


def main():
    print("=" * 80)
    print("集成预测 - 多模型投票提升准确率")
    print("=" * 80)
    
    # 1. 加载验证数据
    print("\n步骤 1/4: 加载验证数据")
    print("-" * 80)
    
    loader = CaptchaDataLoader()
    loader.load_data()
    _, _, val_images, val_labels = loader.prepare_dataset()
    
    print(f"验证集大小: {len(val_images)}")
    
    # 2. 加载多个checkpoint模型
    print("\n步骤 2/4: 加载checkpoint模型")
    print("-" * 80)
    
    # 指定要集成的checkpoint步数（根据训练日志选择表现好的）
    checkpoint_steps = [145000, 148000, 150000, 155000, 160000]
    
    models = load_models(config.MODEL_DIR, checkpoint_steps)
    
    if len(models) < 2:
        print("\n❌ 错误：至少需要2个模型才能进行集成预测")
        print("   请检查models目录下是否有足够的checkpoint文件")
        return
    
    # 3. 评估单个模型（作为baseline）
    print("\n步骤 3/4: 评估单个模型（baseline）")
    print("-" * 80)
    
    for i, model in enumerate(models):
        pred = model.predict(val_images, verbose=0)
        pred_texts = [utils.vector_to_text(p) for p in pred]
        true_texts = [utils.vector_to_text(label) for label in val_labels]
        acc = utils.calculate_accuracy(true_texts, pred_texts)
        print(f"  模型{i+1} (step {checkpoint_steps[i]}): {acc:.2%}")
    
    # 4. 集成预测并比较方法
    print("\n步骤 4/4: 集成预测")
    print("-" * 80)
    
    results = compare_methods(models, val_images, val_labels)
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 集成预测总结")
    print("=" * 80)
    print(f"使用模型数量: {len(models)}")
    print(f"验证集大小: {len(val_images)}")
    print()
    
    for method, acc in results.items():
        print(f"  {method:10} : {acc:.2%}")
    
    # 计算提升
    best_single = max([
        utils.calculate_accuracy(
            [utils.vector_to_text(label) for label in val_labels],
            [utils.vector_to_text(p) for p in model.predict(val_images, verbose=0)]
        )
        for model in models
    ])
    
    best_ensemble = max(results.values())
    improvement = best_ensemble - best_single
    
    print()
    print(f"最佳单模型: {best_single:.2%}")
    print(f"最佳集成: {best_ensemble:.2%}")
    print(f"提升: {improvement:+.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()
