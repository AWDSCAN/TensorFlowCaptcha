#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试加权BCE Loss是否解决类别不平衡问题
"""
import sys
import os
sys.path.insert(0, 'caocrvfy')

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 导入模型
from model_enhanced import WeightedBinaryCrossentropy, compile_model, create_enhanced_cnn_model

print("=" * 80)
print(" " * 25 + "类别不平衡修复测试")
print("=" * 80)
print()

# 1. 测试WeightedBinaryCrossentropy
print("【1. 测试加权BCE Loss】")
print("-" * 80)

# 创建测试数据：模拟类别不平衡
y_true = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0],  # 1个正类，7个负类
                      [1, 1, 0, 0, 0, 0, 0, 0],  # 2个正类，6个负类
                      [1, 1, 1, 1, 0, 0, 0, 0]], dtype=tf.float32)  # 4个正类，4个负类

y_pred = tf.constant([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 完美预测
                      [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 完美预测
                      [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]], dtype=tf.float32)  # 完美预测

# 标准BCE
bce_standard = keras.losses.BinaryCrossentropy()
loss_standard = bce_standard(y_true, y_pred)
print(f"标准BCE Loss: {loss_standard.numpy():.6f}")

# 加权BCE (pos_weight=3.0)
bce_weighted = WeightedBinaryCrossentropy(pos_weight=3.0)
loss_weighted = bce_weighted(y_true, y_pred)
print(f"加权BCE Loss (pos_weight=3.0): {loss_weighted.numpy():.6f}")

print()
print("解释：")
print("  • 加权BCE给正类更高损失，迫使模型更关注实际字符")
print("  • pos_weight=3.0 意味着漏掉一个字符的惩罚是误判的3倍")
print()

# 2. 测试错误预测的惩罚
print("【2. 测试错误预测惩罚】")
print("-" * 80)

# 错误预测：模型过度预测负类（当前问题）
y_true_bad = tf.constant([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=tf.float32)
y_pred_bad = tf.constant([[0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1]], dtype=tf.float32)  # 召回率低

loss_standard_bad = bce_standard(y_true_bad, y_pred_bad)
loss_weighted_bad = bce_weighted(y_true_bad, y_pred_bad)

print(f"模型过度保守（召回率低）:")
print(f"  标准BCE Loss: {loss_standard_bad.numpy():.6f}")
print(f"  加权BCE Loss: {loss_weighted_bad.numpy():.6f}")
print(f"  惩罚提升: {(loss_weighted_bad / loss_standard_bad).numpy():.2f}x")
print()

# 3. 测试模型编译
print("【3. 测试模型编译】")
print("-" * 80)

try:
    model = create_enhanced_cnn_model()
    model = compile_model(model, use_focal_loss=False, pos_weight=3.0)
    
    print("✅ 模型编译成功")
    print(f"   损失函数: {model.loss.__class__.__name__}")
    print(f"   评估指标: {[m.name for m in model.metrics]}")
    
    # 检查是否有F1-score
    has_f1 = any('f1' in m.name for m in model.metrics)
    if has_f1:
        print("   ✓ 已添加F1-score指标")
    
    print()
    
    # 测试前向传播
    print("【4. 测试前向传播】")
    print("-" * 80)
    
    dummy_input = tf.random.normal((2, 60, 200, 3))  # 2个样本
    dummy_output = model(dummy_input, training=False)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {dummy_output.shape}")
    print(f"输出范围: [{dummy_output.numpy().min():.4f}, {dummy_output.numpy().max():.4f}]")
    print()
    
    # 统计预测分布
    predictions = (dummy_output.numpy() > 0.5).astype(int)
    positive_rate = predictions.sum() / predictions.size
    print(f"预测正类比例: {positive_rate:.2%} (随机初始化)")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print(" " * 30 + "测试总结")
print("=" * 80)
print()
print("预期效果:")
print("  • 召回率: 37% → 85%+")
print("  • 完整匹配: 0% → 60%+")
print("  • F1-score: 提供更平衡的评估")
print()
print("关键改进:")
print("  1. 加权BCE Loss - 正类权重3.0，强制关注实际字符")
print("  2. F1-score指标 - 平衡精确率和召回率")
print("  3. 阈值调整 - 所有指标使用0.5阈值")
print("=" * 80)
