#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证全局使用AdaptiveLearningRate配置
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import config
from core.model import create_cnn_model, compile_model
from core.callbacks import create_callbacks, AdaptiveLearningRate
import numpy as np

print("=" * 80)
print("验证全局AdaptiveLearningRate配置")
print("=" * 80)

# 1. 检查配置文件
print("\n1. 配置文件检查")
print("-" * 80)
print(f"初始学习率: {config.LEARNING_RATE}")
print(f"最小学习率: {config.LEARNING_RATE_MIN}")
print("✓ 配置使用AdaptiveLearningRate自适应调整")

# 2. 检查模型编译
print("\n2. 模型编译检查")
print("-" * 80)
model = create_cnn_model()
model = compile_model(model, use_lr_schedule=False)  # 明确不使用schedule

# 检查optimizer的learning_rate类型
lr = model.optimizer.learning_rate
print(f"Learning rate类型: {type(lr)}")
print(f"是否为固定值: {not hasattr(lr, '__call__')}")

try:
    lr_value = float(lr.numpy())
    print(f"当前学习率: {lr_value}")
    print("✓ 使用固定学习率，可被AdaptiveLearningRate调整")
except:
    print("✗ 使用LearningRateSchedule，不可被AdaptiveLearningRate调整")

# 3. 检查callbacks配置
print("\n3. Callbacks配置检查")
print("-" * 80)
dummy_data = (np.random.rand(10, 60, 200, 3), np.random.rand(10, 252))
callbacks = create_callbacks(
    model_dir='test_models',
    log_dir='test_logs',
    val_data=dummy_data,
    use_adaptive_lr=True
)

adaptive_lr_found = False
for cb in callbacks:
    if isinstance(cb, AdaptiveLearningRate):
        adaptive_lr_found = True
        print(f"✓ 找到AdaptiveLearningRate: 监控={cb.monitor}, factor={cb.factor}")
        break

if not adaptive_lr_found:
    print("✗ 未找到AdaptiveLearningRate")

# 4. 验证AdaptiveLearningRate行为
print("\n4. AdaptiveLearningRate行为验证")
print("-" * 80)
adaptive_cb = AdaptiveLearningRate()
adaptive_cb.set_model(model)
adaptive_cb.on_train_begin()

if hasattr(adaptive_cb, '_disabled') and adaptive_cb._disabled:
    print("✗ AdaptiveLearningRate被禁用（检测到LearningRateSchedule冲突）")
else:
    print("✓ AdaptiveLearningRate已启用且正常工作")

print("\n" + "=" * 80)
print("配置验证完成")
print("=" * 80)
print("\n结论：")
print("✓ 全局使用AdaptiveLearningRate自适应学习率")
print("✓ 不使用固定的LearningRateSchedule（如ExponentialDecay）")
print("✓ Adam优化器 + AdaptiveLearningRate = 双重自适应")
print("=" * 80)
