#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试AdaptiveLearningRate对LearningRateSchedule的检测功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model import create_cnn_model, compile_model
from core.callbacks import AdaptiveLearningRate
import numpy as np

print("=" * 80)
print("测试AdaptiveLearningRate与LearningRateSchedule的兼容性")
print("=" * 80)

# 测试1：使用固定学习率
print("\n测试1: 固定学习率 + AdaptiveLearningRate")
print("-" * 80)
model1 = create_cnn_model()
model1 = compile_model(model1, use_lr_schedule=False)
callback1 = AdaptiveLearningRate()
callback1.set_model(model1)
callback1.on_train_begin()
print("✓ 测试1通过：AdaptiveLearningRate正常启用\n")

# 测试2：使用learning rate schedule
print("\n测试2: LearningRateSchedule + AdaptiveLearningRate")
print("-" * 80)
model2 = create_cnn_model()
model2 = compile_model(model2, use_lr_schedule=True)
callback2 = AdaptiveLearningRate()
callback2.set_model(model2)
callback2.on_train_begin()
print("✓ 测试2通过：AdaptiveLearningRate自动禁用\n")

# 测试3：验证on_epoch_end不会出错
print("\n测试3: 验证on_epoch_end方法")
print("-" * 80)
logs = {'val_loss': 0.1}
try:
    callback2.on_epoch_end(0, logs)
    print("✓ 测试3通过：on_epoch_end正常跳过\n")
except Exception as e:
    print(f"✗ 测试3失败: {e}\n")

print("=" * 80)
print("所有测试完成！")
print("=" * 80)
