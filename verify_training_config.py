#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练配置验证脚本
验证优化后的配置是否正确应用
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

import config
from model_enhanced import compile_model, create_enhanced_cnn_model
import inspect

print("=" * 80)
print(" " * 25 + "训练配置验证报告")
print("=" * 80)
print()

# 1. 验证损失函数配置
print("【1. 损失函数配置】")
print("-" * 80)
sig = inspect.signature(compile_model)
use_focal_loss_default = sig.parameters['use_focal_loss'].default

if use_focal_loss_default == False:
    print("✅ 损失函数: 标准 BCE Loss (正确)")
    print("   理由: GPU实测 BCE=75% > Focal Loss=52%")
else:
    print("❌ 损失函数: Focal Loss (错误)")
    print("   ⚠️  警告: 这会导致准确率下降至52%")
print()

# 2. 验证Dropout配置
print("【2. Dropout配置】")
print("-" * 80)
dropout_conv = getattr(config, 'DROPOUT_CONV', None) or config.DROPOUT_RATE
dropout_fc = getattr(config, 'DROPOUT_FC', None) or config.FC_DROPOUT_RATE
print(f"卷积层Dropout: {dropout_conv}")
print(f"全连接层Dropout: {dropout_fc}")

if dropout_conv <= 0.25 and dropout_fc <= 0.5:
    print("✅ Dropout配置合理（0.2/0.4 最优）")
else:
    print("⚠️  Dropout可能过高，建议降低至0.2/0.4")
print()

# 3. 验证早停配置
print("【3. 早停策略】")
print("-" * 80)
patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 20)
start_epoch = getattr(config, 'EARLY_STOPPING_START_EPOCH', 40)
print(f"耐心值: {patience} 轮")
print(f"延迟启动: 第 {start_epoch} 轮开始监控")

if patience >= 30 and start_epoch >= 50:
    print("✅ 早停配置合理（避免过早停止）")
elif patience >= 20:
    print("⚠️  建议提高耐心值至35，延迟启动至第50轮")
else:
    print("❌ 耐心值过低，可能导致训练不充分")
print()

# 4. 验证学习率配置
print("【4. 学习率配置】")
print("-" * 80)
lr = config.LEARNING_RATE
warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
warmup_lr = getattr(config, 'WARMUP_LR_START', lr)
print(f"初始学习率: {lr}")
print(f"Warmup轮数: {warmup_epochs}")
print(f"Warmup起始学习率: {warmup_lr}")

if 0.001 <= lr <= 0.0015:
    print("✅ 学习率在合理范围内")
else:
    print("⚠️  学习率建议范围: 0.001-0.0015")
print()

# 5. 验证批次大小
print("【5. 批次大小】")
print("-" * 80)
batch_size = config.BATCH_SIZE
print(f"Batch Size: {batch_size}")

if batch_size >= 64:
    print("✅ 批次大小合理（充分利用GPU）")
else:
    print("⚠️  建议提高至64或128以提升训练效率")
print()

# 6. 创建模型并检查
print("【6. 模型架构验证】")
print("-" * 80)
try:
    model = create_enhanced_cnn_model()
    model = compile_model(model, use_focal_loss=False)
    
    print("✅ 模型创建成功")
    print(f"   总参数: {model.count_params():,}")
    print(f"   输入形状: {model.input_shape}")
    print(f"   输出形状: {model.output_shape}")
    print()
    
    # 检查优化器配置
    optimizer = model.optimizer
    print(f"   优化器: {optimizer.__class__.__name__}")
    print(f"   AMSGrad: {optimizer.amsgrad}")
    print(f"   Clipnorm: {optimizer.clipnorm}")
    
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
print()

# 7. 总结
print("=" * 80)
print(" " * 30 + "验证总结")
print("=" * 80)
print()

all_ok = (
    use_focal_loss_default == False and
    dropout_conv <= 0.25 and dropout_fc <= 0.5 and
    patience >= 30 and
    0.001 <= lr <= 0.0015
)

if all_ok:
    print("🎉 所有配置检查通过！")
    print()
    print("预期训练效果:")
    print("  • 完整匹配准确率: 75-80%")
    print("  • 召回率: 90-95%")
    print("  • 精确率: 95-97%")
    print()
    print("可以开始训练:")
    print("  cd caocrvfy && python train.py")
else:
    print("⚠️  部分配置需要调整")
    print()
    print("建议修改:")
    if use_focal_loss_default != False:
        print("  • model_enhanced.py: use_focal_loss=False")
    if dropout_conv > 0.25 or dropout_fc > 0.5:
        print("  • config.py: DROPOUT_CONV=0.2, DROPOUT_FC=0.4")
    if patience < 30:
        print("  • config.py: EARLY_STOPPING_PATIENCE=35")

print("=" * 80)
