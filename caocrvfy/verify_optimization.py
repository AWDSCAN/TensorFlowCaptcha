#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练优化验证脚本
功能：检查所有优化配置是否正确生效
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_config():
    """检查config.py配置"""
    from core import config
    
    print("=" * 80)
    print("检查 config.py 配置")
    print("=" * 80)
    
    checks = []
    
    # 学习率
    if config.LEARNING_RATE == 0.001:
        checks.append(("✓", "学习率", f"{config.LEARNING_RATE} (正确)"))
    else:
        checks.append(("✗", "学习率", f"{config.LEARNING_RATE} (应为0.001)"))
    
    # Warmup
    if config.WARMUP_LR_START >= 0.0001:
        checks.append(("✓", "Warmup起始", f"{config.WARMUP_LR_START} (正确)"))
    else:
        checks.append(("✗", "Warmup起始", f"{config.WARMUP_LR_START} (应>=0.0001)"))
    
    # 衰减因子
    if config.LR_DECAY_FACTOR == 0.7:
        checks.append(("✓", "衰减因子", f"{config.LR_DECAY_FACTOR} (正确)"))
    else:
        checks.append(("✗", "衰减因子", f"{config.LR_DECAY_FACTOR} (应为0.7)"))
    
    # 衰减耐心
    if config.LR_DECAY_PATIENCE >= 15:
        checks.append(("✓", "衰减耐心", f"{config.LR_DECAY_PATIENCE} (正确)"))
    else:
        checks.append(("✗", "衰减耐心", f"{config.LR_DECAY_PATIENCE} (应>=15)"))
    
    for symbol, item, value in checks:
        print(f"  {symbol} {item:12s}: {value}")
    
    print()


def check_trainer():
    """检查trainer.py配置"""
    print("=" * 80)
    print("检查 trainer.py 配置")
    print("=" * 80)
    
    # 读取文件内容
    trainer_path = os.path.join(os.path.dirname(__file__), 'trainer.py')
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = []
    
    # 检查Focal Loss
    if 'use_focal_loss=True' in content:
        checks.append(("✓", "Focal Loss", "已启用"))
    else:
        checks.append(("✗", "Focal Loss", "未启用 (应为True)"))
    
    # 检查focal_gamma
    if 'focal_gamma=2.0' in content:
        checks.append(("✓", "Focal Gamma", "2.0 (正确)"))
    elif 'focal_gamma=1.5' in content:
        checks.append(("✗", "Focal Gamma", "1.5 (应为2.0)"))
    else:
        checks.append(("?", "Focal Gamma", "未找到"))
    
    # 检查衰减步数
    if 'decay_steps=15000' in content:
        checks.append(("✓", "衰减步数", "15000 (正确)"))
    elif 'decay_steps=10000' in content:
        checks.append(("✗", "衰减步数", "10000 (应为15000)"))
    else:
        checks.append(("?", "衰减步数", "未找到"))
    
    # 检查衰减率
    if 'decay_rate=0.99' in content:
        checks.append(("✓", "衰减率", "0.99 (正确)"))
    elif 'decay_rate=0.98' in content:
        checks.append(("✗", "衰减率", "0.98 (应为0.99)"))
    else:
        checks.append(("?", "衰减率", "未找到"))
    
    for symbol, item, value in checks:
        print(f"  {symbol} {item:12s}: {value}")
    
    print()


def check_callbacks():
    """检查callbacks.py配置"""
    print("=" * 80)
    print("检查 callbacks.py 配置")
    print("=" * 80)
    
    # 读取文件内容
    callbacks_path = os.path.join(os.path.dirname(__file__), 'core', 'callbacks.py')
    with open(callbacks_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到StepBasedCallbacks的__init__行
    init_line = None
    for i, line in enumerate(lines):
        if 'def __init__(self, val_data, model_dir' in line and 'StepBasedCallbacks' in ''.join(lines[max(0, i-20):i]):
            init_line = line
            break
    
    checks = []
    
    if init_line:
        # 检查validation_steps
        if 'validation_steps=300' in init_line:
            checks.append(("✓", "验证间隔", "300步 (正确)"))
        elif 'validation_steps=500' in init_line:
            checks.append(("✗", "验证间隔", "500步 (应为300)"))
        else:
            checks.append(("?", "验证间隔", "未找到"))
        
        # 检查end_acc
        if 'end_acc=0.80' in init_line or 'end_acc=0.8' in init_line:
            checks.append(("✓", "目标准确率", "0.80 (正确)"))
        elif 'end_acc=0.95' in init_line:
            checks.append(("✗", "目标准确率", "0.95 (应为0.80)"))
        else:
            checks.append(("?", "目标准确率", "未找到"))
        
        # 检查end_loss
        if 'end_loss=0.02' in init_line:
            checks.append(("✓", "目标损失", "0.02 (正确)"))
        elif 'end_loss=0.01' in init_line:
            checks.append(("✗", "目标损失", "0.01 (应为0.02)"))
        else:
            checks.append(("?", "目标损失", "未找到"))
        
        # 检查max_steps
        if 'max_steps=300000' in init_line:
            checks.append(("✓", "最大步数", "300000 (正确)"))
        elif 'max_steps=50000' in init_line:
            checks.append(("✗", "最大步数", "50000 (应为300000)"))
        else:
            checks.append(("?", "最大步数", "未找到"))
    else:
        checks.append(("✗", "配置检查", "未找到StepBasedCallbacks.__init__"))
    
    for symbol, item, value in checks:
        print(f"  {symbol} {item:12s}: {value}")
    
    print()


def check_data_augmentation():
    """检查data_augmentation.py配置"""
    print("=" * 80)
    print("检查 data_augmentation.py 配置")
    print("=" * 80)
    
    # 读取文件内容
    aug_path = os.path.join(os.path.dirname(__file__), 'core', 'data_augmentation.py')
    with open(aug_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = []
    
    # 检查亮度调整
    if 'max_delta=0.12' in content:
        checks.append(("✓", "亮度调整", "±12% (正确)"))
    elif 'max_delta=0.10' in content:
        checks.append(("✗", "亮度调整", "±10% (应为±12%)"))
    else:
        checks.append(("?", "亮度调整", "未找到"))
    
    # 检查亮度概率
    if '>0.4' in content and 'random_brightness' in content:
        checks.append(("✓", "亮度概率", "60% (正确)"))
    elif '>0.5' in content and 'random_brightness' in content:
        checks.append(("✗", "亮度概率", "50% (应为60%)"))
    else:
        checks.append(("?", "亮度概率", "未确定"))
    
    # 检查对比度范围
    if 'lower=0.85, upper=1.15' in content:
        checks.append(("✓", "对比度范围", "85-115% (正确)"))
    elif 'lower=0.90, upper=1.10' in content:
        checks.append(("✗", "对比度范围", "90-110% (应为85-115%)"))
    else:
        checks.append(("?", "对比度范围", "未找到"))
    
    for symbol, item, value in checks:
        print(f"  {symbol} {item:12s}: {value}")
    
    print()


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" " * 25 + "训练优化配置验证")
    print(" " * 20 + "(2026-02-02 优化方案)")
    print("*" * 80)
    print()
    
    try:
        check_config()
        check_trainer()
        check_callbacks()
        check_data_augmentation()
        
        print("=" * 80)
        print("验证完成")
        print("=" * 80)
        print()
        print("如果所有检查项都是 ✓，则优化配置已正确应用")
        print("如果有 ✗ 标记，请检查对应文件并重新修改")
        print()
        print("下一步: 运行训练")
        print("  cd /data/coding/caocrvfy")
        print("  python train_v4.py")
        print()
        
    except Exception as e:
        print(f"\n✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
