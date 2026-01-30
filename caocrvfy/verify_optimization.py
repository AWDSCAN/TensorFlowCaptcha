#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练配置验证脚本
验证优化后的训练配置是否正确加载
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model_enhanced import create_enhanced_cnn_model, compile_model

def verify_config():
    """验证配置参数"""
    print("=" * 80)
    print("配置参数验证")
    print("=" * 80)
    
    issues = []
    
    # 1. 学习率检查
    print(f"✓ 学习率: {config.LEARNING_RATE}")
    if config.LEARNING_RATE != 0.001:
        issues.append(f"⚠️  学习率应为 0.001，当前为 {config.LEARNING_RATE}")
    
    # 2. 批次大小检查
    print(f"✓ 批次大小: {config.BATCH_SIZE}")
    if config.BATCH_SIZE != 128:
        issues.append(f"⚠️  批次大小应为 128，当前为 {config.BATCH_SIZE}")
    
    # 3. 训练轮数检查
    print(f"✓ 训练轮数上限: {config.EPOCHS}")
    
    # 4. 早停耐心值检查
    print(f"✓ 早停耐心值: {config.EARLY_STOPPING_PATIENCE}")
    
    # 5. 图像参数检查
    print(f"✓ 图像尺寸: {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}x{config.IMAGE_CHANNELS}")
    
    # 6. 字符集检查
    print(f"✓ 字符集大小: {config.CHAR_SET_LEN}")
    print(f"✓ 验证码最大长度: {config.MAX_CAPTCHA}")
    print(f"✓ 输出层大小: {config.OUTPUT_SIZE} ({config.MAX_CAPTCHA} × {config.CHAR_SET_LEN})")
    
    print("=" * 80)
    
    if issues:
        print("⚠️  发现配置问题：")
        for issue in issues:
            print(f"  {issue}")
        print("=" * 80)
        return False
    else:
        print("✅ 所有配置参数正确！")
        print("=" * 80)
        return True


def verify_model():
    """验证模型创建和编译"""
    print("\n" + "=" * 80)
    print("模型验证")
    print("=" * 80)
    
    try:
        # 创建模型
        print("正在创建增强版CNN模型...")
        model = create_enhanced_cnn_model()
        print("✓ 模型创建成功")
        
        # 编译模型
        print("正在编译模型...")
        model = compile_model(model, learning_rate=config.LEARNING_RATE)
        print("✓ 模型编译成功")
        
        # 检查优化器配置
        optimizer = model.optimizer
        print(f"\n优化器配置：")
        print(f"  类型: {optimizer.__class__.__name__}")
        
        # 检查学习率
        try:
            lr = float(optimizer.learning_rate.numpy())
            print(f"  学习率: {lr}")
            if abs(lr - config.LEARNING_RATE) > 1e-6:
                print(f"  ⚠️  学习率不匹配！期望 {config.LEARNING_RATE}，实际 {lr}")
        except:
            print(f"  学习率: {optimizer.learning_rate}")
        
        # 检查AMSGrad
        if hasattr(optimizer, 'amsgrad'):
            print(f"  AMSGrad: {optimizer.amsgrad}")
            if not optimizer.amsgrad:
                print(f"  ⚠️  AMSGrad未启用！")
        
        # 检查梯度裁剪
        if hasattr(optimizer, 'clipnorm'):
            print(f"  梯度裁剪 (clipnorm): {optimizer.clipnorm}")
            if optimizer.clipnorm is None or optimizer.clipnorm == 0:
                print(f"  ⚠️  梯度裁剪未启用！")
        elif hasattr(optimizer, '_clipnorm'):
            print(f"  梯度裁剪 (clipnorm): {optimizer._clipnorm}")
        else:
            print(f"  ⚠️  无法检测梯度裁剪配置")
        
        # 检查损失函数
        print(f"\n损失函数：")
        print(f"  类型: {model.loss.__class__.__name__ if hasattr(model.loss, '__class__') else model.loss}")
        
        # 检查评估指标
        print(f"\n评估指标：")
        for metric in model.metrics:
            print(f"  - {metric.name}")
        
        # 模型参数统计
        print(f"\n模型参数：")
        total_params = model.count_params()
        print(f"  总参数量: {total_params:,}")
        print(f"  估计大小: {total_params * 4 / (1024**2):.2f} MB")
        
        print("=" * 80)
        print("✅ 模型验证成功！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


def verify_imports():
    """验证关键依赖导入"""
    print("\n" + "=" * 80)
    print("依赖验证")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow 版本: {tf.__version__}")
        
        # 检查GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ 检测到 {len(gpus)} 个GPU:")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
        
        # 检查Keras版本
        from tensorflow import keras
        print(f"✓ Keras 版本: {keras.__version__}")
        
        print("=" * 80)
        print("✅ 所有依赖正常！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 依赖验证失败: {e}")
        print("=" * 80)
        return False


def main():
    """主函数"""
    print("\n" + "=" * 100)
    print(" " * 35 + "训练配置验证工具")
    print(" " * 30 + "优化版本: 2026-01-30 v2.0")
    print("=" * 100)
    print()
    
    results = []
    
    # 1. 验证依赖
    results.append(("依赖导入", verify_imports()))
    
    # 2. 验证配置
    results.append(("配置参数", verify_config()))
    
    # 3. 验证模型
    results.append(("模型创建", verify_model()))
    
    # 总结
    print("\n" + "=" * 100)
    print(" " * 40 + "验证结果汇总")
    print("=" * 100)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name:.<50} {status}")
        if not result:
            all_passed = False
    
    print("=" * 100)
    
    if all_passed:
        print()
        print("🎉 所有验证通过！可以开始训练。")
        print()
        print("开始训练命令：")
        print("  python train.py")
        print()
        print("预期结果：")
        print("  - 初始学习率: 0.001")
        print("  - Warmup阶段: 15轮，从 0.0001 → 0.001")
        print("  - 批次大小: 128")
        print("  - 早停监控: 第50轮后启用，耐心值25")
        print("  - 完整匹配准确率: 目标 75-85%")
        print()
    else:
        print()
        print("⚠️  存在配置问题，请修复后重新验证。")
        print()
    
    print("=" * 100)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
