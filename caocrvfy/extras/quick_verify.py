#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练配置快速验证脚本（无需TensorFlow）
仅验证配置参数是否正确修改
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_config_file():
    """验证config.py文件内容"""
    print("=" * 80)
    print("验证 config.py 配置参数")
    print("=" * 80)
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'LEARNING_RATE = 0.001': '学习率应为 0.001',
        'BATCH_SIZE = 128': '批次大小应为 128',
        'EPOCHS = 150': '训练轮数上限应为 150',
    }
    
    results = []
    for pattern, description in checks.items():
        if pattern in content:
            print(f"✓ {description} - 已确认")
            results.append(True)
        else:
            print(f"✗ {description} - 未找到")
            results.append(False)
    
    print("=" * 80)
    return all(results)


def verify_train_file():
    """验证train.py文件内容"""
    print("\n" + "=" * 80)
    print("验证 train.py 训练策略")
    print("=" * 80)
    
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    
    with open(train_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'warmup_epochs=15': 'Warmup轮数应为 15',
        'target_lr=config.LEARNING_RATE': 'Warmup目标学习率应使用配置值',
        'start_lr=0.0001': 'Warmup起始学习率应为 0.0001',
        'start_epoch=50': '早停起始轮次应为 50',
        'patience=25': '早停耐心值应为 25',
        'factor=0.5': '学习率衰减因子应为 0.5',
        'patience=8': '学习率衰减耐心值应为 8',
    }
    
    results = []
    for pattern, description in checks.items():
        if pattern in content:
            print(f"✓ {description} - 已确认")
            results.append(True)
        else:
            print(f"✗ {description} - 未找到")
            results.append(False)
    
    print("=" * 80)
    return all(results)


def verify_model_file():
    """验证model_enhanced.py文件内容"""
    print("\n" + "=" * 80)
    print("验证 model_enhanced.py 模型配置")
    print("=" * 80)
    
    model_path = os.path.join(os.path.dirname(__file__), 'model_enhanced.py')
    
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'amsgrad=True': 'AMSGrad应启用',
        'clipnorm=1.0': '梯度裁剪应设为 1.0',
    }
    
    results = []
    for pattern, description in checks.items():
        if pattern in content:
            print(f"✓ {description} - 已确认")
            results.append(True)
        else:
            print(f"✗ {description} - 未找到")
            results.append(False)
    
    print("=" * 80)
    return all(results)


def print_training_strategy():
    """打印优化后的训练策略"""
    print("\n" + "=" * 80)
    print("优化后的训练策略总结")
    print("=" * 80)
    print()
    print("【阶段1：Warmup - 第1-15轮】")
    print("  学习率: 0.0001 → 0.001 (线性增长)")
    print("  目的: 平滑启动，避免初期震荡")
    print()
    print("【阶段2：主训练 - 第16-50轮】")
    print("  学习率: 0.001 (固定)")
    print("  批次大小: 128")
    print("  策略: 充分探索，不触发早停")
    print("  学习率衰减: 8轮无改进降低50%")
    print()
    print("【阶段3：精细调优 - 第51-150轮】")
    print("  早停监控: 启用，耐心值25轮")
    print("  学习率衰减: 持续监控")
    print("  双重保存: val_loss最优 + 完整匹配准确率最优")
    print()
    print("【优化器配置】")
    print("  类型: Adam")
    print("  AMSGrad: 启用 (更稳定)")
    print("  梯度裁剪: clipnorm=1.0 (防止梯度爆炸)")
    print()
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 100)
    print(" " * 30 + "训练配置快速验证工具")
    print(" " * 25 + "优化版本: 2026-01-30 v2.0")
    print("=" * 100)
    print()
    
    results = []
    
    # 1. 验证配置文件
    results.append(("config.py", verify_config_file()))
    
    # 2. 验证训练文件
    results.append(("train.py", verify_train_file()))
    
    # 3. 验证模型文件
    results.append(("model_enhanced.py", verify_model_file()))
    
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
        print("🎉 所有配置验证通过！")
        print()
        print_training_strategy()
        print()
        print("【GPU服务器训练步骤】")
        print()
        print("1. 上传代码到服务器:")
        print("   scp -r tensorflow_cnn_captcha user@server:/data/coding/")
        print()
        print("2. SSH登录并训练:")
        print("   ssh user@server")
        print("   cd /data/coding/caocrvfy")
        print("   python train.py")
        print()
        print("3. 预期效果:")
        print("   - 初始完整匹配准确率: 10-20%")
        print("   - 第30轮: 50-60%")
        print("   - 第60轮: 70-80%")
        print("   - 最终目标: 75-85%")
        print()
        print("4. 如果准确率仍低于70%，请检查:")
        print("   - 训练数据数量（建议10000+张）")
        print("   - 验证码干扰强度是否过大")
        print("   - 字符集是否匹配")
        print()
    else:
        print()
        print("⚠️  存在配置问题，请检查上述失败项。")
        print()
    
    print("=" * 100)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
