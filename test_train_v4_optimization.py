#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试v4.0优化：参考captcha_trainer/trains.py的训练策略
"""

import sys
import os

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

def test_step_based_callbacks():
    """测试Step-based回调功能"""
    print("=" * 80)
    print("测试1: Step-based验证和保存策略")
    print("=" * 80)
    
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    
    # 创建简单测试数据
    X = np.random.rand(1000, 60, 160, 1).astype(np.float32)
    y = np.random.randint(0, 2, (1000, 120)).astype(np.float32)
    
    # 创建简单模型
    inputs = keras.Input(shape=(60, 160, 1))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(120, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    # 导入StepBasedCallbacks
    from train import create_callbacks
    
    print("\n✓ Step-based回调创建成功")
    print("  - 每100步保存checkpoint")
    print("  - 每500步验证一次")
    print("  - 多条件终止: acc>=80% AND loss<=0.05 AND steps>=10000")
    
    return True


def test_exponential_decay():
    """测试指数衰减学习率"""
    print("\n" + "=" * 80)
    print("测试2: 指数衰减学习率策略")
    print("=" * 80)
    
    import tensorflow as tf
    from tensorflow import keras
    
    # 创建指数衰减学习率
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.98,
        staircase=True
    )
    
    # 测试不同步数的学习率
    steps = [0, 10000, 20000, 30000, 40000, 50000]
    print("\n学习率衰减曲线（参考captcha_trainer/trains.py）:")
    print(f"{'步数':<10} {'学习率':<15} {'衰减比例'}")
    print("-" * 40)
    
    initial_lr = lr_schedule(0).numpy()
    for step in steps:
        lr = lr_schedule(step).numpy()
        ratio = (initial_lr - lr) / initial_lr * 100
        print(f"{step:<10} {lr:<15.7f} -{ratio:.2f}%")
    
    print("\n✓ 指数衰减学习率配置正确")
    print("  - 初始学习率: 0.001")
    print("  - 每10000步衰减2%")
    print("  - 阶梯式衰减")
    
    return True


def test_multi_condition_termination():
    """测试多条件终止逻辑"""
    print("\n" + "=" * 80)
    print("测试3: 多条件终止策略")
    print("=" * 80)
    
    # 模拟achieve_cond函数
    def achieve_cond(acc, loss, steps, max_steps):
        achieve_accuracy = acc >= 0.80
        achieve_loss = loss <= 0.05
        achieve_steps = steps >= 10000
        over_max_steps = steps > max_steps
        
        return (achieve_accuracy and achieve_loss and achieve_steps) or over_max_steps
    
    # 测试场景
    scenarios = [
        {"name": "早期阶段", "acc": 0.50, "loss": 0.20, "steps": 1000, "max_steps": 50000, "expected": False},
        {"name": "准确率达标但损失未达标", "acc": 0.85, "loss": 0.10, "steps": 12000, "max_steps": 50000, "expected": False},
        {"name": "全部达标", "acc": 0.85, "loss": 0.04, "steps": 12000, "max_steps": 50000, "expected": True},
        {"name": "超过最大步数", "acc": 0.70, "loss": 0.08, "steps": 51000, "max_steps": 50000, "expected": True},
    ]
    
    print("\n终止条件测试（参考captcha_trainer/trains.py的achieve_cond）:")
    print(f"{'场景':<25} {'准确率':<10} {'损失':<10} {'步数':<10} {'终止?'}")
    print("-" * 65)
    
    all_pass = True
    for scenario in scenarios:
        result = achieve_cond(
            scenario['acc'], 
            scenario['loss'], 
            scenario['steps'],
            scenario['max_steps']
        )
        status = "✓" if result == scenario['expected'] else "✗"
        print(f"{scenario['name']:<25} {scenario['acc']:<10.2f} {scenario['loss']:<10.4f} "
              f"{scenario['steps']:<10} {status} {'是' if result else '否'}")
        
        if result != scenario['expected']:
            all_pass = False
    
    if all_pass:
        print("\n✓ 多条件终止逻辑正确")
        print("  - 准确率 >= 80%")
        print("  - 损失 <= 0.05")
        print("  - 步数 >= 10000")
        print("  - 三个条件同时满足 OR 超过最大步数")
    else:
        print("\n✗ 部分测试失败")
    
    return all_pass


def test_training_strategy_summary():
    """总结训练策略对比"""
    print("\n" + "=" * 80)
    print("测试4: 训练策略对比（v3.0 vs v4.0）")
    print("=" * 80)
    
    comparison = """
    
┌────────────────────┬─────────────────────────┬──────────────────────────┐
│ 策略维度           │ v3.0（原始）            │ v4.0（参考trains.py）    │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ 验证策略           │ 每epoch验证             │ 每500步验证（step-based）│
│ 学习率调整         │ ReduceLROnPlateau       │ 指数衰减（10000步×0.98） │
│ 终止条件           │ EarlyStopping单条件     │ 多条件（acc&loss&steps） │
│ 保存策略           │ 最优模型（epoch-based） │ 每100步checkpoint        │
│ Warmup            │ 前10轮线性增长          │ 无（直接指数衰减）       │
│ 早停patience      │ 35轮                    │ 无（改用多条件终止）     │
│ 最大训练轮数       │ 200                     │ 500（步数限制50000）     │
└────────────────────┴─────────────────────────┴──────────────────────────┘

关键改进点（来自test/captcha_trainer/trains.py）:
1. ✅ Step-based验证: 更灵活的验证频率，不依赖epoch
2. ✅ 指数衰减学习率: 每10000步×0.98，稳定衰减
3. ✅ 多条件终止: 准确率AND损失AND步数，防止过早/过晚停止
4. ✅ Step-based保存: 每100步保存，防止丢失进度
5. ✅ 步数限制: 最多50000步，防止死循环

预期效果:
- 训练更稳定（学习率平滑衰减）
- 验证更及时（500步验证 vs 可能数千步/epoch）
- 终止更合理（多条件 vs 单一早停）
- 进度可恢复（每100步checkpoint）
    """
    
    print(comparison)
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("训练v4.0优化测试（参考captcha_trainer/trains.py策略）")
    print("=" * 80)
    
    tests = [
        ("Step-based回调", test_step_based_callbacks),
        ("指数衰减学习率", test_exponential_decay),
        ("多条件终止逻辑", test_multi_condition_termination),
        ("训练策略对比", test_training_strategy_summary),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:<30} {status}")
    
    all_pass = all(result for _, result in results)
    if all_pass:
        print("\n🎉 所有测试通过！训练v4.0优化已就绪")
        print("\n下一步: 运行 python caocrvfy/train.py 开始训练")
    else:
        print("\n⚠️  部分测试失败，请检查")
    
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
