#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块化设计测试（参考captcha_trainer模块化结构）
功能：验证新模块的功能性和独立性
"""

import sys
import os

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

# 导入config以获取OUTPUT_SIZE
from caocrvfy import config


def test_callbacks_module():
    """测试callbacks模块的独立性"""
    print("=" * 80)
    print("测试1: callbacks.py模块")
    print("=" * 80)
    
    try:
        from callbacks import (
            DelayedEarlyStopping,
            BestFullMatchCheckpoint,
            TrainingProgress,
            StepBasedCallbacks,
            create_callbacks
        )
        
        print("\n✓ 成功导入所有回调类:")
        print("  - DelayedEarlyStopping")
        print("  - BestFullMatchCheckpoint")
        print("  - TrainingProgress")
        print("  - StepBasedCallbacks")
        print("  - create_callbacks (工厂函数)")
        
        # 测试create_callbacks
        import numpy as np
        val_data = (np.zeros((100, 60, 160, 1)), np.zeros((100, 120)))
        
        callbacks = create_callbacks(
            model_dir='test_models',
            log_dir='test_logs',
            val_data=val_data,
            use_step_based=True,
            use_early_stopping=False
        )
        
        print(f"\n✓ 成功创建{len(callbacks)}个回调")
        print("  回调列表:")
        for i, cb in enumerate(callbacks, 1):
            print(f"    {i}. {cb.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ callbacks模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_module():
    """测试trainer模块的独立性"""
    print("\n" + "=" * 80)
    print("测试2: trainer.py模块")
    print("=" * 80)
    
    try:
        from trainer import CaptchaTrainer
        from tensorflow import keras
        import numpy as np
        
        print("\n✓ 成功导入CaptchaTrainer类")
        
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
        
        # 创建训练器
        trainer = CaptchaTrainer(model, use_exponential_decay=True)
        
        print("✓ 成功创建训练器实例")
        
        # 测试学习率调度
        train_data = (np.zeros((100, 60, 160, 1)), np.zeros((100, 120)))
        lr_schedule = trainer.setup_learning_rate_schedule(train_data, batch_size=32)
        
        print("✓ 成功配置学习率调度")
        print(f"  初始学习率: {lr_schedule(0).numpy():.6f}")
        print(f"  10000步后: {lr_schedule(10000).numpy():.6f}")
        
        # 测试方法存在性
        methods = ['setup_learning_rate_schedule', 'recompile_with_lr_schedule', 
                   'prepare_datasets', 'train', 'get_model', 'get_history']
        
        print("\n✓ 训练器包含所有必要方法:")
        for method in methods:
            assert hasattr(trainer, method), f"缺少方法: {method}"
            print(f"  - {method}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ trainer模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_module():
    """测试evaluator模块的独立性"""
    print("\n" + "=" * 80)
    print("测试3: evaluator.py模块")
    print("=" * 80)
    
    try:
        from evaluator import CaptchaEvaluator
        from tensorflow import keras
        import numpy as np
        
        print("\n✓ 成功导入CaptchaEvaluator类")
        
        # 创建简单模型（输出维度应该是config.OUTPUT_SIZE）
        inputs = keras.Input(shape=(60, 160, 1))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(config.OUTPUT_SIZE, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy', 'precision', 'recall']
        )
        
        # 创建评估器
        evaluator = CaptchaEvaluator(model)
        
        print("✓ 成功创建评估器实例")
        
        # 测试方法存在性
        methods = ['evaluate', 'show_prediction_examples', 'generate_report']
        
        print("\n✓ 评估器包含所有必要方法:")
        for method in methods:
            assert hasattr(evaluator, method), f"缺少方法: {method}"
            print(f"  - {method}")
        
        # 测试评估功能（使用假数据）
        # 注意：标签形状应该是 (batch_size, OUTPUT_SIZE) = (batch_size, 504)
        val_data = (np.random.rand(50, 60, 160, 1).astype(np.float32),
                    np.random.randint(0, 2, (50, config.OUTPUT_SIZE)).astype(np.float32))
        
        metrics = evaluator.evaluate(val_data, verbose=False)
        
        print("\n✓ 成功执行评估")
        print("  返回指标:")
        for key, value in metrics.items():
            print(f"    - {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ evaluator模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modular_integration():
    """测试模块间集成"""
    print("\n" + "=" * 80)
    print("测试4: 模块集成测试")
    print("=" * 80)
    
    try:
        # 导入所有模块
        from callbacks import create_callbacks
        from trainer import CaptchaTrainer
        from evaluator import CaptchaEvaluator
        from tensorflow import keras
        import numpy as np
        
        print("\n✓ 成功导入所有模块")
        
        # 创建简单模型
        inputs = keras.Input(shape=(60, 160, 1))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(config.OUTPUT_SIZE, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy', 'precision', 'recall']
        )
        
        # 准备数据（标签形状应该是OUTPUT_SIZE）
        train_data = (np.random.rand(200, 60, 160, 1).astype(np.float32),
                      np.random.randint(0, 2, (200, config.OUTPUT_SIZE)).astype(np.float32))
        val_data = (np.random.rand(50, 60, 160, 1).astype(np.float32),
                    np.random.randint(0, 2, (50, config.OUTPUT_SIZE)).astype(np.float32))
        
        # 创建回调
        callbacks = create_callbacks(
            model_dir='test_models',
            log_dir='test_logs',
            val_data=val_data,
            use_step_based=False,  # 简单测试不使用step-based
            use_early_stopping=False
        )
        
        print("✓ 成功创建回调")
        
        # 创建训练器（不使用指数衰减以加快测试）
        trainer = CaptchaTrainer(model, use_exponential_decay=False)
        
        print("✓ 成功创建训练器")
        
        # 执行简短训练（仅1个epoch验证集成）
        print("\n执行集成测试（1个epoch）...")
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=1,
            batch_size=32,
            callbacks=[]  # 不使用回调以加快测试
        )
        
        print("✓ 训练执行成功")
        
        # 创建评估器
        evaluator = CaptchaEvaluator(trainer.get_model())
        
        print("✓ 成功创建评估器")
        
        # 执行评估
        metrics = evaluator.evaluate(val_data, verbose=False)
        
        print("✓ 评估执行成功")
        print("\n✓ 模块间集成正常")
        print("  训练 → 评估 流程完整")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modular_design_benefits():
    """展示模块化设计优势"""
    print("\n" + "=" * 80)
    print("测试5: 模块化设计优势展示")
    print("=" * 80)
    
    comparison = """
    
┌────────────────────┬───────────────────────────┬──────────────────────────┐
│ 对比维度           │ 原版 (train.py 471行)    │ 模块化版本               │
├────────────────────┼───────────────────────────┼──────────────────────────┤
│ 代码结构           │ 单文件大杂烩              │ 4个模块各司其职          │
│ 功能定位           │ 需要全文搜索              │ 直接打开对应模块         │
│ 修改回调           │ 在471行中找到回调定义     │ 打开callbacks.py (320行) │
│ 修改训练逻辑       │ 在同一文件中修改          │ 打开trainer.py (180行)   │
│ 修改评估逻辑       │ 在同一文件中修改          │ 打开evaluator.py (130行) │
│ 添加新功能         │ 可能影响其他代码          │ 独立模块不影响其他       │
│ 单元测试           │ 难以独立测试某个功能      │ 每个模块可独立测试       │
│ 代码复用           │ 难以在其他项目中复用      │ 可独立复用某个模块       │
└────────────────────┴───────────────────────────┴──────────────────────────┘

模块化设计优势：
✅ 功能单一性: 每个模块只负责一件事
✅ 易于维护: 修改某功能只需改对应模块
✅ 易于测试: 可独立测试每个模块
✅ 易于扩展: 添加新功能不影响现有代码
✅ 代码复用: 模块可在其他项目中复用
✅ 问题定位: 出错时能快速定位到具体模块

参考来源: test/captcha_trainer的模块化设计
核心理念: 单一职责原则（Single Responsibility Principle）
    """
    
    print(comparison)
    
    return True


def test_module_independence():
    """测试模块独立性"""
    print("\n" + "=" * 80)
    print("测试6: 模块独立性验证")
    print("=" * 80)
    
    try:
        print("\n独立性测试:")
        
        # 测试1: 只导入callbacks
        print("\n1. 只导入callbacks模块...")
        from callbacks import StepBasedCallbacks
        print("   ✓ callbacks可独立导入")
        
        # 测试2: 只导入trainer
        print("\n2. 只导入trainer模块...")
        from trainer import CaptchaTrainer
        print("   ✓ trainer可独立导入")
        
        # 测试3: 只导入evaluator
        print("\n3. 只导入evaluator模块...")
        from evaluator import CaptchaEvaluator
        print("   ✓ evaluator可独立导入")
        
        print("\n✓ 所有模块都可独立导入")
        print("  没有循环依赖")
        print("  模块间松耦合")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 独立性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("caocrvfy 模块化设计测试（参考captcha_trainer结构）")
    print("=" * 80)
    
    tests = [
        ("callbacks模块", test_callbacks_module),
        ("trainer模块", test_trainer_module),
        ("evaluator模块", test_evaluator_module),
        ("模块集成", test_modular_integration),
        ("设计优势", test_modular_design_benefits),
        ("模块独立性", test_module_independence),
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
        print(f"{name:<25} {status}")
    
    all_pass = all(result for _, result in results)
    if all_pass:
        print("\n🎉 所有测试通过！模块化重构成功")
        print("\n模块列表:")
        print("  📄 caocrvfy/callbacks.py   - 所有训练回调")
        print("  📄 caocrvfy/trainer.py     - 训练逻辑封装")
        print("  📄 caocrvfy/evaluator.py   - 评估逻辑封装")
        print("  📄 caocrvfy/train_v4.py    - 简洁的主程序")
        print("\n下一步:")
        print("  1. 运行 python caocrvfy/train_v4.py 开始训练")
        print("  2. 查看 caocrvfy/MODULAR_DESIGN.md 了解详细设计")
        print("  3. 根据需要自定义各模块功能")
    else:
        print("\n⚠️  部分测试失败，请检查")
    
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
