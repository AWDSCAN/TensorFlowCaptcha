#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练主程序（重构版 - 模块化设计）
功能：作为训练入口，协调各模块工作

模块化设计参考：test/captcha_trainer
- callbacks.py: 所有训练回调
- trainer.py: 训练逻辑封装
- evaluator.py: 评估逻辑封装
- train.py: 主程序入口（本文件）

优势：
1. 功能单一：每个模块职责明确
2. 易于维护：修改某功能只需改对应模块
3. 易于测试：可单独测试每个模块
4. 易于扩展：添加新功能不影响其他模块
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# 导入配置
from core import config

# 导入模块化组件（参考captcha_trainer设计）
from core.data_loader import CaptchaDataLoader
from core.callbacks import create_callbacks
from trainer import CaptchaTrainer
from core.evaluator import CaptchaEvaluator

# 选择使用增强版模型还是基础模型
USE_ENHANCED_MODEL = True

if USE_ENHANCED_MODEL:
    from extras.model_enhanced import create_enhanced_cnn_model as create_model
    from extras.model_enhanced import compile_model, print_model_summary
    print("使用增强版CNN模型（5层卷积 + BatchNorm + 更大FC层 + 数据增强）")
else:
    from model import create_cnn_model as create_model
    from model import compile_model, print_model_summary
    print("使用基础版CNN模型（3层卷积）")


def save_model(model, save_path=None):
    """
    保存模型
    
    参考：captcha_trainer/trains.py的compile_graph
    
    参数:
        model: Keras模型
        save_path: 保存路径
    """
    save_path = save_path or os.path.join(config.MODEL_DIR, 'final_model.keras')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    print(f"\n✓ 模型已保存到: {save_path}")
    
    # 保存模型大小
    model_size = os.path.getsize(save_path) / (1024 ** 2)
    print(f"模型文件大小: {model_size:.2f} MB")


def main():
    """
    主训练流程（模块化设计）
    
    参考：captcha_trainer/trains.py的train_process
    设计理念：每个步骤由专门的模块负责
    """
    print("=" * 80)
    print(" " * 25 + "验证码识别模型训练")
    print(" " * 20 + "（模块化设计 v4.0）")
    print("=" * 80)
    print()
    
    # ========== 步骤1: 加载数据 ==========
    print("步骤 1/5: 加载数据")
    print("-" * 80)
    loader = CaptchaDataLoader()
    loader.load_data()
    loader.print_statistics()
    print()
    
    # ========== 步骤2: 准备数据集 ==========
    print("步骤 2/5: 准备数据集")
    print("-" * 80)
    train_images, train_labels, val_images, val_labels = loader.prepare_dataset()
    print()
    
    # ========== 步骤3: 创建模型 ==========
    print("步骤 3/5: 创建模型")
    print("-" * 80)
    model = create_model()
    # 使用加权BCE Loss（pos_weight=3.0，解决类别不平衡）
    model = compile_model(model, use_focal_loss=False, pos_weight=3.0)
    print_model_summary(model)
    print()
    
    # ========== 步骤4: 训练模型 ==========
    print("步骤 4/5: 训练模型")
    print("-" * 80)
    
    # 创建回调（模块化）
    callbacks = create_callbacks(
        model_dir=config.MODEL_DIR,
        log_dir=config.LOG_DIR,
        val_data=(val_images, val_labels),
        use_step_based=True,  # 使用step-based策略（参考trains.py）
        use_early_stopping=False  # 不使用早停（已有多条件终止）
    )
    
    # 创建训练器（模块化）
    trainer = CaptchaTrainer(
        model=model,
        use_exponential_decay=True  # 使用指数衰减学习率
    )
    
    # 执行训练
    history = trainer.train(
        train_data=(train_images, train_labels),
        val_data=(val_images, val_labels),
        epochs=500,  # 500轮上限（step-based终止会提前停止）
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )
    print()
    
    # ========== 步骤5: 评估模型 ==========
    print("步骤 5/5: 评估模型")
    print("-" * 80)
    
    # 创建评估器（模块化）
    evaluator = CaptchaEvaluator(model=trainer.get_model())
    
    # 生成评估报告
    metrics = evaluator.generate_report(val_data=(val_images, val_labels))
    print()
    
    # 保存最终模型
    save_model(trainer.get_model())
    
    # ========== 训练完成 ==========
    print("\n" + "=" * 80)
    print(" " * 30 + "训练完成")
    print("=" * 80)
    print(f"\n最终验证集完整匹配准确率: {metrics['full_match_accuracy']*100:.2f}%")
    print("\n模块化设计优势:")
    print("  ✓ callbacks.py: 所有回调逻辑集中管理")
    print("  ✓ trainer.py: 训练流程清晰封装")
    print("  ✓ evaluator.py: 评估逻辑独立模块")
    print("  ✓ train.py: 简洁的入口程序")
    print()
    
    return trainer.get_model(), history, metrics


if __name__ == '__main__':
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ 检测到 {len(gpus)} 个GPU，已启用内存增长模式")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")
    else:
        print("未检测到GPU，将使用CPU训练")
    print()
    
    # 运行训练
    main()
