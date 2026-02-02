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

# 选择使用增强版模型还是ResNet-34模型
USE_ENHANCED_MODEL = False  # 改为False以使用ResNet-34

if USE_ENHANCED_MODEL:
    from extras.model_enhanced import create_enhanced_cnn_model as create_model
    from extras.model_enhanced import compile_model, print_model_summary
    print("使用增强版CNN模型（5层卷积 + BatchNorm + 更大FC层 + 数据增强 + Focal Loss）")
else:
    from core.model import create_cnn_model as create_model
    from core.model import compile_model, print_model_summary
    print("使用ResNet-34模型（34层残差网络 + LSTM + 自适应学习率）")


def save_model(model, save_path=None):
    """
    保存完整模型（.keras + checkpoint格式）
    
    生成文件:
    - crack_captcha_model.keras  （完整模型）
    - checkpoint                  （checkpoint元数据）
    - ckpt-1.index               （变量索引）
    - ckpt-1.data-00000-of-00001 （变量数据）
    
    参数:
        model: Keras模型
        save_path: 保存路径（可选，使用默认目录）
    """
    from core.model_saver import save_model_complete
    
    # 使用默认模型目录
    model_dir = save_path or config.MODEL_DIR
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)
    
    print(f"\n" + "=" * 80)
    print("正在保存模型...")
    print("=" * 80)
    
    # 保存完整模型
    saved_files = save_model_complete(model, model_dir, 'crack_captcha_model')
    
    print(f"\n✓ 模型保存完成！共 {len(saved_files)} 个文件:")
    print(f"  目录: {model_dir}")
    print("\n文件列表:")
    for filepath in saved_files:
        filename = os.path.basename(filepath)
        print(f"  ✓ {filename}")
    
    print("=" * 80)


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
    print("\n🎯 训练策略优化 v2 (余弦退火):")
    print("   - Focal Loss: 启用 (gamma=2.0, pos_weight=3.0)")
    print("   - 学习率策略: 余弦退火 (0.001 → 0.00001)")
    print("   - Warmup: 前5000步")
    print("   - 余弦周期: 150k步")
    print("   - 最大步数: 300000")
    print("   - 目标准确率: 80%")
    print("   - 预计时间: 4-6小时 (比之前快40%+)")
    
    # 优化策略组合：
    # 1. 使用Focal Loss处理困难样本（gamma=2.0，更关注错误样本）
    # 2. 增加pos_weight到3.5（进一步强调实际字符识别）
    print("🔧 优化配置：Focal Loss (gamma=2.0) + pos_weight=3.5")
    model = compile_model(model, use_focal_loss=True, pos_weight=3.5, focal_gamma=2.0)
    print_model_summary(model)
    print()
    
    # ========== 步骤4: 训练模型 ==========
    print("步骤 4/5: 训练模型")
    print("-" * 80)
    
    # 创建回调（模块化）- ResNet-34优化策略
    callbacks = create_callbacks(
        model_dir=config.MODEL_DIR,
        log_dir=config.LOG_DIR,
        val_data=(val_images, val_labels),
        use_step_based=True,  # 使用step-based策略
        use_early_stopping=False,  # 不使用早停
        use_adaptive_lr=True,  # ✅ 启用自适应学习率
        checkpoint_save_step=500,  # 每500步保存checkpoint
        validation_steps=300,  # 每300步验证
        max_checkpoints_keep=3,  # 只保留最近3个checkpoint
        end_acc=0.85,  # 目标准确率85%（ResNet-34更高目标）
        max_steps=200000  # 最大200000步
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
    evaluator = CaptchaEvaluator(
        model=trainer.get_model(),
        image_paths=loader.image_paths
    )
    
    # 生成评估报告
    metrics = evaluator.generate_report(
        val_data=(val_images, val_labels),
        include_math_validation=False
    )
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
