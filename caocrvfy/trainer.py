#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块（参考captcha_trainer模块化设计）
功能：封装训练逻辑，确保功能单一性
"""

from tensorflow import keras
from core import config
from core.data_augmentation import create_augmented_dataset


class CaptchaTrainer:
    """
    验证码训练器
    
    参考：captcha_trainer/trains.py的Trains类
    职责：
    - 管理训练流程
    - 配置学习率策略
    - 执行模型训练
    """
    
    def __init__(self, model, use_exponential_decay=True):
        """
        初始化训练器
        
        参数:
            model: Keras模型
            use_exponential_decay: 是否使用指数衰减学习率
        """
        self.model = model
        self.use_exponential_decay = use_exponential_decay
        self.history = None
    
    def setup_learning_rate_schedule(self, train_data, batch_size):
        """
        配置学习率调度（参考trains.py的指数衰减策略）
        
        参数:
            train_data: 训练数据 (X, y)
            batch_size: 批次大小
        
        返回:
            学习率调度器或固定学习率
        """
        if not self.use_exponential_decay:
            return config.LEARNING_RATE
        
        print("\n🔄 使用指数衰减学习率（参考captcha_trainer/trains.py）")
        
        # 计算每个epoch的步数
        train_images, train_labels = train_data
        steps_per_epoch = len(train_images) // batch_size
        
        # 指数衰减：每10000步×0.98（参考trains.py）
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=10000,
            decay_rate=0.98,
            staircase=True
        )
        
        print(f"  初始学习率: {config.LEARNING_RATE}")
        print(f"  衰减策略: 每10000步 × 0.98")
        print(f"  每轮步数: {steps_per_epoch}")
        print()
        
        return lr_schedule
    
    def recompile_with_lr_schedule(self, lr_schedule, use_enhanced_model=True):
        """
        使用新的学习率调度重新编译模型
        
        参数:
            lr_schedule: 学习率调度器
            use_enhanced_model: 是否使用增强模型
        """
        if use_enhanced_model:
            from model_enhanced import compile_model
            self.model = compile_model(
                self.model,
                use_focal_loss=False,
                pos_weight=3.0,
                learning_rate=lr_schedule
            )
        else:
            from model import compile_model
            self.model = compile_model(self.model, learning_rate=lr_schedule)
    
    def prepare_datasets(self, train_data, val_data, batch_size):
        """
        准备训练和验证数据集
        
        参考：captcha_trainer/utils/data.py的数据管道
        
        参数:
            train_data: 训练数据 (X, y)
            val_data: 验证数据 (X, y)
            batch_size: 批次大小
        
        返回:
            (train_dataset, val_dataset)
        """
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        
        print("创建增强数据集...")
        train_dataset = create_augmented_dataset(
            train_images, train_labels,
            batch_size=batch_size,
            training=True
        )
        val_dataset = create_augmented_dataset(
            val_images, val_labels,
            batch_size=batch_size,
            training=False
        )
        print("✓ 数据增强pipeline已启用")
        print()
        
        return train_dataset, val_dataset
    
    def print_training_strategy(self):
        """
        打印训练策略信息
        
        参考：captcha_trainer的训练配置输出
        """
        print("=" * 80)
        print("训练策略（v4.0 - 完整参考captcha_trainer/trains.py）:")
        print("  🔧 核心策略（来自test/captcha_trainer）:")
        print("     - Step-based验证: 每500步验证一次（而非每epoch）")
        print("     - 指数衰减学习率: 每10000步 × 0.98（阶梯式衰减）")
        print("     - 多条件终止: 准确率>=80% AND 损失<=0.05 AND 步数>=10000")
        print("     - Step-based保存: 每100步保存checkpoint")
        print("  📊 数据处理:")
        print("     - 数据增强: 亮度/对比度变化 + 随机噪声")
        print("     - 批次大小: 128")
        print("  🎯 模型配置:")
        print("     - 正则化: BatchNorm + Dropout 0.25/0.5")
        print("     - 损失函数: WeightedBCE (pos_weight=3.0)")
        print("     - 优化器: Adam with AMSGrad")
        print("  ⏱️ 终止条件:")
        print("     - 完整匹配>=80% AND 损失<=0.05 AND 步数>=10000")
        print("     - 或超过最大步数50000（防止死循环）")
        print("=" * 80)
        print()
    
    def train(self, train_data, val_data, epochs=None, batch_size=None, callbacks=None):
        """
        执行训练
        
        参考：captcha_trainer/trains.py的train_process
        
        参数:
            train_data: 训练数据 (X, y)
            val_data: 验证数据 (X, y)
            epochs: 训练轮数
            batch_size: 批次大小
            callbacks: 回调函数列表
        
        返回:
            训练历史
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        # 配置学习率策略
        if self.use_exponential_decay:
            lr_schedule = self.setup_learning_rate_schedule(train_data, batch_size)
            self.recompile_with_lr_schedule(lr_schedule, use_enhanced_model=True)
        
        # 打印训练信息
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        
        print("\n" + "=" * 80)
        print(" " * 30 + "开始训练")
        print("=" * 80)
        print(f"训练样本数: {len(train_images)}")
        print(f"验证样本数: {len(val_images)}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数上限: {epochs}")
        print(f"初始学习率: {config.LEARNING_RATE}")
        print(f"优化器: Adam with AMSGrad")
        print("=" * 80)
        
        self.print_training_strategy()
        
        # 准备数据集
        train_dataset, val_dataset = self.prepare_datasets(
            train_data, val_data, batch_size
        )
        
        # 训练模型
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=2
        )
        
        return self.history
    
    def get_model(self):
        """获取训练后的模型"""
        return self.model
    
    def get_history(self):
        """获取训练历史"""
        return self.history
