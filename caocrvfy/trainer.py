#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块（参考captcha_trainer模块化设计）
功能：封装训练逻辑，确保功能单一性
"""

import tensorflow as tf
from tensorflow import keras
from core import config
from core.data_augmentation import create_augmented_dataset


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup + 余弦退火学习率调度器
    
    前期: 线性增长（Warmup）
    后期: 余弦退火
    """
    
    def __init__(self, cosine_schedule, warmup_steps, warmup_lr_start):
        super().__init__()
        self.cosine_schedule = cosine_schedule
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.warmup_lr_start = tf.cast(warmup_lr_start, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup阶段: 线性增长
        warmup_lr = (
            self.warmup_lr_start + 
            (config.LEARNING_RATE - self.warmup_lr_start) * 
            (step / self.warmup_steps)
        )
        
        # 余弦退火阶段
        cosine_lr = self.cosine_schedule(step)
        
        # 前warmup_steps使用warmup_lr，之后使用cosine_lr
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: cosine_lr
        )
    
    def get_config(self):
        return {
            'cosine_schedule': self.cosine_schedule,
            'warmup_steps': self.warmup_steps,
            'warmup_lr_start': self.warmup_lr_start
        }


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
        配置学习率调度（余弦退火策略）
        
        余弦退火优势:
        1. 前期快速收敛（学习率从高到低）
        2. 后期精细优化（学习率接近最小值）
        3. 周期性回升可跳出局部最优
        4. 与Focal Loss完美搭配
        
        参数:
            train_data: 训练数据 (X, y)
            batch_size: 批次大小
        
        返回:
            学习率调度器
        """
        print("\n🔄 使用余弦退火学习率（Cosine Annealing with Warmup）")
        
        # 计算每个epoch的步数
        train_images, train_labels = train_data
        steps_per_epoch = len(train_images) // batch_size
        
        # 余弦退火 + Warmup
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=config.LEARNING_RATE,
            first_decay_steps=config.COSINE_DECAY_STEPS,
            t_mul=1.5,  # 每个周期增长1.5倍
            m_mul=0.9,  # 每个周期最大学习率衰减至0.9倍
            alpha=config.COSINE_ALPHA  # 最小学习率比例
        )
        
        # 包装Warmup
        if config.WARMUP_STEPS > 0:
            lr_schedule = WarmupCosineDecay(
                lr_schedule,
                warmup_steps=config.WARMUP_STEPS,
                warmup_lr_start=config.LEARNING_RATE_MIN
            )
        
        print(f"  初始学习率: {config.LEARNING_RATE}")
        print(f"  最小学习率: {config.LEARNING_RATE_MIN}")
        print(f"  Warmup步数: {config.WARMUP_STEPS}")
        print(f"  余弦周期: {config.COSINE_DECAY_STEPS}步")
        print(f"  每轮步数: {steps_per_epoch}")
        print(f"  预计100k步时学习率: ~0.0002（精细优化阶段）")
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
            from extras.model_enhanced import compile_model
            self.model = compile_model(
                self.model,
                use_focal_loss=True,      # 启用Focal Loss
                focal_gamma=2.0,          # 提升gamma到2.0
                pos_weight=3.0,
                learning_rate=lr_schedule
            )
        else:
            from core.model import compile_model
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
        print("训练策略（v4.1 - 余弦退火优化版）:")
        print("  🔧 核心策略:")
        print("     - Step-based验证: 每300步验证一次")
        print("     - 余弦退火学习率: 0.001 → 0.00001（周期性衰减）")
        print("     - Warmup: 前5000步线性增长")
        print("     - 多条件终止: 准确率>=80% AND 损失<=0.02 AND 步数>=10000")
        print("     - Step-based保存: 每100步保存checkpoint")
        print("  📊 数据处理:")
        print("     - 数据增强: 亮度±12% + 对比度85-115%")
        print("     - 批次大小: 128")
        print("  🎯 模型配置:")
        print("     - 正则化: BatchNorm + Dropout 0.25/0.5")
        print("     - 损失函数: Focal Loss (gamma=2.0) + WeightedBCE (pos_weight=3.0)")
        print("     - 优化器: Adam with AMSGrad")
        print("  ⏱️ 终止条件:")
        print("     - 完整匹配>=80% AND 损失<=0.02 AND 步数>=10000")
        print("     - 或超过最大步数300000")
        print("  ⚡ 预计训练时间: 4-6小时 (余弦退火收敛更快)")
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
