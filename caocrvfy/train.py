#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块
功能：训练验证码识别模型
"""

import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import config
from data_loader import CaptchaDataLoader
import utils

# 选择使用增强版模型还是基础模型
USE_ENHANCED_MODEL = True  # 改为True使用增强版模型

if USE_ENHANCED_MODEL:
    from model_enhanced import create_enhanced_cnn_model as create_model
    from model_enhanced import compile_model, print_model_summary
    print("使用增强版CNN模型（5层卷积 + BatchNorm + 更大FC层）")
else:
    from model import create_cnn_model as create_model
    from model import compile_model, print_model_summary
    print("使用基础版CNN模型（3层卷积）")


def create_callbacks(model_dir=None, log_dir=None, val_data=None):
    """
    创建训练回调函数
    
    参数:
        model_dir: 模型保存目录
        log_dir: 日志保存目录
        val_data: 验证数据 (X, y)，用于计算完整匹配准确率
    
    返回:
        回调函数列表
    """
    model_dir = model_dir or config.MODEL_DIR
    log_dir = log_dir or config.LOG_DIR
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = []
    
    # 模型检查点：保存最优模型（监控val_loss更可靠）
    checkpoint_path = os.path.join(model_dir, 'best_model.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',  # 改为监控损失
        mode='min',  # 损失越小越好
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 延迟早停：前85轮充分训练，之后才启用早停监控
    class DelayedEarlyStopping(keras.callbacks.EarlyStopping):
        """延迟早停回调：在指定轮次之前不触发早停"""
        def __init__(self, start_epoch=85, **kwargs):
            super().__init__(**kwargs)
            self.start_epoch = start_epoch
            self.delayed_mode = True  # 标记是否处于延迟模式
        
        def on_epoch_end(self, epoch, logs=None):
            # 只在达到start_epoch后才调用父类的早停逻辑
            if epoch >= self.start_epoch - 1:  # epoch从0开始，第85轮时epoch=84
                if self.delayed_mode:
                    # 第一次启用早停时，打印提示信息
                    print(f"\n⏰ 已达到第{self.start_epoch}轮，启用早停监控（耐心值: {self.patience}轮）")
                    self.delayed_mode = False
                # 调用父类的早停逻辑
                super().on_epoch_end(epoch, logs)
            # 前85轮完全跳过早停检查
    
    early_stop = DelayedEarlyStopping(
        start_epoch=50,  # 从第50轮开始启用早停
        monitor='val_loss',
        mode='min',
        patience=35,  # 35轮无改进才停止（增加patience）
        verbose=1,
        restore_best_weights=True,
        min_delta=0.00005  # 降低阈值
    )
    callbacks.append(early_stop)
    
    # TensorBoard：可视化训练过程
    tensorboard_log_dir = os.path.join(
        log_dir,
        f'run_{time.strftime("%Y%m%d_%H%M%S")}'
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    
    # 学习率衰减（v2.3优化：更长的patience）
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=10,  # 10轮无改进即衰减（增加patience）
        min_lr=5e-7,  # 最小学习率降低到5e-7
        verbose=1,
        cooldown=3,  # 冷却3轮
        min_delta=0.00005  # 降低阈值，更敏感
    )
    callbacks.append(reduce_lr)
    
    # Warmup学习率策略（前15轮渐进提升）
    class WarmupLearningRate(keras.callbacks.Callback):
        def __init__(self, warmup_epochs=15, target_lr=0.001, start_lr=0.0001):
            super().__init__()
            self.warmup_epochs = warmup_epochs
            self.target_lr = target_lr
            self.start_lr = start_lr
        
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                # 线性增长：从start_lr逐渐增加到target_lr
                lr = self.start_lr + (self.target_lr - self.start_lr) * ((epoch + 1) / self.warmup_epochs)
                # 兼容不同Keras版本的学习率设置方式
                try:
                    # 尝试使用assign方法（TensorFlow 2.x推荐）
                    self.model.optimizer.learning_rate.assign(lr)
                except AttributeError:
                    # 降级到backend.set_value（旧版本）
                    import tensorflow.keras.backend as K
                    K.set_value(self.model.optimizer.lr, lr)
                print(f"  [Warmup] Epoch {epoch+1}/{self.warmup_epochs}, LR: {lr:.6f}")
    
    callbacks.append(WarmupLearningRate(warmup_epochs=10, target_lr=config.LEARNING_RATE, start_lr=0.0001))
    
    # 保存最佳完整匹配准确率模型
    class BestFullMatchCheckpoint(keras.callbacks.Callback):
        def __init__(self, val_data, model_dir):
            super().__init__()
            self.val_images, self.val_labels = val_data
            self.best_full_match_acc = 0
            self.model_dir = model_dir
        
        def on_epoch_end(self, epoch, logs=None):
            # 每5轮计算一次完整匹配准确率
            if (epoch + 1) % 5 != 0:
                return
            
            import numpy as np
            # 随机采样2000个验证样本
            sample_size = min(2000, len(self.val_images))
            indices = np.random.choice(len(self.val_images), sample_size, replace=False)
            sample_images = self.val_images[indices]
            sample_labels = self.val_labels[indices]
            
            predictions = self.model.predict(sample_images, verbose=0)
            pred_texts = [utils.vector_to_text(pred) for pred in predictions]
            true_texts = [utils.vector_to_text(label) for label in sample_labels]
            full_match_acc = utils.calculate_accuracy(true_texts, pred_texts)
            
            if full_match_acc > self.best_full_match_acc:
                self.best_full_match_acc = full_match_acc
                # 保存最佳完整匹配模型
                save_path = os.path.join(self.model_dir, 'best_full_match_model.keras')
                self.model.save(save_path)
                print(f"  ⭐ 完整匹配准确率提升至 {full_match_acc*100:.2f}%，模型已保存！")
    
    if val_data is not None:
        callbacks.append(BestFullMatchCheckpoint(val_data=val_data, model_dir=model_dir))
    
    # 训练进度打印（每轮计算完整匹配准确率）
    class TrainingProgress(keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.val_images, self.val_labels = val_data
            self.best_full_match_acc = 0
        
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get('val_loss', 0)
            val_binary_acc = logs.get('val_binary_accuracy', 0)
            
            # 获取当前学习率（兼容不同Keras版本）
            try:
                # 尝试直接获取numpy值（TensorFlow 2.x）
                current_lr = float(self.model.optimizer.learning_rate.numpy())
            except:
                # 降级到backend.get_value（旧版本）
                try:
                    import tensorflow.keras.backend as K
                    current_lr = float(K.get_value(self.model.optimizer.lr))
                except:
                    current_lr = 0.001  # 默认值
            
            # 计算完整匹配准确率（每轮都计算，了解真实进度）
            import numpy as np
            # 随机采样1000个验证样本计算（加快速度）
            sample_size = min(1000, len(self.val_images))
            indices = np.random.choice(len(self.val_images), sample_size, replace=False)
            sample_images = self.val_images[indices]
            sample_labels = self.val_labels[indices]
            
            predictions = self.model.predict(sample_images, verbose=0)
            pred_texts = [utils.vector_to_text(pred) for pred in predictions]
            true_texts = [utils.vector_to_text(label) for label in sample_labels]
            full_match_acc = utils.calculate_accuracy(true_texts, pred_texts)
            
            # 打印训练进度
            print(f"\n[Epoch {epoch+1}] 训练损失: {logs.get('loss', 0):.4f} | "
                  f"验证损失: {val_loss:.4f} | "
                  f"二进制准确率: {val_binary_acc:.4f} | "
                  f"完整匹配: {full_match_acc*100:.2f}% | "
                  f"学习率: {current_lr:.6f}")
            
            # 跟踪最佳完整匹配准确率
            if full_match_acc > self.best_full_match_acc:
                self.best_full_match_acc = full_match_acc
                print(f"    ⬆ 完整匹配准确率提升！当前: {full_match_acc*100:.2f}% (历史最佳: {self.best_full_match_acc*100:.2f}%)")
    
    # 添加训练进度回调（需要验证数据计算完整匹配准确率）
    if val_data is not None:
        callbacks.append(TrainingProgress(val_data=val_data))
    
    return callbacks


def train_model(
    model,
    train_data,
    val_data,
    epochs=None,
    batch_size=None,
    callbacks=None
):
    """
    训练模型
    
    参数:
        model: Keras模型
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
    print("训练策略（2026-01-30 v2.3 - 突破73%瓶颈）:")
    print("  - Warmup阶段: 前10轮学习率从0.0001→0.0012逐步提升")
    print("  - 主训练阶段: 前50轮充分训练，不触发早停")
    print("  - 早停监控: 第50轮后启用，35轮无改进自动停止")
    print("  - 学习率衰减: 10轮无改进降低50%（更长的patience）")
    print("  - 批次大小: 128")
    print("  - 正则化: BatchNorm + Dropout 0.2/0.4（降低Dropout提高召回率）")
    print("  - 损失函数: BinaryCrossentropy")
    print("  - 每轮计算: 完整匹配准确率（采样1000个验证样本）")
    print("=" * 80)
    print()
    
    # 训练模型
    history = model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=2
    )
    
    return history


def evaluate_model(model, val_data):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        val_data: 验证数据 (X, y)
    
    返回:
        评估指标字典
    """
    val_images, val_labels = val_data
    
    print("\n" + "=" * 80)
    print(" " * 30 + "模型评估")
    print("=" * 80)
    
    # Keras评估
    results = model.evaluate(val_images, val_labels, verbose=0)
    
    print(f"验证集损失: {results[0]:.4f}")
    print(f"二进制准确率: {results[1]:.4f}")
    print(f"精确率: {results[2]:.4f}")
    print(f"召回率: {results[3]:.4f}")
    print()
    
    # 完整匹配准确率评估
    print("计算完整验证码匹配准确率...")
    predictions = model.predict(val_images, verbose=0)
    
    # 解码预测和真实标签
    pred_texts = [utils.vector_to_text(pred) for pred in predictions]
    true_texts = [utils.vector_to_text(label) for label in val_labels]
    
    # 计算准确率
    accuracy = utils.calculate_accuracy(true_texts, pred_texts)
    
    print(f"完整匹配准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 显示示例预测
    print("示例预测（前10个）:")
    print("-" * 80)
    print(f"{'真实值':<15} {'预测值':<15} {'匹配':<10}")
    print("-" * 80)
    for i in range(min(10, len(true_texts))):
        match = "✓" if true_texts[i] == pred_texts[i] else "✗"
        print(f"{true_texts[i]:<15} {pred_texts[i]:<15} {match:<10}")
    print("=" * 80)
    
    return {
        'loss': results[0],
        'binary_accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'full_match_accuracy': accuracy
    }


def save_model(model, save_path=None):
    """
    保存模型
    
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


# 主训练流程
def main():
    """主训练流程"""
    print("=" * 80)
    print(" " * 25 + "验证码识别模型训练")
    print("=" * 80)
    print()
    
    # 1. 加载数据
    print("步骤 1/5: 加载数据")
    print("-" * 80)
    loader = CaptchaDataLoader()
    loader.load_data()
    loader.print_statistics()
    print()
    
    # 2. 准备数据集
    print("步骤 2/5: 准备数据集")
    print("-" * 80)
    train_images, train_labels, val_images, val_labels = loader.prepare_dataset()
    print()
    
    # 3. 创建模型
    print("步骤 3/5: 创建模型")
    print("-" * 80)
    model = create_model()
    # 使用标准BCE Loss（GPU服务器验证：BCE 75% > Focal Loss 52%）
    model = compile_model(model, use_focal_loss=False)
    print_model_summary(model)
    print()
    
    # 4. 训练模型
    print("步骤 4/5: 训练模型")
    print("-" * 80)
    callbacks = create_callbacks(val_data=(val_images, val_labels))
    history = train_model(
        model,
        train_data=(train_images, train_labels),
        val_data=(val_images, val_labels),
        callbacks=callbacks,
        epochs=200  # 200轮上限 + 15轮早停
    )
    print()
    
    # 5. 评估模型
    print("步骤 5/5: 评估模型")
    print("-" * 80)
    metrics = evaluate_model(model, val_data=(val_images, val_labels))
    print()
    
    # 保存最终模型
    save_model(model)
    
    print("\n" + "=" * 80)
    print(" " * 30 + "训练完成")
    print("=" * 80)
    print(f"\n最终验证集完整匹配准确率: {metrics['full_match_accuracy']*100:.2f}%")
    print()
    
    return model, history, metrics


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
