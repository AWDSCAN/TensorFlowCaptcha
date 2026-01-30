#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块
功能：训练验证码识别模型
"""

import os
import time
import tensorflow as tf
from tensorflow import keras
import config
from data_loader import CaptchaDataLoader
from model import create_cnn_model, compile_model, print_model_summary
import utils


def create_callbacks(model_dir=None, log_dir=None):
    """
    创建训练回调函数
    
    参数:
        model_dir: 模型保存目录
        log_dir: 日志保存目录
    
    返回:
        回调函数列表
    """
    model_dir = model_dir or config.MODEL_DIR
    log_dir = log_dir or config.LOG_DIR
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = []
    
    # 模型检查点：保存最优模型
    checkpoint_path = os.path.join(model_dir, 'best_model.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 早停：防止过拟合
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True
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
    
    # 学习率衰减
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # 训练进度打印
    class TrainingProgress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(f"\n[Epoch {epoch+1}] 训练准确率: {logs.get('binary_accuracy', 0):.4f} | "
                  f"验证准确率: {logs.get('val_binary_accuracy', 0):.4f} | "
                  f"训练损失: {logs.get('loss', 0):.4f} | "
                  f"验证损失: {logs.get('val_loss', 0):.4f}")
    
    callbacks.append(TrainingProgress())
    
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
    print(f"训练轮数: {epochs}")
    print(f"学习率: {config.LEARNING_RATE}")
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
    model = create_cnn_model()
    model = compile_model(model)
    print_model_summary(model)
    print()
    
    # 4. 训练模型
    print("步骤 4/5: 训练模型")
    print("-" * 80)
    callbacks = create_callbacks()
    history = train_model(
        model,
        train_data=(train_images, train_labels),
        val_data=(val_images, val_labels),
        callbacks=callbacks
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
