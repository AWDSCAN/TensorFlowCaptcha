#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN模型定义模块
功能：定义验证码识别的卷积神经网络模型
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from . import config


def create_cnn_model():
    """
    创建验证码识别CNN模型
    
    架构:
        - 3层卷积层 (32 → 64 → 64 filters, 3×3 kernel)
        - 每层卷积后接MaxPooling (2×2)
        - Dropout层 (0.25)
        - 全连接层 (1024 units)
        - 输出层 (OUTPUT_SIZE units)
    
    返回:
        Keras模型
    """
    # 输入层
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # 第一层卷积
    x = layers.Conv2D(
        filters=config.CONV_FILTERS[0],
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv1'
    )(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    # 第二层卷积
    x = layers.Conv2D(
        filters=config.CONV_FILTERS[1],
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv2'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    
    # 第三层卷积
    x = layers.Conv2D(
        filters=config.CONV_FILTERS[2],
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='conv3'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    
    # Dropout层
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout')(x)
    
    # 展平层
    x = layers.Flatten(name='flatten')(x)
    
    # 全连接层
    x = layers.Dense(
        config.FC_UNITS,
        activation='relu',
        name='fc'
    )(x)
    
    # 输出层
    outputs = layers.Dense(
        config.OUTPUT_SIZE,
        activation='sigmoid',
        name='output'
    )(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs, name='captcha_cnn')
    
    return model


def compile_model(model, learning_rate=None):
    """
    编译模型
    
    参数:
        model: Keras模型
        learning_rate: 学习率
    
    返回:
        编译后的模型
    """
    lr = learning_rate or config.LEARNING_RATE
    
    # 优化器
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    # 损失函数：sigmoid交叉熵
    loss = keras.losses.BinaryCrossentropy()
    
    # 评估指标
    metrics = [
        keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def print_model_summary(model):
    """
    打印模型摘要
    
    参数:
        model: Keras模型
    """
    print("=" * 80)
    print(" " * 30 + "模型结构")
    print("=" * 80)
    model.summary()
    print("=" * 80)
    
    # 计算参数量
    total_params = model.count_params()
    print(f"\n总参数量: {total_params:,}")
    
    # 计算模型大小（MB）
    model_size = total_params * 4 / (1024 ** 2)  # 假设每个参数4字节
    print(f"估计模型大小: {model_size:.2f} MB")
    print("=" * 80)


# 测试模型创建
if __name__ == '__main__':
    print("测试CNN模型创建...")
    print()
    
    # 创建模型
    model = create_cnn_model()
    
    # 编译模型
    model = compile_model(model)
    
    # 打印模型摘要
    print_model_summary(model)
    
    # 测试模型输入输出
    print("\n测试模型预测...")
    import numpy as np
    
    # 创建随机输入
    test_input = np.random.rand(
        1,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    ).astype(np.float32)
    
    # 预测
    output = model.predict(test_input, verbose=0)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试解码
    import utils
    predicted_text = utils.vector_to_text(output[0])
    print(f"预测验证码: {predicted_text}")
    
    print()
    print("=" * 80)
    print("✓ 模型创建测试完成")
