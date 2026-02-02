#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN模型定义模块
功能：定义验证码识别的卷积神经网络模型
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 支持直接运行和模块导入两种方式
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
else:
    from . import config


def residual_block(x, filters, stride=1, conv_shortcut=False, name='residual'):
    """
    ResNet残差块
    
    参数:
        x: 输入张量
        filters: 滤波器数量
        stride: 步长
        conv_shortcut: 是否使用卷积捷径
        name: 层名称前缀（默认'residual'）
    
    返回:
        残差块输出
    """
    bn_axis = 3  # channels_last格式
    
    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding='same',
            name=f'{name}_0_conv' if name else None
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis,
            name=f'{name}_0_bn' if name else None
        )(shortcut)
    else:
        if stride > 1:
            shortcut = layers.MaxPooling2D(1, strides=stride)(x)
        else:
            shortcut = x
    
    # 第一个卷积
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding='same',
        name=f'{name}_1_conv' if name else None
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f'{name}_1_bn' if name else None)(x)
    x = layers.Activation('relu', name=f'{name}_1_relu' if name else None)(x)
    
    # 第二个卷积
    x = layers.Conv2D(
        filters,
        3,
        padding='same',
        name=f'{name}_2_conv' if name else None
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f'{name}_2_bn' if name else None)(x)
    
    # 残差连接
    x = layers.Add(name=f'{name}_add' if name else None)([shortcut, x])
    x = layers.Activation('relu', name=f'{name}_out' if name else None)(x)
    
    return x


def residual_stack(x, filters, blocks, stride=1, name='stack'):
    """
    ResNet残差堆叠
    
    参数:
        x: 输入张量
        filters: 滤波器数量
        blocks: 残差块数量
        stride: 第一个块的步长
        name: 名称前缀（默认'stack'）
    
    返回:
        堆叠输出
    """
    # 第一个残差块（可能有stride）
    x = residual_block(x, filters, stride=stride, conv_shortcut=True, name=f'{name}_block1' if name else 'block1')
    
    # 后续残差块
    for i in range(2, blocks + 1):
        x = residual_block(x, filters, name=f'{name}_block{i}' if name else f'block{i}')
    
    return x


def create_cnn_model():
    """
    创建验证码识别ResNet-34模型
    
    架构:
        - 初始卷积层: Conv2D(64, 7×7) + BN + ReLU + MaxPool
        - conv2_x: 3个残差块 × 64 filters
        - conv3_x: 4个残差块 × 128 filters (stride=2)
        - conv4_x: 6个残差块 × 256 filters (stride=2)
        - conv5_x: 3个残差块 × 512 filters (stride=2)
        - 双向LSTM层 (256 units) - 序列建模
        - 全连接层 (2048 units) + Dropout(0.5)
        - 输出层 (OUTPUT_SIZE units)
    
    总计: 1 + (3+4+6+3)×2 + 1 + 1 = 34层
    
    返回:
        Keras模型
    """
    # 输入层
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # ==================== Stage 1: 初始卷积 ====================
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding='same',
        name='conv1_conv'
    )(inputs)
    x = layers.BatchNormalization(axis=3, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)
    
    # ==================== Stage 2: conv2_x (3个残差块) ====================
    x = residual_stack(x, 64, 3, stride=1, name='conv2')
    
    # ==================== Stage 3: conv3_x (4个残差块) ====================
    x = residual_stack(x, 128, 4, stride=2, name='conv3')
    
    # ==================== Stage 4: conv4_x (6个残差块) ====================
    x = residual_stack(x, 256, 6, stride=2, name='conv4')
    
    # ==================== Stage 5: conv5_x (3个残差块) ====================
    x = residual_stack(x, 512, 3, stride=2, name='conv5')
    
    # ==================== 序列建模层 ====================
    # 获取特征图的shape进行reshape
    shape = x.shape
    # 将特征图reshape为序列形式: (batch, width, height*channels)
    x = layers.Reshape((shape[2], shape[1] * shape[3]), name='reshape_for_lstm')(x)
    
    # 双向LSTM层（256 units，学习字符序列依赖关系）
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
        name='bidirectional_lstm'
    )(x)
    
    # 展平LSTM输出
    x = layers.Flatten(name='flatten')(x)
    
    # ==================== 全连接层 ====================
    x = layers.Dense(
        2048,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001),
        name='fc'
    )(x)
    x = layers.Dropout(0.5, name='dropout_fc')(x)  # 只在全连接层使用dropout
    
    # 输出层
    outputs = layers.Dense(
        config.OUTPUT_SIZE,
        activation='sigmoid',
        name='output'
    )(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs, name='captcha_resnet34')
    
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
    import sys
    import os
    
    # 添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import config
    
    print("测试ResNet-34模型创建...")
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
    
    print()
    print("=" * 80)
    print("✓ ResNet-34模型创建测试完成")
    print("=" * 80)
