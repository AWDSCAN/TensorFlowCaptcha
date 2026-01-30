#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版CNN模型定义 - 用于强干扰验证码识别
架构改进：
  - 更深的网络（5层卷积）
  - BatchNormalization加速收敛
  - Residual连接增强梯度流
  - 更大的全连接层
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


def create_enhanced_cnn_model():
    """
    创建增强版验证码识别CNN模型（用于强干扰）
    
    架构配置（v2.2 - 简化优化）：
        - 5层卷积（32→64→128→128→256）
        - BatchNormalization 加速收敛
        - Dropout 0.25（适度正则化）
        - 全连接层2048 + 1024
        - 移除L2正则化，简化模型
    
    返回:
        Keras模型
    """
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # 第一层卷积块
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # 第二层卷积块
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # 第三层卷积块
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.2, name='dropout1')(x)
    
    # 第四层卷积块
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    # 第五层卷积块
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)
    x = layers.Dropout(0.2, name='dropout3')(x)
    
    # 展平
    x = layers.Flatten(name='flatten')(x)
    
    # 第一个全连接层
    x = layers.Dense(2048, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Dropout(0.4, name='dropout_fc1')(x)
    
    # 第二个全连接层
    x = layers.Dense(1024, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='bn_fc2')(x)
    x = layers.Dropout(0.4, name='dropout_fc2')(x)
    
    # 输出层
    outputs = layers.Dense(config.OUTPUT_SIZE, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='captcha_enhanced_cnn')
    
    return model


def create_resnet_style_model():
    """
    创建ResNet风格的模型（使用残差连接）
    适合非常强的干扰
    
    返回:
        Keras模型
    """
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # 初始卷积
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv_init')(inputs)
    x = layers.BatchNormalization(name='bn_init')(x)
    x = layers.Activation('relu', name='relu_init')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool_init')(x)
    
    # 残差块1
    def residual_block(x, filters, name):
        shortcut = x
        
        # 主路径
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # 调整shortcut维度
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=f'{name}_shortcut')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_bn_shortcut')(shortcut)
        
        # 相加并激活
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    # 多个残差块
    x = residual_block(x, 64, 'res1')
    x = residual_block(x, 64, 'res2')
    x = layers.MaxPooling2D((2, 2), name='pool_res1')(x)
    
    x = residual_block(x, 128, 'res3')
    x = residual_block(x, 128, 'res4')
    x = layers.MaxPooling2D((2, 2), name='pool_res2')(x)
    
    x = residual_block(x, 256, 'res5')
    x = residual_block(x, 256, 'res6')
    
    # 全局平均池化
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # 全连接层
    x = layers.Dense(2048, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    
    x = layers.Dense(1024, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.4, name='dropout2')(x)
    
    # 输出层
    outputs = layers.Dense(config.OUTPUT_SIZE, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='captcha_resnet')
    
    return model


def compile_model(model, learning_rate=None, use_focal_loss=False, focal_gamma=1.5):
    """
    编译模型
    
    参数:
        model: Keras模型
        learning_rate: 学习率
        use_focal_loss: 是否使用Focal Loss（默认False，标准BCE效果更好：75% vs 52%）
        focal_gamma: Focal Loss的gamma参数（1.0-2.0，默认1.5）
    
    返回:
        编译后的模型
    """
    lr = learning_rate or config.LEARNING_RATE
    
    # 使用优化的Adam配置：beta_1=0.9, beta_2=0.999（默认），添加更强的梯度裁剪
    optimizer = keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,      # 一阶矩估计的衰减率
        beta_2=0.999,    # 二阶矩估计的衰减率
        amsgrad=True,    # 使用AMSGrad变体，更稳定
        clipnorm=1.0     # 梯度裁剪，防止梯度爆炸
    )
    
    # 损失函数：标准BCE Loss（经GPU服务器实测：BCE=75% > Focal Loss=52%）
    if use_focal_loss:
        from focal_loss import BinaryFocalLoss
        loss = BinaryFocalLoss(gamma=focal_gamma, alpha=0.75)
        print(f"⚠️  使用Focal Loss (gamma={focal_gamma}, alpha=0.75) - 注意：可能降低准确率")
    else:
        loss = keras.losses.BinaryCrossentropy()
        print("✓ 使用标准BCE Loss（已验证最优：75% > Focal Loss 52%）")
    
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
    model_size = total_params * 4 / (1024 ** 2)
    print(f"估计模型大小: {model_size:.2f} MB")
    print("=" * 80)


# 测试模型创建
if __name__ == '__main__':
    print("测试增强版CNN模型创建...")
    print()
    
    # 测试增强版模型
    print("1. 增强版CNN模型:")
    print("-" * 80)
    model = create_enhanced_cnn_model()
    model = compile_model(model)
    print_model_summary(model)
    
    print("\n" + "=" * 80)
    
    # 测试ResNet风格模型
    print("\n2. ResNet风格模型:")
    print("-" * 80)
    model_resnet = create_resnet_style_model()
    model_resnet = compile_model(model_resnet)
    print_model_summary(model_resnet)
    
    # 测试预测
    print("\n测试模型预测...")
    import numpy as np
    
    test_input = np.random.rand(
        2,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    ).astype(np.float32)
    
    output = model.predict(test_input, verbose=0)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print()
    print("=" * 80)
    print("✓ 增强版模型创建测试完成")
