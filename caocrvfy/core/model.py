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


def create_cnn_model():
    """
    创建验证码识别9层卷积神经网络模型
    
    架构:
        - 卷积层1: Conv2D(32, 3×3) + BN + ReLU + MaxPool
        - 卷积层2: Conv2D(64, 3×3) + BN + ReLU
        - 卷积层3: Conv2D(64, 3×3) + BN + ReLU + MaxPool
        - 卷积层4: Conv2D(128, 3×3) + BN + ReLU
        - 卷积层5: Conv2D(128, 3×3) + BN + ReLU + MaxPool
        - 卷积层6: Conv2D(256, 3×3) + BN + ReLU
        - 卷积层7: Conv2D(256, 3×3) + BN + ReLU + MaxPool
        - 卷积层8: Conv2D(512, 3×3) + BN + ReLU
        - 卷积层9: Conv2D(512, 3×3) + BN + ReLU + MaxPool
        - 双向LSTM层 (256 units) - 序列建模
        - 全连接层 (2048 units) + Dropout(0.5)
        - 输出层 (OUTPUT_SIZE units)
    
    总计: 9层卷积 + LSTM + 全连接，卷积层不使用dropout
    
    返回:
        Keras模型
    """
    # 输入层
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # ==================== 卷积层1-2：32、64 ====================
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    
    # ==================== 卷积层3-4：64、128 ====================
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    
    # ==================== 卷积层5-6：128、256 ====================
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv6')(x)
    x = layers.BatchNormalization(name='bn6')(x)
    x = layers.Activation('relu', name='relu6')(x)
    
    # ==================== 卷积层7-8：256、512 ====================
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv7')(x)
    x = layers.BatchNormalization(name='bn7')(x)
    x = layers.Activation('relu', name='relu7')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same', name='conv8')(x)
    x = layers.BatchNormalization(name='bn8')(x)
    x = layers.Activation('relu', name='relu8')(x)
    
    # ==================== 卷积层9：512 ====================
    x = layers.Conv2D(512, (3, 3), padding='same', name='conv9')(x)
    x = layers.BatchNormalization(name='bn9')(x)
    x = layers.Activation('relu', name='relu9')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool5')(x)
    
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
    model = models.Model(inputs=inputs, outputs=outputs, name='captcha_cnn9')
    
    return model


def compile_model(model, learning_rate=None, use_lr_schedule=False, 
                  use_focal_loss=False, pos_weight=None, focal_gamma=None, **kwargs):
    """
    编译模型（使用自适应学习率）
    
    参数:
        model: Keras模型
        learning_rate: 初始学习率
        use_lr_schedule: 是否使用指数衰减学习率调度（默认False，不推荐使用）
        use_focal_loss: 是否使用Focal Loss（仅用于兼容性，core模型不支持）
        pos_weight: 正样本权重（仅用于兼容性，core模型不支持）
        focal_gamma: Focal Loss的gamma参数（仅用于兼容性，core模型不支持）
        **kwargs: 其他参数（用于兼容性）
    
    返回:
        编译后的模型
    
    注：
        - Adam优化器本身就是自适应学习率算法
        - 结合callbacks中的AdaptiveLearningRate进行动态调整
        - 不推荐使用use_lr_schedule，会与AdaptiveLearningRate冲突
        - use_focal_loss等参数仅用于与model_enhanced的接口兼容
    """
    # 如果调用了Focal Loss相关参数，给出警告
    if use_focal_loss or pos_weight or focal_gamma:
        print("⚠️  注意：当前使用core.model，不支持Focal Loss")
        print("    如需使用Focal Loss，请设置 USE_ENHANCED_MODEL = True")
        print("    将使用标准BinaryCrossentropy损失函数\n")
    
    # 如果使用了learning rate schedule，给出警告
    if use_lr_schedule:
        print("⚠️  不推荐使用use_lr_schedule，它会与AdaptiveLearningRate冲突")
        print("    建议使用AdaptiveLearningRate进行自适应调整\n")
    
    initial_lr = learning_rate or config.LEARNING_RATE
    
    # 使用固定学习率，通过AdaptiveLearningRate进行动态调整
    lr = initial_lr
    
    # 优化器：Adam自适应学习率优化器
    # Adam = Adaptive Moment Estimation，自动调整每个参数的学习率
    optimizer = keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,      # 一阶矩估计的指数衰减率
        beta_2=0.999,    # 二阶矩估计的指数衰减率
        epsilon=1e-7     # 数值稳定性常数
    )
    
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
    
    print("测试9层卷积神经网络模型创建...")
    print()
    
    # 创建模型
    model = create_cnn_model()
    
    # 编译模型（使用自适应学习率）
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
    print("✓ 9层卷积神经网络模型创建测试完成")
    print("✓ 使用自适应学习率（Adam优化器）")
    print("=" * 80)
