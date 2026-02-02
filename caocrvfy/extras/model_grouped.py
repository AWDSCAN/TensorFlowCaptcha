#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版CNN模型 v2 - 分组输出架构
关键改进: 为每个字符位置单独预测，提升序列建模能力

核心思想:
    原架构: 1个输出层 (504维) → reshape成 (8, 63)
    新架构: 8个独立输出层，每层63维
    
优势:
    1. 每个字符位置有独立的特征提取路径
    2. 避免位置之间的相互干扰
    3. 更容易学习位置特定的模式
    4. 显著提升完整匹配准确率
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from core import config


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss - 专注于难分类样本
    
    公式: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    参数:
        gamma: 聚焦参数，越大越关注难样本 (默认2.0)
        alpha: 类别平衡参数 (默认0.25)
    """
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # 防止log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # 计算交叉熵
        ce = -y_true * tf.math.log(y_pred)
        
        # 计算focal权重
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # 应用focal权重
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(focal_loss)


def create_grouped_output_model():
    """
    创建分组输出模型
    
    架构:
        - 5层卷积 + BatchNorm (特征提取骨干)
        - 共享特征层 (2048 + 1024 FC)
        - 8个独立输出头 (每个字符位置一个)
    
    返回:
        Keras Model with grouped outputs
    """
    inputs = layers.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name='input_image'
    )
    
    # ========== 特征提取骨干网络 (共享) ==========
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    # 展平
    x = layers.Flatten(name='flatten')(x)
    
    # ========== 共享特征层 ==========
    shared = layers.Dense(2048, activation='relu', name='shared_fc1')(x)
    shared = layers.BatchNormalization(name='bn_shared1')(shared)
    shared = layers.Dropout(0.5, name='dropout_shared1')(shared)
    
    shared = layers.Dense(1024, activation='relu', name='shared_fc2')(shared)
    shared = layers.BatchNormalization(name='bn_shared2')(shared)
    shared = layers.Dropout(0.4, name='dropout_shared2')(shared)
    
    # ========== 分组输出头 (8个独立分类器) ==========
    char_outputs = []
    for i in range(config.MAX_CAPTCHA):
        # 每个字符位置有自己的特征层
        char_features = layers.Dense(256, activation='relu', name=f'char_{i}_fc')(shared)
        char_features = layers.Dropout(0.3, name=f'char_{i}_dropout')(char_features)
        
        # 输出层 (softmax分类)
        char_out = layers.Dense(
            config.CHAR_SET_LEN,
            activation='softmax',
            name=f'char_{i}_output'
        )(char_features)
        
        char_outputs.append(char_out)
    
    # 合并所有输出
    # outputs = layers.Concatenate(name='concat_outputs')(char_outputs)
    
    model = models.Model(
        inputs=inputs,
        outputs=char_outputs,  # 返回列表
        name='captcha_grouped_cnn'
    )
    
    return model


def compile_grouped_model(model, learning_rate=None, use_focal_loss=True, focal_gamma=2.0):
    """
    编译分组输出模型
    
    参数:
        model: Keras模型
        learning_rate: 学习率（可以是float或LearningRateSchedule）
        use_focal_loss: 是否使用Focal Loss
        focal_gamma: Focal Loss的gamma参数
    
    返回:
        编译后的模型
    """
    lr = learning_rate or config.LEARNING_RATE
    
    # 选择损失函数
    if use_focal_loss:
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=0.25)
        loss_name = f"Focal Loss (γ={focal_gamma})"
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = "Categorical Cross-Entropy"
    
    # 为每个输出头指定相同的损失函数
    losses = [loss_fn] * config.MAX_CAPTCHA
    
    # 可选: 为不同位置设置不同权重
    # 前3个字符（通常是数字）更重要
    loss_weights = [1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 0.9, 0.9]
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
        loss=losses,
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    
    print(f"✓ 模型已编译")
    print(f"  优化器: Adam (AMSGrad)")
    print(f"  损失函数: {loss_name}")
    print(f"  学习率: {lr if isinstance(lr, float) else '动态调度'}")
    print(f"  输出架构: 8个独立分类器 (每个{config.CHAR_SET_LEN}类)")
    print(f"  位置权重: {loss_weights}")
    
    return model


def print_model_summary(model):
    """打印模型摘要信息"""
    print("\n" + "=" * 80)
    print(" " * 25 + "模型架构 (分组输出)")
    print("=" * 80)
    model.summary()
    print("=" * 80)
    
    # 统计参数
    total_params = model.count_params()
    print(f"\n总参数量: {total_params:,}")
    print(f"模型类型: 分组输出CNN (8个独立分类器)")
    print(f"输出维度: 8 × {config.CHAR_SET_LEN} = {8 * config.CHAR_SET_LEN}")
    print("=" * 80)


if __name__ == '__main__':
    # 测试模型创建
    print("创建分组输出模型...")
    model = create_grouped_output_model()
    model = compile_grouped_model(model, use_focal_loss=True, focal_gamma=2.0)
    print_model_summary(model)
    
    # 测试输入输出
    import numpy as np
    test_input = np.random.rand(1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS)
    test_output = model.predict(test_input, verbose=0)
    print(f"\n测试预测:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出数量: {len(test_output)}")
    print(f"  每个输出形状: {test_output[0].shape}")
