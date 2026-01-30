#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focal Loss实现 - 专门处理困难样本
用于突破准确率瓶颈
"""

import tensorflow as tf
from tensorflow import keras


class BinaryFocalLoss(keras.losses.Loss):
    """
    二分类Focal Loss
    
    论文: Focal Loss for Dense Object Detection
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数:
        alpha: 正样本权重，默认0.75 (负样本权重=1-alpha=0.25)
        gamma: 聚焦参数，默认1.5，越大越关注困难样本
        from_logits: 输入是否为logits
    """
    
    def __init__(self, alpha=0.75, gamma=1.5, from_logits=False, name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        计算Focal Loss
        
        参数:
            y_true: 真实标签 (batch_size, num_classes)
            y_pred: 预测概率 (batch_size, num_classes)
        
        返回:
            focal_loss: 标量损失值
        """
        # 如果输入是logits，先转换为概率
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        
        # 确保y_pred在有效范围内，避免log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # 计算交叉熵
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # 计算p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # 计算调制因子 (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # 计算alpha_t
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Focal Loss = alpha_t * (1 - p_t)^gamma * CE
        focal_loss = alpha_t * modulating_factor * cross_entropy
        
        # 返回批次平均损失
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class AdaptiveFocalLoss(keras.losses.Loss):
    """
    自适应Focal Loss
    在训练前期使用BCE，后期逐渐切换到Focal Loss
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, warmup_epochs=50, name='adaptive_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.bce = keras.losses.BinaryCrossentropy()
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def call(self, y_true, y_pred):
        """
        自适应混合BCE和Focal Loss
        
        前warmup_epochs轮: 100% BCE
        之后: 逐渐增加Focal Loss权重
        """
        # 计算当前epoch的Focal Loss权重
        epoch_float = tf.cast(self.current_epoch, tf.float32)
        warmup_float = tf.cast(self.warmup_epochs, tf.float32)
        
        # 如果在warmup期内，focal_weight = 0
        # warmup后，focal_weight逐渐增加到1.0
        focal_weight = tf.maximum(0.0, 
                                   tf.minimum(1.0, 
                                             (epoch_float - warmup_float) / warmup_float))
        
        bce_weight = 1.0 - focal_weight
        
        # 混合损失
        bce_loss = self.bce(y_true, y_pred)
        focal_loss = self.focal(y_true, y_pred)
        
        combined_loss = bce_weight * bce_loss + focal_weight * focal_loss
        
        return combined_loss
    
    def update_epoch(self, epoch):
        """更新当前epoch"""
        self.current_epoch.assign(epoch)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'warmup_epochs': self.warmup_epochs
        })
        return config


def get_focal_loss(gamma=2.0, alpha=0.25, adaptive=False, warmup_epochs=50):
    """
    获取Focal Loss实例
    
    参数:
        gamma: 聚焦参数，越大越关注困难样本（推荐1.5-3.0）
        alpha: 平衡因子（推荐0.25）
        adaptive: 是否使用自适应Focal Loss
        warmup_epochs: 自适应模式下的warmup轮数
    
    返回:
        Loss实例
    """
    if adaptive:
        return AdaptiveFocalLoss(alpha=alpha, gamma=gamma, warmup_epochs=warmup_epochs)
    else:
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)


# 测试代码
if __name__ == '__main__':
    import numpy as np
    
    print("测试Focal Loss...")
    
    # 创建测试数据
    y_true = tf.constant([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    
    # 简单样本（预测很准确）
    y_pred_easy = tf.constant([[0.95, 0.05, 0.90], [0.05, 0.95, 0.10]])
    
    # 困难样本（预测不确定）
    y_pred_hard = tf.constant([[0.55, 0.45, 0.60], [0.40, 0.60, 0.35]])
    
    # BCE Loss
    bce = keras.losses.BinaryCrossentropy()
    bce_easy = bce(y_true, y_pred_easy).numpy()
    bce_hard = bce(y_true, y_pred_hard).numpy()
    
    print(f"\nBCE Loss:")
    print(f"  简单样本: {bce_easy:.4f}")
    print(f"  困难样本: {bce_hard:.4f}")
    print(f"  困难/简单比: {bce_hard/bce_easy:.2f}x")
    
    # Focal Loss (gamma=2.0)
    focal = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    focal_easy = focal(y_true, y_pred_easy).numpy()
    focal_hard = focal(y_true, y_pred_hard).numpy()
    
    print(f"\nFocal Loss (gamma=2.0):")
    print(f"  简单样本: {focal_easy:.4f}")
    print(f"  困难样本: {focal_hard:.4f}")
    print(f"  困难/简单比: {focal_hard/focal_easy:.2f}x")
    
    # Focal Loss (gamma=3.0，更激进)
    focal3 = BinaryFocalLoss(gamma=3.0, alpha=0.25)
    focal3_easy = focal3(y_true, y_pred_easy).numpy()
    focal3_hard = focal3(y_true, y_pred_hard).numpy()
    
    print(f"\nFocal Loss (gamma=3.0):")
    print(f"  简单样本: {focal3_easy:.4f}")
    print(f"  困难样本: {focal3_hard:.4f}")
    print(f"  困难/简单比: {focal3_hard/focal3_easy:.2f}x")
    
    print("\n✓ Focal Loss能够更加关注困难样本！")
    print("  gamma越大，对困难样本的关注度越高")
