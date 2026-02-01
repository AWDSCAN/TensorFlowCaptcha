"""
数据增强模块
参考trains.py的数据预处理思路，为TF2实现数据增强策略
目的：减少过拟合（当前训练损失0.0063 vs 验证损失0.0141，比例2.2x）
"""

import tensorflow as tf
import numpy as np
from .config import IMAGE_HEIGHT, IMAGE_WIDTH


def random_brightness(image, max_delta=0.2):
    """随机亮度调整"""
    return tf.image.random_brightness(image, max_delta=max_delta)


def random_contrast(image, lower=0.8, upper=1.2):
    """随机对比度调整"""
    return tf.image.random_contrast(image, lower=lower, upper=upper)


def random_noise(image, stddev=0.02):
    """添加随机噪声（模拟真实验证码的干扰）"""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 1.0)


def augment_image(image, training=True):
    """
    数据增强pipeline（优化版）
    
    优化说明：
    1. 减少亮度变化幅度（验证码已有复杂背景）
    2. 收窄对比度范围（避免过度失真）
    3. 移除随机噪声（验证码本身已有1000-1500个噪点）
    
    参考trains.py的思路：
    1. 亮度变化（模拟不同光照条件）
    2. 对比度变化（增强泛化）
    
    Args:
        image: 输入图像 [H, W, C]
        training: 是否为训练模式
    
    Returns:
        增强后的图像
    """
    if not training:
        return image
    
    # 随机应用亮度调整（50%概率，±10%，从±15%减少）
    if tf.random.uniform([]) > 0.5:
        image = random_brightness(image, max_delta=0.10)
    
    # 随机应用对比度调整（50%概率，90%-110%，从85%-115%收窄）
    if tf.random.uniform([]) > 0.5:
        image = random_contrast(image, lower=0.90, upper=1.10)
    
    # 【移除】随机噪声（验证码本身已有1000-1500个噪点，不需要额外噪声）
    # 原因：过多噪声会干扰字符特征学习
    # if tf.random.uniform([]) > 0.7:
    #     image = random_noise(image, stddev=0.015)
    
    # 确保像素值在[0, 1]范围内
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def create_augmented_dataset(images, labels, batch_size=128, training=True):
    """
    创建带数据增强的Dataset
    
    参考trains.py的批量处理策略：
    - 使用tf.data.Dataset进行高效数据加载
    - 应用数据增强pipeline
    - 批量处理和预取优化
    
    Args:
        images: 图像数据 [N, H, W, C]
        labels: 标签数据 [N, ...]
        batch_size: 批量大小
        training: 是否为训练模式
    
    Returns:
        tf.data.Dataset对象
    """
    # 创建基础Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if training:
        # 训练集：打乱顺序
        dataset = dataset.shuffle(buffer_size=min(10000, len(images)))
        
        # 应用数据增强
        dataset = dataset.map(
            lambda img, lbl: (augment_image(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # 验证集：不应用增强
        dataset = dataset.map(
            lambda img, lbl: (augment_image(img, training=False), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # 批量处理
    dataset = dataset.batch(batch_size)
    
    # 预取优化（参考trains.py的性能优化策略）
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


if __name__ == '__main__':
    # 测试数据增强效果
    print("=" * 80)
    print("数据增强模块测试")
    print("=" * 80)
    
    # 创建测试图像
    test_image = tf.random.uniform([IMAGE_HEIGHT, IMAGE_WIDTH, 1], 0, 1)
    
    print(f"\n原始图像形状: {test_image.shape}")
    print(f"原始图像范围: [{tf.reduce_min(test_image):.4f}, {tf.reduce_max(test_image):.4f}]")
    
    # 测试10次增强
    print("\n测试10次随机增强:")
    for i in range(10):
        augmented = augment_image(test_image, training=True)
        print(f"  第{i+1}次: 范围 [{tf.reduce_min(augmented):.4f}, {tf.reduce_max(augmented):.4f}]")
    
    # 测试Dataset创建
    print("\n测试Dataset创建:")
    dummy_images = np.random.rand(100, IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, (100, 504)).astype(np.float32)
    
    train_ds = create_augmented_dataset(dummy_images, dummy_labels, batch_size=16, training=True)
    val_ds = create_augmented_dataset(dummy_images, dummy_labels, batch_size=16, training=False)
    
    print(f"  训练集批次数: {len(list(train_ds))}")
    print(f"  验证集批次数: {len(list(val_ds))}")
    
    # 检查批次形状
    for images_batch, labels_batch in train_ds.take(1):
        print(f"  批次图像形状: {images_batch.shape}")
        print(f"  批次标签形状: {labels_batch.shape}")
    
    print("\n✓ 数据增强模块测试通过")
    print("=" * 80)
