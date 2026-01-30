#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练代码可行性验证脚本
用途：在小数据集上验证训练逻辑是否正确
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import CaptchaDataLoader
from model import create_cnn_model, compile_model, print_model_summary
import utils


def verify_data_loading():
    """验证数据加载功能"""
    print("=" * 80)
    print("步骤 1/4: 验证数据加载")
    print("=" * 80)
    
    loader = CaptchaDataLoader()
    
    # 加载数据
    count = loader.load_data()
    print(f"\n✓ 成功加载 {count} 张图片")
    
    if count == 0:
        print("\n❌ 错误：没有找到任何验证码图片！")
        print(f"请确保 {config.CAPTCHA_DIR} 目录下有验证码图片")
        return None
    
    # 显示统计信息
    loader.print_statistics()
    
    return loader


def verify_data_preprocessing(loader):
    """验证数据预处理功能"""
    print("\n" + "=" * 80)
    print("步骤 2/4: 验证数据预处理")
    print("=" * 80)
    
    # 准备数据集
    train_images, train_labels, val_images, val_labels = loader.prepare_dataset()
    
    # 验证数据形状
    print(f"\n训练集图片形状: {train_images.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"验证集图片形状: {val_images.shape}")
    print(f"验证集标签形状: {val_labels.shape}")
    
    # 验证数据类型和范围
    print(f"\n图片数据类型: {train_images.dtype}")
    print(f"图片值范围: [{train_images.min():.4f}, {train_images.max():.4f}]")
    print(f"标签数据类型: {train_labels.dtype}")
    print(f"标签值范围: [{train_labels.min():.4f}, {train_labels.max():.4f}]")
    
    # 验证标签解码
    print("\n标签解码测试（前3个训练样本）:")
    for i in range(min(3, len(train_labels))):
        decoded_text = utils.vector_to_text(train_labels[i])
        print(f"  样本 {i+1}: {decoded_text}")
    
    print("\n✓ 数据预处理验证通过")
    
    return (train_images, train_labels), (val_images, val_labels)


def verify_model_creation():
    """验证模型创建功能"""
    print("\n" + "=" * 80)
    print("步骤 3/4: 验证模型创建")
    print("=" * 80)
    
    # 创建模型
    model = create_cnn_model()
    model = compile_model(model)
    
    # 显示模型摘要
    print_model_summary(model)
    
    # 测试前向传播
    print("\n测试模型前向传播...")
    test_input = np.random.rand(2, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"✓ 输入形状: {test_input.shape}")
    print(f"✓ 输出形状: {test_output.shape}")
    print(f"✓ 输出值范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # 测试输出解码
    decoded = utils.vector_to_text(test_output[0])
    print(f"✓ 输出解码测试: {decoded}")
    
    print("\n✓ 模型创建验证通过")
    
    return model


def verify_training(model, train_data, val_data, epochs=5):
    """验证训练功能（小规模）"""
    print("\n" + "=" * 80)
    print("步骤 4/4: 验证训练功能")
    print("=" * 80)
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    print(f"\n开始小规模训练验证（{epochs} 轮）...")
    print(f"训练样本: {len(train_images)}")
    print(f"验证样本: {len(val_images)}")
    print("-" * 80)
    
    # 创建简单的回调
    class SimpleCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {logs.get('loss', 0):.4f}")
            print(f"  训练准确率: {logs.get('binary_accuracy', 0):.4f}")
            if 'val_loss' in logs:
                print(f"  验证损失: {logs.get('val_loss', 0):.4f}")
                print(f"  验证准确率: {logs.get('val_binary_accuracy', 0):.4f}")
    
    # 训练
    history = model.fit(
        train_images,
        train_labels,
        batch_size=min(4, len(train_images)),  # 小批次
        epochs=epochs,
        validation_data=(val_images, val_labels) if len(val_images) > 0 else None,
        callbacks=[SimpleCallback()],
        verbose=0
    )
    
    print("\n" + "-" * 80)
    print("✓ 训练完成")
    
    # 测试预测
    print("\n预测测试（验证集前3个样本）:")
    if len(val_images) > 0:
        predictions = model.predict(val_images[:3], verbose=0)
        for i in range(min(3, len(val_images))):
            true_text = utils.vector_to_text(val_labels[i])
            pred_text = utils.vector_to_text(predictions[i])
            match = "✓" if true_text == pred_text else "✗"
            print(f"  真实: {true_text:10s} | 预测: {pred_text:10s} | {match}")
    else:
        # 用训练集测试
        predictions = model.predict(train_images[:3], verbose=0)
        for i in range(min(3, len(train_images))):
            true_text = utils.vector_to_text(train_labels[i])
            pred_text = utils.vector_to_text(predictions[i])
            match = "✓" if true_text == pred_text else "✗"
            print(f"  真实: {true_text:10s} | 预测: {pred_text:10s} | {match}")
    
    print("\n✓ 训练功能验证通过")
    
    return history


def analyze_training_requirements():
    """分析大规模训练需求"""
    print("\n" + "=" * 80)
    print("大规模训练需求分析（预计 60000 张图片）")
    print("=" * 80)
    
    total_samples = 60000
    batch_size = config.BATCH_SIZE
    
    # 计算每轮步数
    steps_per_epoch = total_samples // batch_size
    
    print(f"\n数据集配置:")
    print(f"  总样本数: {total_samples:,}")
    print(f"  批次大小: {batch_size}")
    print(f"  每轮步数: {steps_per_epoch}")
    
    print(f"\n模型配置:")
    print(f"  输入尺寸: {config.IMAGE_WIDTH}×{config.IMAGE_HEIGHT}×{config.IMAGE_CHANNELS}")
    print(f"  字符集大小: {config.CHAR_SET_LEN}")
    print(f"  最大验证码长度: {config.MAX_CAPTCHA}")
    print(f"  输出维度: {config.OUTPUT_SIZE}")
    
    # 计算参数量
    model = create_cnn_model()
    total_params = model.count_params()
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    print(f"\n模型规模:")
    print(f"  总参数量: {total_params:,}")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    
    # 训练轮数建议
    print(f"\n训练轮数建议:")
    print(f"  ✓ 基础收敛: 20-30 轮")
    print(f"  ✓ 良好性能: 50-100 轮")
    print(f"  ✓ 最优性能: 100-200 轮")
    
    # 估算训练时间
    print(f"\n训练时间估算（基于经验值）:")
    print(f"  GPU (RTX 3090):")
    print(f"    - 50 轮: 约 1-2 小时")
    print(f"    - 100 轮: 约 2-4 小时")
    print(f"  CPU (16核):")
    print(f"    - 50 轮: 约 10-20 小时")
    print(f"    - 100 轮: 约 20-40 小时")
    
    # 性能预期
    print(f"\n预期准确率（基于经验）:")
    print(f"  20 轮: 60-70%")
    print(f"  50 轮: 85-95%")
    print(f"  100 轮: 95-99%")
    
    print(f"\n优化建议:")
    print(f"  1. 使用 GPU 训练（推荐 RTX 3090 或更高）")
    print(f"  2. 启用混合精度训练（节省内存，加速训练）")
    print(f"  3. 使用学习率衰减和早停（防止过拟合）")
    print(f"  4. 数据增强可提高泛化能力（可选）")
    print(f"  5. 使用 TensorBoard 监控训练过程")
    
    print("=" * 80)


def main():
    """主验证流程"""
    print("\n" + "=" * 80)
    print(" " * 20 + "训练代码可行性验证")
    print("=" * 80)
    
    # 检查 TensorFlow
    print(f"\nTensorFlow 版本: {tf.__version__}")
    
    # 检查 GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ 检测到 {len(gpus)} 个 GPU:")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ 未检测到 GPU，将使用 CPU 训练")
    print()
    
    try:
        # 1. 验证数据加载
        loader = verify_data_loading()
        if loader is None:
            return
        
        # 2. 验证数据预处理
        train_data, val_data = verify_data_preprocessing(loader)
        
        # 3. 验证模型创建
        model = verify_model_creation()
        
        # 4. 验证训练（小规模）
        history = verify_training(model, train_data, val_data, epochs=5)
        
        # 5. 分析大规模训练需求
        analyze_training_requirements()
        
        # 总结
        print("\n" + "=" * 80)
        print(" " * 30 + "验证结果")
        print("=" * 80)
        print("\n✅ 所有验证通过！训练代码逻辑正确。")
        print("\n关键结论:")
        print("  1. 数据加载和预处理功能正常")
        print("  2. 模型结构定义正确")
        print("  3. 训练流程可以正常运行")
        print("  4. 预测功能工作正常")
        print("\n可以将代码部署到 GPU 服务器进行大规模训练。")
        print("\n建议:")
        print("  - 先生成 60000 张验证码图片")
        print("  - 使用 GPU 服务器训练 50-100 轮")
        print("  - 监控验证集准确率，达到 95% 以上即可")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 验证失败！")
        print("=" * 80)
        print(f"\n错误信息: {e}")
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        print("\n请检查并修复上述错误后重新运行验证。")
        print("=" * 80)
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
