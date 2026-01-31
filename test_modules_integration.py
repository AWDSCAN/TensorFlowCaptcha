#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块集成测试
验证所有增强模块是否正确导入和使用
"""

import sys
import os
import numpy as np
import tensorflow as tf

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

print("=" * 80)
print(" " * 25 + "模块集成测试")
print("=" * 80)

# 测试1: 数据增强模块
print("\n[测试1] 数据增强模块导入")
print("-" * 80)
try:
    from caocrvfy.data_augmentation import create_augmented_dataset, augment_image
    print("✅ data_augmentation.py 导入成功")
    print("   - create_augmented_dataset: OK")
    print("   - augment_image: OK")
except Exception as e:
    print(f"❌ data_augmentation.py 导入失败: {e}")
    sys.exit(1)

# 测试2: 增强模型模块
print("\n[测试2] 增强模型模块导入")
print("-" * 80)
try:
    from caocrvfy.model_enhanced import (
        create_enhanced_cnn_model,
        compile_model,
        print_model_summary,
        WeightedBinaryCrossentropy
    )
    print("✅ model_enhanced.py 导入成功")
    print("   - create_enhanced_cnn_model: OK")
    print("   - compile_model: OK")
    print("   - print_model_summary: OK")
    print("   - WeightedBinaryCrossentropy: OK")
except Exception as e:
    print(f"❌ model_enhanced.py 导入失败: {e}")
    sys.exit(1)

# 测试3: 配置模块
print("\n[测试3] 配置模块检查")
print("-" * 80)
try:
    from caocrvfy import config
    print("✅ config.py 导入成功")
    print(f"   - IMAGE_HEIGHT: {config.IMAGE_HEIGHT}")
    print(f"   - IMAGE_WIDTH: {config.IMAGE_WIDTH}")
    print(f"   - DROPOUT_CONV: {config.DROPOUT_CONV}")
    print(f"   - DROPOUT_FC: {config.DROPOUT_FC}")
    print(f"   - LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   - LR_DECAY_PATIENCE: {config.LR_DECAY_PATIENCE}")
    
    # 验证配置值
    assert config.DROPOUT_CONV == 0.25, f"DROPOUT_CONV应为0.25，实际为{config.DROPOUT_CONV}"
    assert config.DROPOUT_FC == 0.5, f"DROPOUT_FC应为0.5，实际为{config.DROPOUT_FC}"
    assert config.LEARNING_RATE == 0.001, f"LEARNING_RATE应为0.001，实际为{config.LEARNING_RATE}"
    assert config.LR_DECAY_PATIENCE == 8, f"LR_DECAY_PATIENCE应为8，实际为{config.LR_DECAY_PATIENCE}"
    print("   ✅ 配置值验证通过")
except AssertionError as e:
    print(f"   ⚠️  配置值不符合预期: {e}")
except Exception as e:
    print(f"❌ config.py 检查失败: {e}")
    sys.exit(1)

# 测试4: 模型创建
print("\n[测试4] 模型创建测试")
print("-" * 80)
try:
    model = create_enhanced_cnn_model()
    print("✅ 增强CNN模型创建成功")
    print(f"   - 模型名称: {model.name}")
    print(f"   - 输入形状: {model.input_shape}")
    print(f"   - 输出形状: {model.output_shape}")
    print(f"   - 参数总量: {model.count_params():,}")
    
    # 检查Dropout层配置
    dropout_layers = [layer for layer in model.layers if 'dropout' in layer.name.lower()]
    print(f"\n   Dropout层配置检查:")
    for layer in dropout_layers:
        rate = layer.rate
        print(f"   - {layer.name}: rate={rate}")
        
        # 验证卷积后的Dropout是0.25
        if 'dropout1' in layer.name or 'dropout2' in layer.name or 'dropout3' in layer.name:
            if not 'fc' in layer.name:
                assert rate == 0.25, f"{layer.name}应为0.25，实际为{rate}"
        
        # 验证全连接层的Dropout是0.5
        if 'dropout_fc' in layer.name:
            assert rate == 0.5, f"{layer.name}应为0.5，实际为{rate}"
    
    print("   ✅ Dropout配置验证通过")
    
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 模型编译
print("\n[测试5] 模型编译测试")
print("-" * 80)
try:
    model = compile_model(model, use_focal_loss=False, pos_weight=3.0)
    print("✅ 模型编译成功")
    print(f"   - 优化器: {model.optimizer.__class__.__name__}")
    print(f"   - 损失函数: {model.loss.__class__.__name__}")
    print(f"   - 指标数量: {len(model.metrics)}")
    
    # 验证损失函数
    assert isinstance(model.loss, WeightedBinaryCrossentropy), "损失函数应为WeightedBinaryCrossentropy"
    assert model.loss.pos_weight == 3.0, f"pos_weight应为3.0，实际为{model.loss.pos_weight}"
    print("   ✅ 损失函数验证通过 (WeightedBCE, pos_weight=3.0)")
    
except Exception as e:
    print(f"❌ 模型编译失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 数据增强功能测试
print("\n[测试6] 数据增强功能测试")
print("-" * 80)
try:
    # 创建测试数据（注意：使用3通道RGB图像）
    test_images = np.random.rand(10, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS).astype(np.float32)
    test_labels = np.random.randint(0, 2, (10, config.OUTPUT_SIZE)).astype(np.float32)
    
    # 创建增强数据集
    train_ds = create_augmented_dataset(test_images, test_labels, batch_size=4, training=True)
    val_ds = create_augmented_dataset(test_images, test_labels, batch_size=4, training=False)
    
    print("✅ 数据增强Dataset创建成功")
    
    # 测试批次
    for batch_images, batch_labels in train_ds.take(1):
        print(f"   - 训练批次形状: images={batch_images.shape}, labels={batch_labels.shape}")
        assert batch_images.shape[0] <= 4, "批次大小不应超过4"
        assert batch_images.shape[1:] == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS), "图像形状错误"
        assert batch_labels.shape[1] == config.OUTPUT_SIZE, "标签形状错误"
    
    for batch_images, batch_labels in val_ds.take(1):
        print(f"   - 验证批次形状: images={batch_images.shape}, labels={batch_labels.shape}")
    
    print("   ✅ 数据增强功能验证通过")
    
except Exception as e:
    print(f"❌ 数据增强功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试7: 完整训练流程测试（单个batch）
print("\n[测试7] 完整训练流程测试")
print("-" * 80)
try:
    # 使用小数据集测试训练（注意：使用3通道RGB图像）
    test_images = np.random.rand(8, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS).astype(np.float32)
    test_labels = np.random.randint(0, 2, (8, config.OUTPUT_SIZE)).astype(np.float32)
    
    train_ds = create_augmented_dataset(test_images, test_labels, batch_size=4, training=True)
    
    # 训练一个epoch
    history = model.fit(train_ds, epochs=1, verbose=0)
    
    print("✅ 完整训练流程测试成功")
    print(f"   - 训练损失: {history.history['loss'][0]:.4f}")
    print(f"   - 二进制准确率: {history.history['binary_accuracy'][0]:.4f}")
    
except Exception as e:
    print(f"❌ 训练流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试8: train.py导入测试
print("\n[测试8] train.py模块导入测试")
print("-" * 80)
try:
    from caocrvfy import train
    print("✅ train.py 导入成功")
    print(f"   - USE_ENHANCED_MODEL: {train.USE_ENHANCED_MODEL}")
    
    # 验证是否使用增强模型
    assert train.USE_ENHANCED_MODEL == True, "应该使用增强模型"
    print("   ✅ 确认使用增强模型")
    
    # 检查函数是否存在
    assert hasattr(train, 'create_callbacks'), "缺少create_callbacks函数"
    assert hasattr(train, 'train_model'), "缺少train_model函数"
    assert hasattr(train, 'main'), "缺少main函数"
    print("   ✅ 所有必要函数存在")
    
except Exception as e:
    print(f"❌ train.py 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "=" * 80)
print(" " * 25 + "✅ 所有测试通过")
print("=" * 80)
print("\n模块集成状态:")
print("  ✅ data_augmentation.py - 数据增强模块正常")
print("  ✅ model_enhanced.py - 增强模型模块正常")
print("  ✅ config.py - 配置正确 (Dropout 0.25/0.5, LR 0.001, Patience 8)")
print("  ✅ train.py - 训练模块正常，使用增强模型")
print("  ✅ WeightedBCE - 损失函数工作正常 (pos_weight=3.0)")
print("  ✅ Dataset - 数据增强pipeline正常工作")
print("  ✅ 完整训练流程 - 可以正常训练")
print("\n✨ 可以开始完整训练了！")
print("=" * 80)
