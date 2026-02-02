#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型保存功能测试脚本
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import config
from core.model_saver import ModelSaver
from extras.model_enhanced import create_enhanced_cnn_model


def test_model_save():
    """测试模型保存功能"""
    print("=" * 80)
    print("模型保存功能测试")
    print("=" * 80)
    
    # 1. 创建模型
    print("\n步骤 1: 创建测试模型...")
    model = create_enhanced_cnn_model()
    print("✓ 模型创建成功")
    
    # 2. 创建测试目录
    test_dir = os.path.join(config.MODEL_DIR, 'save_test')
    os.makedirs(test_dir, exist_ok=True)
    print(f"\n步骤 2: 创建测试目录: {test_dir}")
    
    # 3. 保存模型
    print("\n步骤 3: 保存模型...")
    saver = ModelSaver(test_dir)
    saved_files = saver.save_complete_model(model, 'crack_captcha_model')
    
    # 4. 验证文件
    print("\n步骤 4: 验证生成的文件...")
    expected_files = [
        'crack_captcha_model.keras',
        'checkpoint',
        'ckpt-1.index'
    ]
    
    found_files = []
    missing_files = []
    
    for filename in expected_files:
        filepath = os.path.join(test_dir, filename)
        if os.path.exists(filepath):
            found_files.append(filename)
            file_size = os.path.getsize(filepath) / (1024 ** 2)
            print(f"  ✓ {filename} ({file_size:.2f} MB)")
        else:
            missing_files.append(filename)
            print(f"  ✗ {filename} (未找到)")
    
    # 检查.data文件
    data_files = [f for f in os.listdir(test_dir) if f.startswith('ckpt-') and '.data-' in f]
    if data_files:
        for f in data_files:
            filepath = os.path.join(test_dir, f)
            file_size = os.path.getsize(filepath) / (1024 ** 2)
            print(f"  ✓ {f} ({file_size:.2f} MB)")
            found_files.append(f)
    else:
        print(f"  ✗ ckpt-1.data-* (未找到)")
        missing_files.append('ckpt-1.data-*')
    
    # 5. 测试加载
    print("\n步骤 5: 测试加载模型...")
    try:
        loaded_model = saver.load_keras_model('crack_captcha_model')
        print("✓ 模型加载成功")
        
        # 比较模型结构
        if len(loaded_model.layers) == len(model.layers):
            print(f"✓ 模型层数匹配: {len(model.layers)} 层")
        else:
            print(f"✗ 模型层数不匹配: 原始 {len(model.layers)} vs 加载 {len(loaded_model.layers)}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
    
    # 6. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"预期文件: {len(expected_files) + 1}")  # +1 for .data file
    print(f"找到文件: {len(found_files)}")
    print(f"缺失文件: {len(missing_files)}")
    
    if len(missing_files) == 0:
        print("\n✅ 所有测试通过！模型保存功能正常")
        return True
    else:
        print(f"\n❌ 测试失败！缺失文件: {', '.join(missing_files)}")
        return False


if __name__ == '__main__':
    success = test_model_save()
    sys.exit(0 if success else 1)
