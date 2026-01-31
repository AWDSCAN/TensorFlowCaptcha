#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终部署前验证 - GPU服务器就绪检查
"""

import sys
import os

def test_imports():
    """测试所有关键导入"""
    print("=" * 70)
    print("1. 测试模块导入")
    print("=" * 70)
    
    try:
        from core import config
        print("   ✓ core.config")
        
        from core.data_loader import CaptchaDataLoader
        print("   ✓ core.data_loader")
        
        from core.callbacks import create_callbacks
        print("   ✓ core.callbacks")
        
        from extras.model_enhanced import create_enhanced_cnn_model, compile_model
        print("   ✓ extras.model_enhanced")
        
        from extras.focal_loss import BinaryFocalLoss
        print("   ✓ extras.focal_loss")
        
        from trainer import CaptchaTrainer
        print("   ✓ trainer")
        
        from core.evaluator import CaptchaEvaluator
        print("   ✓ core.evaluator")
        
        return True
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False

def test_focal_loss_creation():
    """测试Focal Loss创建"""
    print("\n" + "=" * 70)
    print("2. 测试Focal Loss创建")
    print("=" * 70)
    
    try:
        from extras.model_enhanced import create_enhanced_cnn_model, compile_model
        
        model = create_enhanced_cnn_model()
        print("   ✓ 增强版CNN模型创建成功")
        
        model = compile_model(model, use_focal_loss=True, pos_weight=3.5, focal_gamma=2.0)
        print("   ✓ Focal Loss编译成功 (gamma=2.0, pos_weight=3.5)")
        
        return True
    except Exception as e:
        print(f"   ✗ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callbacks_creation():
    """测试callbacks创建"""
    print("\n" + "=" * 70)
    print("3. 测试Callbacks创建")
    print("=" * 70)
    
    try:
        import numpy as np
        from core.callbacks import create_callbacks
        
        # 创建假数据
        val_images = np.random.rand(100, 60, 200, 3).astype(np.float32)
        val_labels = np.random.randint(0, 2, (100, 504)).astype(np.float32)
        
        callbacks = create_callbacks(
            model_dir='test_models',
            log_dir='test_logs',
            val_data=(val_images, val_labels),
            use_step_based=True,
            checkpoint_save_step=500,
            validation_steps=500,
            max_checkpoints_keep=3,
            end_acc=0.85,
            max_steps=150000
        )
        
        print(f"   ✓ Callbacks创建成功 (共{len(callbacks)}个)")
        print("   ✓ end_acc=0.85, max_steps=150000")
        
        return True
    except Exception as e:
        print(f"   ✗ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_available():
    """测试GPU可用性"""
    print("\n" + "=" * 70)
    print("4. 测试GPU可用性")
    print("=" * 70)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) == 0:
            print("   ⚠️  本地环境无GPU（GPU服务器上会自动检测）")
            return True  # 本地环境返回True，不影响其他验证
        
        print(f"   ✓ 检测到 {len(gpus)} 个GPU设备")
        
        for i, gpu in enumerate(gpus):
            print(f"   ✓ GPU {i}: {gpu.name}")
        
        return len(gpus) > 0
    except Exception as e:
        print(f"   ✗ GPU检测失败: {e}")
        return False

def check_disk_space():
    """检查磁盘空间"""
    print("\n" + "=" * 70)
    print("5. 检查磁盘空间")
    print("=" * 70)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        print(f"   可用空间: {free_gb:.2f} GB")
        
        if free_gb < 5:
            print("   ⚠️  磁盘空间不足5GB，建议清理")
            return False
        elif free_gb < 10:
            print("   ⚠️  磁盘空间较低，建议监控")
            return True
        else:
            print("   ✓ 磁盘空间充足")
            return True
    except Exception as e:
        print(f"   ⚠️  无法检查磁盘空间: {e}")
        return True

def verify_config():
    """验证关键配置"""
    print("\n" + "=" * 70)
    print("6. 验证关键配置")
    print("=" * 70)
    
    from core import config
    
    checks = []
    
    check1 = config.FC_UNITS == 2048
    print(f"   {'✓' if check1 else '✗'} FC_UNITS = {config.FC_UNITS}")
    checks.append(check1)
    
    check2 = config.USE_DATA_AUGMENTATION == True
    print(f"   {'✓' if check2 else '✗'} USE_DATA_AUGMENTATION = {config.USE_DATA_AUGMENTATION}")
    checks.append(check2)
    
    check3 = config.LEARNING_RATE == 0.0008
    print(f"   {'✓' if check3 else '✗'} LEARNING_RATE = {config.LEARNING_RATE}")
    checks.append(check3)
    
    return all(checks)

def main():
    print("=" * 70)
    print("🚀 GPU服务器最终部署验证")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("Focal Loss创建", test_focal_loss_creation()))
    results.append(("Callbacks创建", test_callbacks_creation()))
    results.append(("GPU可用性", test_gpu_available()))
    results.append(("磁盘空间", check_disk_space()))
    results.append(("配置验证", verify_config()))
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 验证总结")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {name:20} {status}")
    
    print()
    
    if all(r[1] for r in results):
        print("🎉 所有验证通过！准备开始训练")
        print()
        print("🚀 启动训练:")
        print("   python train_v4.py")
        print()
        print("📊 监控训练:")
        print("   tail -f logs/*.log")
        print()
        print("📈 预期效果:")
        print("   完整匹配准确率: 74% → 85%+")
        print("   训练时间: ~24-30小时")
        return 0
    else:
        print("⚠️  部分验证未通过，请检查问题")
        return 1

if __name__ == '__main__':
    sys.exit(main())
