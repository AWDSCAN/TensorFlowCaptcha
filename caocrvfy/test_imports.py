#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入测试脚本 - 验证所有模块导入正常
运行方式：python test_imports.py
"""

import sys
import os

# 确保在正确的目录
print(f"当前工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")
print("=" * 60)

def test_imports():
    """测试所有模块导入"""
    
    tests = [
        # Core 模块
        ("core.config", "from core import config"),
        ("core.callbacks", "from core.callbacks import create_callbacks"),
        ("core.evaluator", "from core.evaluator import CaptchaEvaluator"),
        ("core.data_loader", "from core.data_loader import CaptchaDataLoader"),
        ("core.data_augmentation", "from core.data_augmentation import create_augmented_dataset"),
        ("core.model", "from core.model import create_cnn_model"),
        ("core.utils", "from core import utils"),
        
        # Trainer
        ("trainer", "from trainer import CaptchaTrainer"),
        
        # Extras 模块
        ("extras.model_enhanced", "from extras.model_enhanced import create_enhanced_cnn_model"),
        ("extras.focal_loss", "from extras.focal_loss import BinaryFocalLoss"),
        ("extras.predict", "from extras.predict import CaptchaPredictor"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:30s} - 导入成功")
            passed += 1
        except Exception as e:
            print(f"✗ {name:30s} - 导入失败: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有导入测试通过！可以正常运行训练脚本。")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个模块导入失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit_code = test_imports()
    sys.exit(exit_code)
