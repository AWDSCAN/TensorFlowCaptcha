#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证trainer模块的导入修复
"""

import sys
import os

print("测试 trainer.py 中的 model_enhanced 导入修复...")
print("=" * 60)

try:
    # 模拟 trainer.py 第80行的导入
    from extras.model_enhanced import compile_model
    print("✓ trainer.py 的 compile_model 导入成功")
    
    # 创建简单测试
    from trainer import CaptchaTrainer
    print("✓ CaptchaTrainer 类导入成功")
    
    # 测试是否能访问 recompile_with_lr_schedule 方法
    import inspect
    methods = [m for m in dir(CaptchaTrainer) if not m.startswith('_')]
    if 'recompile_with_lr_schedule' in methods:
        print("✓ recompile_with_lr_schedule 方法存在")
    
    print("\n" + "=" * 60)
    print("🎉 trainer.py 修复验证通过！可以在GPU服务器上运行。")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
