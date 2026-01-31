#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证磁盘空间优化 - 确认所有改进已正确实施
"""

import os
import sys

def check_file_contains(filepath, search_strings):
    """检查文件是否包含指定字符串"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        return all(s in content for s in search_strings)

def main():
    print("=" * 70)
    print("🔍 磁盘空间优化验证")
    print("=" * 70)
    
    checks = []
    
    # 1. 检查 callbacks.py
    print("\n1. 检查 core/callbacks.py")
    callbacks_file = "core/callbacks.py"
    
    check1 = check_file_contains(callbacks_file, ["max_checkpoints=5"])
    print(f"   {'✓' if check1 else '✗'} max_checkpoints参数已添加")
    checks.append(check1)
    
    check2 = check_file_contains(callbacks_file, ["self.checkpoint_files = []"])
    print(f"   {'✓' if check2 else '✗'} checkpoint_files列表已添加")
    checks.append(check2)
    
    check3 = check_file_contains(callbacks_file, ["self.checkpoint_files.append(checkpoint_path)"])
    print(f"   {'✓' if check3 else '✗'} checkpoint追踪逻辑已添加")
    checks.append(check3)
    
    check4 = check_file_contains(callbacks_file, ["if len(self.checkpoint_files) > self.max_checkpoints:"])
    print(f"   {'✓' if check4 else '✗'} 自动清理逻辑已实现")
    checks.append(check4)
    
    check5 = check_file_contains(callbacks_file, ["os.remove(old_checkpoint)"])
    print(f"   {'✓' if check5 else '✗'} 文件删除逻辑已实现")
    checks.append(check5)
    
    check6 = check_file_contains(callbacks_file, ["checkpoint_save_step=500"])
    print(f"   {'✓' if check6 else '✗'} 默认保存间隔已优化为500步")
    checks.append(check6)
    
    check7 = check_file_contains(callbacks_file, ["max_checkpoints_keep=5"])
    print(f"   {'✓' if check7 else '✗'} 默认保留数量为5个")
    checks.append(check7)
    
    # 2. 检查 train_v4.py
    print("\n2. 检查 train_v4.py")
    train_file = "train_v4.py"
    
    check8 = check_file_contains(train_file, ["checkpoint_save_step=500"])
    print(f"   {'✓' if check8 else '✗'} checkpoint_save_step=500已配置")
    checks.append(check8)
    
    check9 = check_file_contains(train_file, ["max_checkpoints_keep=3"])
    print(f"   {'✓' if check9 else '✗'} max_checkpoints_keep=3已配置")
    checks.append(check9)
    
    # 3. 检查清理脚本
    print("\n3. 检查清理脚本")
    cleanup_script = "cleanup_old_checkpoints.py"
    
    check10 = os.path.exists(cleanup_script)
    print(f"   {'✓' if check10 else '✗'} cleanup_old_checkpoints.py已创建")
    checks.append(check10)
    
    # 4. 检查文档
    print("\n4. 检查操作文档")
    doc_file = "../docs/GPU_DISK_SPACE_OPTIMIZATION.md"
    
    check11 = os.path.exists(doc_file)
    print(f"   {'✓' if check11 else '✗'} GPU_DISK_SPACE_OPTIMIZATION.md已创建")
    checks.append(check11)
    
    # 总结
    print("\n" + "=" * 70)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 所有检查通过！({passed}/{total})")
        print("\n✅ 磁盘空间优化已完成：")
        print("   - checkpoint保存间隔：100步 → 500步")
        print("   - checkpoint保留数量：无限制 → 3个")
        print("   - 自动清理旧文件：已启用")
        print("   - 预计磁盘占用：21GB → 252MB")
        print("\n📖 GPU服务器操作指南：docs/GPU_DISK_SPACE_OPTIMIZATION.md")
        return 0
    else:
        print(f"⚠️  {total - passed} 项检查未通过")
        return 1

if __name__ == '__main__':
    sys.exit(main())
