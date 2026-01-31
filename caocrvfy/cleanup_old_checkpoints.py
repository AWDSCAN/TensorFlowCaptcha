#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理旧的checkpoint文件 - 释放磁盘空间
"""

import os
import glob
import argparse
from datetime import datetime

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    return os.path.getsize(filepath) / (1024 * 1024)

def cleanup_checkpoints(model_dir, keep_count=3, dry_run=False):
    """
    清理旧的checkpoint文件
    
    Args:
        model_dir: 模型目录
        keep_count: 保留最近N个checkpoint
        dry_run: 如果为True，只显示将要删除的文件，不实际删除
    """
    print("=" * 70)
    print(f"清理checkpoint文件 - {model_dir}")
    print("=" * 70)
    
    # 查找所有checkpoint文件
    checkpoint_pattern = os.path.join(model_dir, "checkpoint_step_*.keras")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_files:
        print("✓ 没有找到checkpoint文件")
        return
    
    print(f"\n找到 {len(checkpoint_files)} 个checkpoint文件")
    
    # 计算总大小
    total_size = sum(get_file_size_mb(f) for f in checkpoint_files)
    print(f"总大小: {total_size:.2f} MB")
    
    # 确定要删除的文件
    if len(checkpoint_files) > keep_count:
        files_to_delete = checkpoint_files[:-keep_count]
        files_to_keep = checkpoint_files[-keep_count:]
        
        delete_size = sum(get_file_size_mb(f) for f in files_to_delete)
        keep_size = sum(get_file_size_mb(f) for f in files_to_keep)
        
        print(f"\n{'=' * 70}")
        print(f"保留最近 {keep_count} 个checkpoint ({keep_size:.2f} MB):")
        for f in files_to_keep:
            print(f"  ✓ {os.path.basename(f)} ({get_file_size_mb(f):.2f} MB)")
        
        print(f"\n{'=' * 70}")
        print(f"{'[预览模式] ' if dry_run else ''}将删除 {len(files_to_delete)} 个旧checkpoint ({delete_size:.2f} MB):")
        for f in files_to_delete:
            print(f"  {'🔍' if dry_run else '🗑️'}  {os.path.basename(f)} ({get_file_size_mb(f):.2f} MB)")
        
        if not dry_run:
            # 执行删除
            deleted_count = 0
            for f in files_to_delete:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️  删除失败: {f} - {e}")
            
            print(f"\n{'=' * 70}")
            print(f"✅ 成功删除 {deleted_count} 个文件，释放 {delete_size:.2f} MB 空间")
        else:
            print(f"\n{'=' * 70}")
            print(f"💡 预览模式 - 添加 --execute 参数执行实际删除")
    else:
        print(f"\n✓ checkpoint数量({len(checkpoint_files)})未超过保留数量({keep_count})，无需清理")

def main():
    parser = argparse.ArgumentParser(description='清理旧的checkpoint文件')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型目录 (默认: models)')
    parser.add_argument('--keep', type=int, default=3,
                        help='保留最近N个checkpoint (默认: 3)')
    parser.add_argument('--execute', action='store_true',
                        help='执行实际删除（默认为预览模式）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"❌ 模型目录不存在: {args.model_dir}")
        return
    
    cleanup_checkpoints(
        model_dir=args.model_dir,
        keep_count=args.keep,
        dry_run=not args.execute
    )

if __name__ == '__main__':
    main()
