"""
启动Focal Loss优化训练
清除缓存并开始训练，使用Focal Loss处理困难样本
"""
import os
import shutil
import sys

def clean_pycache(root_dir):
    """清除所有__pycache__目录"""
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                count += 1
                print(f"✓ 已清除: {pycache_path}")
            except Exception as e:
                print(f"✗ 清除失败: {pycache_path}, 错误: {e}")
    return count

if __name__ == "__main__":
    print("=" * 80)
    print(" " * 25 + "Focal Loss 优化训练启动器")
    print("=" * 80)
    print()
    
    # 1. 清除Python缓存
    print("步骤 1/2: 清除Python缓存")
    print("-" * 80)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    count = clean_pycache(current_dir)
    print(f"✓ 共清除 {count} 个 __pycache__ 目录")
    print()
    
    # 2. 启动训练
    print("步骤 2/2: 启动Focal Loss训练")
    print("-" * 80)
    print("✓ 已启用 Focal Loss (gamma=2.0)")
    print("✓ 目标：突破75%准确率瓶颈")
    print("✓ 策略：给困难样本更高权重")
    print()
    
    # 切换到caocrvfy目录并导入训练模块
    sys.path.insert(0, os.path.join(current_dir, 'caocrvfy'))
    from train import main
    
    # 开始训练
    main()
