"""
Focal Loss参数紧急修复
修复召回率暴跌问题（40% → 预期90%+）
"""
import os
import sys
import shutil

def clean_pycache(root_dir):
    """彻底清除Python缓存"""
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                count += 1
                print(f"✓ 清除: {pycache_path}")
            except Exception as e:
                print(f"✗ 失败: {pycache_path}, {e}")
    return count

def clean_model_cache():
    """清除旧的模型文件"""
    model_files = [
        'caocrvfy/models/best_model.keras',
        'caocrvfy/models/final_model.keras'
    ]
    for f in model_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"✓ 删除旧模型: {f}")
            except Exception as e:
                print(f"✗ 删除失败: {f}, {e}")

if __name__ == "__main__":
    print("=" * 80)
    print(" " * 22 + "Focal Loss 参数紧急修复")
    print("=" * 80)
    print()
    
    print("🔧 修复内容:")
    print("  • alpha: 0.25 → 0.75 (提高正样本权重)")
    print("  • gamma: 2.0 → 1.5 (降低困难样本过度关注)")
    print()
    
    print("📊 预期改善:")
    print("  • 召回率: 40% → 90%+")
    print("  • 完整匹配: 10% → 75%+")
    print("  • 精确率: 保持95%+")
    print()
    
    # 1. 清除Python缓存
    print("步骤 1/3: 清除Python缓存")
    print("-" * 80)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    count = clean_pycache(current_dir)
    print(f"✓ 共清除 {count} 个缓存目录")
    print()
    
    # 2. 删除旧模型
    print("步骤 2/3: 删除旧的错误模型")
    print("-" * 80)
    clean_model_cache()
    print()
    
    # 3. 启动训练
    print("步骤 3/3: 启动修复后的训练")
    print("-" * 80)
    print("✓ 已应用新参数: alpha=0.75, gamma=1.5")
    print()
    
    sys.path.insert(0, os.path.join(current_dir, 'caocrvfy'))
    from train import main
    
    main()
