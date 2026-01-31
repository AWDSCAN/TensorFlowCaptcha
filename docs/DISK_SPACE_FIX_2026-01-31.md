# 磁盘空间不足问题修复总结

**日期**: 2026-01-31  
**问题**: GPU服务器训练时出现 `OSError: [Errno 28] No space left on device`  
**影响**: 训练在Epoch 41 (Step 25300) 被迫中断  

---

## 🔍 问题分析

### 根本原因
- **频繁保存**: 每100步保存一次checkpoint
- **无清理机制**: 旧checkpoint永不删除
- **大量堆积**: 训练25000步产生250+个文件
- **磁盘占用**: 每个文件84MB → 总计21GB+

### 错误堆栈
```
Step 25300, Train Loss: 0.0210, Train Acc: 99.39%, Val Loss: 0.0234, Val Acc: 99.11%
保存checkpoint: models/checkpoint_step_25300.keras
OSError: [Errno 28] No space left on device
```

---

## ✅ 已实施的优化

### 1. core/callbacks.py 改进

#### ① 添加参数控制
```python
def __init__(self, val_data, model_dir, save_step=100, validation_steps=500,
             end_acc=0.95, end_loss=0.01, max_steps=50000, max_checkpoints=5):
    # 新增
    self.max_checkpoints = max_checkpoints
    self.checkpoint_files = []
```

#### ② 实现自动清理
```python
def on_batch_end(self, batch, logs=None):
    # ... 保存checkpoint后
    self.checkpoint_files.append(checkpoint_path)
    
    # 自动删除旧文件
    if len(self.checkpoint_files) > self.max_checkpoints:
        old_checkpoint = self.checkpoint_files.pop(0)
        try:
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"  🗑️  删除旧checkpoint: {os.path.basename(old_checkpoint)}")
        except Exception as e:
            print(f"  ⚠️  删除checkpoint失败: {e}")
```

#### ③ 优化默认参数
```python
def create_callbacks(..., checkpoint_save_step=500, validation_steps=500,
                     max_checkpoints_keep=5):
    # 从100步改为500步
    # 添加max_checkpoints_keep参数
```

**文件位置**: [caocrvfy/core/callbacks.py](caocrvfy/core/callbacks.py)

---

### 2. train_v4.py 配置优化

```python
callbacks = create_callbacks(
    model_dir=config.MODEL_DIR,
    log_dir=config.LOG_DIR,
    val_data=(val_images, val_labels),
    use_step_based=True,
    checkpoint_save_step=500,      # ← 优化：500步保存
    validation_steps=500,
    max_checkpoints_keep=3,        # ← 只保留3个
    end_acc=config.END_ACC,
    end_loss=config.END_LOSS,
    max_steps=config.MAX_STEPS
)
```

**文件位置**: [caocrvfy/train_v4.py](caocrvfy/train_v4.py)

---

### 3. 辅助工具创建

#### cleanup_old_checkpoints.py
手动清理旧checkpoint的脚本：
```bash
# 预览模式
python cleanup_old_checkpoints.py --model-dir models --keep 3

# 执行删除
python cleanup_old_checkpoints.py --model-dir models --keep 3 --execute
```

**文件位置**: [caocrvfy/cleanup_old_checkpoints.py](caocrvfy/cleanup_old_checkpoints.py)

#### GPU_DISK_SPACE_OPTIMIZATION.md
GPU服务器完整操作指南，包含：
- 问题分析
- 清理步骤
- 参数调优
- 故障排查

**文件位置**: [docs/GPU_DISK_SPACE_OPTIMIZATION.md](docs/GPU_DISK_SPACE_OPTIMIZATION.md)

---

## 📊 优化效果对比

| 指标 | 优化前 | 优化后 | 改进 |
|-----|--------|--------|------|
| **保存频率** | 100步 | 500步 | 减少80% |
| **保留数量** | 无限制 | 3个 | 固定上限 |
| **文件数(25000步)** | 250个 | 3个 | 减少98.8% |
| **磁盘占用** | 21GB | 252MB | 减少98.8% |
| **自动清理** | ❌ | ✅ | 新增功能 |

---

## 🧪 测试验证

### 本地测试
```bash
cd caocrvfy

# 功能测试
python test_checkpoint_optimization.py  # ✅ 通过

# 完整性验证
python verify_disk_optimization.py      # ✅ 11/11通过
```

### GPU服务器部署步骤

#### 步骤1：清理现有文件（紧急）
```bash
cd /data/coding/caocrvfy
python cleanup_old_checkpoints.py --model-dir models --keep 3 --execute
```
**预期**：释放20GB+磁盘空间

#### 步骤2：同步优化代码
```bash
git pull origin main
# 或手动更新 core/callbacks.py 和 train_v4.py
```

#### 步骤3：重新启动训练
```bash
python train_v4.py
```

#### 步骤4：监控（可选）
```bash
# 查看checkpoint数量
ls -lh models/checkpoint_step_*.keras | wc -l

# 查看磁盘使用
du -sh models/
```

---

## 🔧 参数调优指南

### 场景1：磁盘空间仍然不足
```python
# 更激进的策略
callbacks = create_callbacks(
    checkpoint_save_step=1000,     # 1000步保存
    max_checkpoints_keep=2         # 只保留2个
)
# 磁盘占用: 168MB
```

### 场景2：只需要最佳模型
```python
# 禁用step-based保存
callbacks = create_callbacks(
    use_step_based=False,          # 关闭
    use_model_checkpoint=True      # 只保留val_loss最小的
)
# 磁盘占用: 84MB
```

### 场景3：需要更多历史版本
```python
# 增加保留数量
callbacks = create_callbacks(
    checkpoint_save_step=500,
    max_checkpoints_keep=10        # 保留10个
)
# 磁盘占用: 840MB
```

---

## 📝 文件修改清单

### 修改的文件
1. ✅ `caocrvfy/core/callbacks.py` - 核心优化逻辑
2. ✅ `caocrvfy/train_v4.py` - 配置参数更新

### 新增的文件
1. ✅ `caocrvfy/cleanup_old_checkpoints.py` - 清理工具
2. ✅ `caocrvfy/test_checkpoint_optimization.py` - 功能测试
3. ✅ `caocrvfy/verify_disk_optimization.py` - 验证脚本
4. ✅ `docs/GPU_DISK_SPACE_OPTIMIZATION.md` - 操作指南
5. ✅ `docs/DISK_SPACE_FIX_2026-01-31.md` - 本文档

---

## 🚨 注意事项

### ⚠️ 重要提醒
1. **删除不可恢复**: 清理旧checkpoint后无法找回
2. **保留重要版本**: 如需保留特定checkpoint，请重命名移除`checkpoint_step_`前缀
3. **首次部署**: 建议先运行预览模式 (`--keep 3` 不加 `--execute`)

### 💡 最佳实践
- 定期检查磁盘空间：`df -h`
- 监控checkpoint数量：`ls models/*.keras | wc -l`
- 及时清理无用文件：`python cleanup_old_checkpoints.py --execute`

---

## ✅ 验证清单

- [x] core/callbacks.py 添加 max_checkpoints 参数
- [x] core/callbacks.py 实现自动清理逻辑
- [x] core/callbacks.py 优化默认保存间隔为500步
- [x] train_v4.py 更新callbacks配置
- [x] 创建清理脚本 cleanup_old_checkpoints.py
- [x] 创建操作指南 GPU_DISK_SPACE_OPTIMIZATION.md
- [x] 本地功能测试通过
- [x] 本地完整性验证通过
- [ ] GPU服务器清理旧文件
- [ ] GPU服务器部署新代码
- [ ] GPU服务器训练测试

---

## 📖 相关文档

- [GPU服务器操作指南](GPU_DISK_SPACE_OPTIMIZATION.md)
- [训练优化总结V4](TRAINING_OPTIMIZATION_V4_2026-01-31.md)
- [模块化设计文档](../caocrvfy/docs/MODULAR_DESIGN.md)

---

**修复状态**: ✅ 代码优化完成，等待GPU服务器部署验证  
**预期效果**: 磁盘占用从21GB降至252MB，训练可正常完成
