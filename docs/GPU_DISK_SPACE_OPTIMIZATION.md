# GPU服务器磁盘空间优化指南

## 问题描述

训练时遇到 `OSError: [Errno 28] No space left on device` 错误，原因是：
- 每100步保存一次checkpoint
- 训练到25000+步产生250+个checkpoint文件  
- 每个.keras文件约84MB，总计超过20GB磁盘空间

## 已实施的优化

### 1. 代码级优化（已完成）

**core/callbacks.py** 改进：
- ✅ checkpoint保存间隔：100步 → 500步（减少80%文件数）
- ✅ 自动清理机制：只保留最近N个checkpoint
- ✅ 新增参数：`max_checkpoints`控制保留数量

**train_v4.py** 配置优化：
```python
callbacks = create_callbacks(
    checkpoint_save_step=500,      # 500步保存一次
    validation_steps=500,
    max_checkpoints_keep=3         # 只保留最近3个
)
```

**预期效果**：
- 磁盘占用：21GB → 252MB（减少98.8%）
- checkpoint数量：250+ → 3个

### 2. 测试验证

本地测试通过：
```bash
cd caocrvfy
python test_checkpoint_optimization.py
```

结果：✅ 所有配置验证通过

## GPU服务器操作步骤

### 步骤1：清理现有checkpoint（紧急）

```bash
# 进入项目目录
cd /data/coding/caocrvfy

# 【预览模式】查看将要删除的文件
python cleanup_old_checkpoints.py --model-dir models --keep 3

# 【执行删除】只保留最近3个checkpoint
python cleanup_old_checkpoints.py --model-dir models --keep 3 --execute
```

**说明**：
- `--keep 3`：保留最近3个checkpoint
- `--execute`：执行实际删除（默认为预览模式）
- 预计释放：20GB+ 磁盘空间

### 步骤2：同步优化后的代码

```bash
# 拉取最新代码
git pull

# 或手动更新这两个文件：
# - caocrvfy/core/callbacks.py
# - caocrvfy/train_v4.py
```

### 步骤3：重新运行训练

```bash
cd /data/coding/caocrvfy
python train_v4.py
```

**新的训练特性**：
- 每500步保存一次checkpoint（vs 之前100步）
- 自动删除旧checkpoint，只保留最近3个
- 磁盘占用稳定在 252MB（3个文件 × 84MB）

### 步骤4：监控磁盘使用（可选）

```bash
# 查看models目录大小
du -sh models/

# 查看checkpoint文件数量
ls -lh models/checkpoint_step_*.keras | wc -l

# 实时监控磁盘空间
watch -n 60 'df -h | grep /data && ls -lh models/checkpoint_step_*.keras | tail -5'
```

## 参数调优建议

如果训练过程中仍然遇到磁盘空间问题，可以调整参数：

### 方案A：更激进的清理策略
```python
# train_v4.py 修改
callbacks = create_callbacks(
    checkpoint_save_step=1000,     # 1000步保存一次
    validation_steps=1000,
    max_checkpoints_keep=2         # 只保留2个
)
```

### 方案B：仅保留最终模型
```python
# 完全禁用step-based checkpoint
callbacks = create_callbacks(
    use_step_based=False,          # 禁用step-based保存
    use_model_checkpoint=True      # 只保留最佳模型
)
```

## 磁盘空间计算

| 配置 | 保存频率 | 保留数量 | 文件数（训练25000步） | 磁盘占用 |
|------|---------|---------|---------------------|---------|
| 旧配置 | 100步 | 无限制 | 250个 | ~21GB |
| 新配置 | 500步 | 3个 | 3个 | ~252MB |
| 激进配置 | 1000步 | 2个 | 2个 | ~168MB |
| 仅最佳模型 | - | 1个 | 1个 | ~84MB |

## 常见问题

### Q1: 删除旧checkpoint会影响训练恢复吗？
A: 不会。训练恢复只需要最后一个checkpoint，旧的checkpoint仅用于回溯历史版本。

### Q2: 如何手动保留某个特定的checkpoint？
A: 重命名文件，移除`checkpoint_step_`前缀：
```bash
mv models/checkpoint_step_15000.keras models/important_backup_15000.keras
```

### Q3: 如何完全禁用checkpoint保存？
A: 修改train_v4.py：
```python
callbacks = create_callbacks(
    use_step_based=False,          # 禁用step-based
    use_model_checkpoint=False,    # 禁用model checkpoint
    use_early_stopping=True        # 只保留early stopping
)
```

## 回滚方案

如果新配置有问题，可以临时恢复旧行为：

```python
# train_v4.py - 临时回退
callbacks = create_callbacks(
    checkpoint_save_step=100,      # 恢复100步保存
    max_checkpoints_keep=10        # 增加保留数量
)
```

## 联系支持

如遇到其他问题：
1. 检查训练日志：`tail -100 caocrvfy/logs/training.log`
2. 查看磁盘使用：`df -h`
3. 提供错误堆栈信息
