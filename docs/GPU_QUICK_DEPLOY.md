# GPU服务器快速部署指南

## 🎯 目标
将完整匹配准确率从 **74.57%** 提升至 **85%+**

## 📋 优化方案总结

| 优化项 | 改进内容 | 预期提升 |
|-------|---------|---------|
| **Loss函数** | Focal Loss (gamma=2.0) | +2-3% |
| **模型容量** | FC层 1024→2048 | +1-2% |
| **学习率** | 0.001→0.0008 + 优化衰减 | +1-2% |
| **数据增强** | 启用 | +1-2% |
| **pos_weight** | 3.0→3.5 | +1% |
| **训练步数** | 100000→150000 | +0-1% |
| **总计** | - | **+6-11%** |

---

## 🚀 GPU服务器部署步骤

### 步骤1: 同步最新代码
```bash
cd /data/coding/caocrvfy
git pull origin main
```

### 步骤2: 验证配置（可选）
```bash
python verify_accuracy_boost.py
```

**期望输出**: 🎉 所有优化配置验证通过！

### 步骤3: 启动训练
```bash
# 使用tmux/screen保持会话
tmux new -s training

# 运行训练
python train_v4.py
```

**训练时间**: 约24-30小时（150000步）

---

## 📊 训练监控

### 实时查看日志
```bash
# 另开一个终端
tail -f logs/*.log
```

### 查看完整匹配率
```bash
watch -n 60 'tail -100 logs/*.log | grep "完整匹配"'
```

### 监控checkpoint数量
```bash
ls -lh models/checkpoint_step_*.keras | wc -l
# 应该稳定在3个
```

---

## 🎯 预期训练表现

### 初期 (0-20000步)
- 完整匹配: 30-50%
- 验证损失: 0.15-0.08
- 学习率: warmup中

### 中期 (20000-80000步)
- 完整匹配: 50-75%
- 验证损失: 0.08-0.01
- 学习率: 正常衰减

### 后期 (80000-150000步)
- **完整匹配: 75-85%+** ✅
- 验证损失: <0.01
- 学习率: 精细优化

---

## ⚠️ 关键配置确认

### train_v4.py
```python
USE_ENHANCED_MODEL = True
model = compile_model(
    model, 
    use_focal_loss=True,      # ✅
    pos_weight=3.5,            # ✅
    focal_gamma=2.0            # ✅
)

callbacks = create_callbacks(
    end_acc=0.85,              # ✅ 目标85%
    max_steps=150000           # ✅ 最大150000步
)
```

### core/config.py
```python
FC_UNITS = 2048                 # ✅
USE_DATA_AUGMENTATION = True    # ✅
LEARNING_RATE = 0.0008          # ✅
WARMUP_EPOCHS = 15              # ✅
LR_DECAY_PATIENCE = 12          # ✅
```

---

## 🏁 训练完成标志

训练将在以下情况自动终止：
1. ✅ 完整匹配率 ≥ 85%
2. ✅ 验证损失 ≤ 0.05
3. ✅ 训练步数 ≥ 150000

---

## 📈 效果验证

训练完成后：
```bash
# 查看最终评估
tail -50 logs/*.log | grep -A 10 "模型评估"

# 检查模型文件
ls -lh models/final_model.keras
ls -lh models/best_model.keras
```

**期望输出**:
```
完整匹配准确率: 0.8XXX (8X.XX%)  # ≥85%
二进制准确率: 0.998X
验证损失: 0.00XX
```

---

## 💡 优化亮点

1. **Focal Loss处理困难样本**
   - gamma=2.0强力聚焦错误预测
   - 减少字符混淆（l/I, 0/O等）

2. **模型容量翻倍**
   - FC层2048单元
   - 更强的特征表达能力

3. **精细学习率策略**
   - 更小的初始LR (0.0008)
   - 更长的warmup (15 epochs)
   - 更平滑的衰减 (factor=0.6)

4. **数据增强**
   - 亮度/对比度随机调整
   - 提高模型泛化能力

5. **强化字符识别**
   - pos_weight=3.5
   - 更重视实际字符vs空格

6. **充分训练**
   - 最大150000步
   - 给模型充足时间收敛

---

## 🔍 故障排查

### 如果训练中断
```bash
# 重新进入tmux会话
tmux attach -t training

# 或查看最后的日志
tail -100 logs/*.log
```

### 如果磁盘空间不足
```bash
# 清理旧checkpoint（保留最近3个）
python cleanup_old_checkpoints.py --model-dir models --keep 3 --execute
```

### 如果准确率提升缓慢
- 正常现象，后期提升需要更多步数
- 关注验证损失是否持续下降
- gamma=2.0会让收敛更慢但更精准

---

## 📞 需要帮助？

查看详细文档：
- [准确率提升方案](docs/ACCURACY_BOOST_2026-01-31.md)
- [磁盘空间优化](docs/GPU_DISK_SPACE_OPTIMIZATION.md)
- [训练优化总结](docs/TRAINING_OPTIMIZATION_V4_2026-01-31.md)

---

**状态**: ✅ 所有优化已配置，准备部署  
**预期**: 完整匹配准确率 74.57% → 85%+  
**时间**: 约24-30小时训练
