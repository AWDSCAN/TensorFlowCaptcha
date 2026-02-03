# AdaptiveLearningRate全局配置说明

## 配置变更（2026-02-03）

### 核心变更
✅ **全局使用AdaptiveLearningRate自适应学习率**  
❌ **不再使用固定的LearningRateSchedule（如ExponentialDecay）**

---

## 配置说明

### 1. 学习率策略

**采用双重自适应机制：**

1. **Adam优化器自适应** - 为每个参数维护独立的学习率
2. **AdaptiveLearningRate回调** - 基于验证损失动态调整全局学习率

**配置参数：**
```python
# config.py
LEARNING_RATE = 0.001       # 初始学习率
LEARNING_RATE_MIN = 1e-7    # 最小学习率（AdaptiveLearningRate的下限）

# AdaptiveLearningRate配置
monitor = 'val_loss'        # 监控验证损失
factor = 0.5                # 学习率减半
patience = 5                # 5轮无改善后降低
min_lr = 1e-7               # 最小学习率
```

### 2. 为什么不使用LearningRateSchedule

**问题：**
- `ExponentialDecay`等LearningRateSchedule与`ReduceLROnPlateau`（AdaptiveLearningRate的父类）冲突
- 使用Schedule后，learning rate变为不可修改（immutable）
- 导致TypeError: "learning rate is not settable"

**解决方案：**
- 使用固定初始学习率 + AdaptiveLearningRate
- 让学习率根据训练表现自适应调整（更智能）

### 3. 优势对比

| 特性 | ExponentialDecay | AdaptiveLearningRate |
|------|------------------|---------------------|
| 调整依据 | 固定步数 | 验证损失表现 |
| 智能程度 | 盲目衰减 | 智能响应 |
| 与Adam兼容 | 可能冲突 | 完美兼容 |
| 跳出局部最优 | 较弱 | 较强 |
| 灵活性 | 固定 | 动态 |

---

## 已修改的文件

### 1. `core/config.py`
```python
# 修改前（余弦退火配置）
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 0.00001
WARMUP_STEPS = 5000
COSINE_DECAY_STEPS = 150000
COSINE_ALPHA = 0.01

# 修改后（AdaptiveLearningRate配置）
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 1e-7
# 注意：使用AdaptiveLearningRate进行自适应调整
```

### 2. `core/model.py`
```python
# compile_model函数
def compile_model(model, learning_rate=None, use_lr_schedule=False, ...):
    # 默认 use_lr_schedule=False
    # 如果设为True会给出警告
    
    # 始终使用固定学习率
    lr = initial_lr  # 不使用ExponentialDecay
    
    optimizer = keras.optimizers.Adam(learning_rate=lr, ...)
```

### 3. `trainer.py`
```python
# CaptchaTrainer.__init__
def __init__(self, model, use_exponential_decay=False):
    # 默认 use_exponential_decay=False
    # 如果设为True会给出警告
```

### 4. `train_v4.py`
```python
# 创建训练器
trainer = CaptchaTrainer(
    model=model,
    use_exponential_decay=False  # 不使用指数衰减
)
```

### 5. `core/callbacks.py`
```python
# AdaptiveLearningRate类
class AdaptiveLearningRate(keras.callbacks.ReduceLROnPlateau):
    def on_train_begin(self, logs=None):
        # 检测LearningRateSchedule冲突
        # 如果检测到，自动禁用
        
    def on_epoch_end(self, epoch, logs=None):
        # 捕获TypeError异常
        # 防止与LearningRateSchedule冲突
```

---

## GPU服务器部署

### 1. 同步代码
```bash
cd /home/ubuntu/tensorflowcatpache
git pull
# 或手动上传修改后的文件
```

### 2. 验证配置
```bash
cd caocrvfy
python test_adaptive_lr_config.py
```

**预期输出：**
```
✓ 配置使用AdaptiveLearningRate自适应调整
✓ 使用固定学习率，可被AdaptiveLearningRate调整
✓ 找到AdaptiveLearningRate: 监控=val_loss, factor=0.5
✓ AdaptiveLearningRate已启用且正常工作
```

### 3. 启动训练
```bash
python train_v4.py
```

**预期日志：**
```
📊 自适应学习率已启用
   初始学习率: 0.001000
   监控指标: val_loss
   降低因子: 0.5
   耐心值: 5 epochs
   最小学习率: 1.00e-07
```

---

## 训练行为说明

### 学习率调整示例

**场景1：训练正常**
```
Epoch 1: val_loss=0.05
Epoch 2: val_loss=0.04  ✓ 改善
Epoch 3: val_loss=0.03  ✓ 改善
...
学习率保持 0.001
```

**场景2：出现过拟合**
```
Epoch 10: val_loss=0.02
Epoch 11: val_loss=0.021
Epoch 12: val_loss=0.022
Epoch 13: val_loss=0.023
Epoch 14: val_loss=0.024
Epoch 15: val_loss=0.025

🔻 学习率已调整！
   0.001000 → 0.000500 (降低 50.0%)
   原因: val_loss 在 5 轮内无改善
```

**场景3：继续调整**
```
Epoch 20: 学习率降至 0.000250
Epoch 25: 学习率降至 0.000125
...
最终稳定在 1e-7（最小值）
```

---

## 监控和调试

### 1. 查看学习率变化
```bash
# TensorBoard
tensorboard --logdir=logs/

# 或直接看日志
tail -f train.log | grep "学习率"
```

### 2. 调整AdaptiveLearningRate参数
如果需要调整，修改 `core/callbacks.py` 中的 `create_callbacks` 函数：

```python
adaptive_lr = AdaptiveLearningRate(
    monitor='val_loss',
    factor=0.5,      # 改为0.7可以降低更慢
    patience=5,      # 改为10可以更耐心
    min_lr=1e-7,     # 最小学习率
    verbose=1
)
```

### 3. 常见问题

**Q1: 学习率下降太快？**
```python
# 增加patience
patience=10  # 从5改为10

# 或减小factor
factor=0.7  # 从0.5改为0.7
```

**Q2: 想要更激进的学习率调整？**
```python
# 减小patience
patience=3  # 从5改为3

# 或增大factor（更大的降幅）
factor=0.3  # 从0.5改为0.3
```

**Q3: 遇到LearningRateSchedule冲突？**
- 检查是否误用了 `use_lr_schedule=True`
- AdaptiveLearningRate会自动检测并禁用自己
- 查看日志中的警告信息

---

## 性能预期

### RTX 4090训练表现

**学习率调整曲线（预期）：**
```
Epoch 1-20:   学习率=0.001000 (快速收敛)
Epoch 21-35:  学习率=0.000500 (精细调整)
Epoch 36-50:  学习率=0.000250 (微调优化)
Epoch 51+:    学习率<0.000250 (稳定提升)
```

**准确率提升（预期）：**
```
Epoch 10:  50-60% (初期)
Epoch 30:  70-80% (中期)
Epoch 50:  80-85% (后期)
Epoch 100: 85-90% (收敛)
```

---

## 总结

### ✅ 优势
1. **智能调整** - 基于实际训练表现
2. **无冲突** - 与Adam完美兼容
3. **灵活性高** - 可动态调整参数
4. **易于监控** - 清晰的日志输出

### 📋 注意事项
1. 确保 `use_lr_schedule=False`
2. 确保 `use_exponential_decay=False`
3. 监控 `val_loss` 的变化趋势
4. 如有冲突，检查日志中的警告

### 🎯 下一步
1. 部署到GPU服务器
2. 运行 `test_adaptive_lr_config.py` 验证
3. 启动训练 `train_v4.py`
4. 监控TensorBoard观察学习率变化
5. 根据需要调整patience和factor参数

---

**更新日期**: 2026年2月3日  
**配置状态**: ✅ 已验证通过  
**部署状态**: ⏳ 待GPU服务器验证
