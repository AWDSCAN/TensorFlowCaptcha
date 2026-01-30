# Focal Loss 优化训练指南

## 📋 更新内容

### 1. 核心改进
- ✅ **引入 Focal Loss**: 专门处理困难样本，突破75%准确率瓶颈
- ✅ **保持稳定策略**: 继续使用BatchNormalization + Dropout(0.2/0.4)
- ✅ **优化学习率**: 保持0.0012初始值 + Warmup + ReduceLROnPlateau

### 2. Focal Loss 原理

#### 公式
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

其中:
- `p_t`: 预测为真实类别的概率
- `γ (gamma)`: 聚焦参数，控制对困难样本的关注度（默认2.0）
- `α (alpha)`: 平衡参数（默认0.25）

#### 优势
- **易分样本权重降低**: `(1-p_t)^γ` 项使得高置信度预测损失接近0
- **难分样本权重提升**: 低置信度预测获得更大梯度
- **解决样本不平衡**: 自动关注被误分类的困难字符

### 3. 当前问题分析

#### 训练瓶颈
```
准确率: 74.7% → 75.3% → 77.1% → 75.8% (震荡)
损失值: train=0.0036, val=0.0063 (轻微过拟合)
召回率: 91% (有提升空间)
精确率: 97% (过于保守)
```

#### 困难样本示例
```
真实值: qHSlZt
预测值: qHSKZt
问题: 'l' 和 'K' 混淆（字形相似）
```

### 4. Focal Loss 预期效果

#### 参数选择
- **gamma=2.0**: 标准值，平衡易难样本
- **alpha=0.25**: 适中的平衡系数
- 如需更激进关注困难样本，可尝试gamma=3.0

#### 预期改进
```
当前: 75% → 目标: 82-88%
召回率: 91% → 95%+
困难字符识别: 明显提升
```

## 🚀 使用方法

### 方法1: 使用启动脚本（推荐）
```bash
python start_focal_training.py
```
此脚本会：
1. 清除所有Python缓存（__pycache__）
2. 自动启动Focal Loss训练

### 方法2: 直接训练
```bash
cd caocrvfy
python train.py
```

### 方法3: 手动控制参数
在 [train.py](caocrvfy/train.py#L404) 中修改:
```python
# 启用Focal Loss（默认）
model = compile_model(model, use_focal_loss=True, focal_gamma=2.0)

# 禁用Focal Loss（回退到BCE）
model = compile_model(model, use_focal_loss=False)

# 调整gamma参数（更关注困难样本）
model = compile_model(model, use_focal_loss=True, focal_gamma=3.0)
```

## 📊 训练监控

### 关键指标
1. **完整匹配准确率** (Full Match Accuracy)
   - 目标: > 85%
   - 当前: 75.3%

2. **召回率** (Recall)
   - 目标: > 95%
   - 当前: 91%

3. **验证损失** (Val Loss)
   - 期望: 逐渐下降至 < 0.005
   - 当前: 0.0063

### 预期训练曲线
```
Epoch 1-10:   Warmup阶段，损失快速下降
Epoch 10-30:  Focal Loss开始生效，困难样本改善
Epoch 30-80:  准确率稳步提升至80-85%
Epoch 80-150: 精细调优，接近90%+
```

## 🔧 故障排查

### 问题1: 训练不稳定
**症状**: 验证损失剧烈波动
**解决**: 降低gamma参数
```python
model = compile_model(model, use_focal_loss=True, focal_gamma=1.5)
```

### 问题2: 准确率仍无提升
**症状**: 停留在75-77%
**解决**: 
1. 检查是否正确清除了__pycache__
2. 尝试提高gamma（更关注困难样本）
3. 考虑数据增强（见下节）

### 问题3: ImportError: focal_loss
**症状**: 找不到focal_loss模块
**解决**:
```bash
# 确认focal_loss.py存在
ls caocrvfy/focal_loss.py

# 清除缓存重试
python start_focal_training.py
```

## 📈 后续优化方向

### 如果Focal Loss效果有限
1. **数据增强**:
   ```python
   - 随机旋转 ±5°
   - 颜色抖动
   - 轻微缩放
   ```

2. **注意力机制**:
   ```python
   - 在最后一层卷积后添加Channel Attention
   - 强调重要特征通道
   ```

3. **集成学习**:
   ```python
   - 训练多个模型
   - 投票决策
   ```

## 📝 代码修改记录

### [model_enhanced.py](caocrvfy/model_enhanced.py#L157)
```python
def compile_model(model, learning_rate=None, use_focal_loss=True, focal_gamma=2.0):
    # 新增use_focal_loss和focal_gamma参数
    if use_focal_loss:
        from focal_loss import BinaryFocalLoss
        loss = BinaryFocalLoss(gamma=focal_gamma, alpha=0.25)
    else:
        loss = keras.losses.BinaryCrossentropy()
```

### [train.py](caocrvfy/train.py#L404)
```python
# 默认启用Focal Loss
model = compile_model(model, use_focal_loss=True, focal_gamma=2.0)
```

### [focal_loss.py](caocrvfy/focal_loss.py)
```python
# 新增完整Focal Loss实现
class BinaryFocalLoss(keras.losses.Loss):
    # FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

## ⚠️ 重要提示

1. **清除缓存**: 每次修改代码后必须清除__pycache__
2. **GPU显存**: Focal Loss计算量略大，确保显存充足
3. **训练时长**: 预计150-200轮收敛，约2-3小时（GPU）
4. **早停耐心**: Patience=35，避免过早停止

## 🎯 成功标准

训练成功的标志:
- ✅ 完整匹配准确率 > 85%
- ✅ 召回率 > 95%
- ✅ 精确率 > 95%
- ✅ 验证损失 < 0.005
- ✅ 训练/验证损失比值 < 1.5

达到这些指标后，模型即可用于生产环境。
