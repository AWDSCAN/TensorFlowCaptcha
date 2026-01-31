# 类别不平衡问题修复 - 2026-01-31

## 🚨 紧急问题：模型不学习实际字符

### GPU服务器训练症状
```
Epoch 1-5:
  二进制准确率: 76% → 99% ✓ (看似正常)
  召回率: 45% → 37% ❌ (持续下降！)
  精确率: 3% → 92% ✓ (模型极度保守)
  完整匹配: 0.00% ❌ (每个验证码都错)
```

**诊断：模型学会了过度预测负类（padding/空格）**

## 🔍 根本原因分析

### 类别严重不平衡
```
输出维度结构:
  • 总维度: 8字符 × 63类 = 504维
  • 实际字符位: 平均4-6个 (32-48维为1)
  • Padding位: 2-4个 (剩余456-472维为0)
  • 负样本比例: >90%

模型学习策略:
  ✗ 预测所有位为0 → 二进制准确率90%+
  ✗ 仅预测极少量确信的字符 → 精确率92%，召回率37%
  ✗ 完整验证码必错 → 完整匹配0%
```

### 为什么标准BCE Loss失效？

标准BCE对所有位一视同仁：
- 正确预测1个padding: +1分
- 错误漏掉1个实际字符: -1分
- **结果**: 模型倾向于保守预测，避免false positive

## ✅ 解决方案：加权BCE Loss

### 核心思想
给正类（实际字符）**更高的损失权重**，迫使模型必须关注它们。

### 实现细节

#### 1. WeightedBinaryCrossentropy类
```python
class WeightedBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, pos_weight=3.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        # 正类损失 × pos_weight
        pos_loss = -y_true * tf.math.log(y_pred) * self.pos_weight
        # 负类损失保持不变
        neg_loss = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(pos_loss + neg_loss)
```

**关键参数: pos_weight=3.0**
- 含义: 漏掉一个实际字符的惩罚 = 误判3个padding的惩罚
- 效果: 强制模型更关注召回率

#### 2. 修改编译函数
```python
def compile_model(model, pos_weight=3.0):
    loss = WeightedBinaryCrossentropy(pos_weight=pos_weight)
    # ...
```

#### 3. 训练调用
```python
model = compile_model(model, use_focal_loss=False, pos_weight=3.0)
```

## 📊 预期效果对比

### 修复前（标准BCE）
```
Epoch 5:
  二进制准确率: 99%
  召回率: 37%  ❌
  精确率: 92%
  完整匹配: 0%  ❌
  
问题: 模型过度保守，漏掉大量实际字符
```

### 修复后（加权BCE, pos_weight=3.0）
```
预期 Epoch 10-20:
  二进制准确率: 95-97%  (略降，正常)
  召回率: 85-90%  ✅ (+48-53%)
  精确率: 90-95%  ✅ (保持高位)
  完整匹配: 60-75%  ✅ (+60%)
  
改进: 模型积极预测字符，召回率大幅提升
```

## 🎯 权重选择指南

### pos_weight参数调优

| pos_weight | 召回率预期 | 精确率预期 | 适用场景 |
|-----------|----------|----------|---------|
| 1.0 | 40-50% | 95%+ | 标准BCE（当前问题） |
| 2.0 | 70-80% | 90-95% | 轻度不平衡 |
| **3.0** | **85-90%** | **90-95%** | **中度不平衡（推荐）** |
| 4.0 | 90-95% | 85-90% | 重度不平衡 |
| 5.0+ | 95%+ | 80-85% | 极度不平衡（可能过拟合） |

**当前选择: pos_weight=3.0**
- 理由: 平衡召回率和精确率
- 预期F1-score: ~87-92%

## 🔧 代码变更清单

### 1. [model_enhanced.py](caocrvfy/model_enhanced.py)

**新增类:**
```python
class WeightedBinaryCrossentropy(keras.losses.Loss):
    # 加权BCE实现
    # pos_weight=3.0：正类损失权重
```

**修改函数签名:**
```python
def compile_model(model, ..., pos_weight=3.0):
    loss = WeightedBinaryCrossentropy(pos_weight=pos_weight)
```

### 2. [train.py](caocrvfy/train.py#L404)

**修改编译调用:**
```python
# 旧: model = compile_model(model, use_focal_loss=False)
# 新:
model = compile_model(model, use_focal_loss=False, pos_weight=3.0)
```

## 🧪 验证测试

### 运行测试脚本
```bash
python test_weighted_bce.py
```

**预期输出:**
```
【2. 测试错误预测惩罚】
模型过度保守（召回率低）:
  标准BCE Loss: 0.654667
  加权BCE Loss: 1.858639
  惩罚提升: 2.84x  ✅
```

**解释:**
- 加权BCE对召回率低的预测惩罚是标准BCE的2.84倍
- 这迫使模型必须提高召回率

## 📈 训练监控要点

### 关键指标变化（前10轮）

**Epoch 1-5: Warmup阶段**
```
预期变化:
  召回率: 45% → 60-70%  (快速上升)
  精确率: 92% → 85-90%  (轻微下降，正常)
  完整匹配: 0% → 10-20%  (开始生效)
```

**Epoch 5-10: 加速学习**
```
预期变化:
  召回率: 70% → 85%  (持续上升)
  精确率: 88% → 92%  (回升)
  完整匹配: 20% → 50%  (快速提升)
```

**Epoch 10-30: 稳定收敛**
```
预期变化:
  召回率: 85% → 90%  (逼近目标)
  精确率: 92% → 95%  (优化)
  完整匹配: 50% → 75%  (达标)
```

### ⚠️ 异常情况处理

**情况1: 召回率仍然低 (<60%)**
```
原因: pos_weight太小
解决: 提高到4.0或5.0
修改: compile_model(model, pos_weight=4.0)
```

**情况2: 精确率过低 (<80%)**
```
原因: pos_weight太大
解决: 降低到2.0
修改: compile_model(model, pos_weight=2.0)
```

**情况3: 训练不稳定（损失震荡）**
```
原因: pos_weight太大 + 学习率太高
解决: 
  1. 降低pos_weight到2.5
  2. 降低学习率到0.001
```

## 🚀 GPU服务器部署

### 立即执行步骤

1. **清除旧缓存**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -f caocrvfy/models/best_model.keras
```

2. **启动训练**
```bash
cd caocrvfy
python train.py
```

3. **监控关键指标**
```bash
# 实时查看日志
tail -f logs/training_*.log

# 重点关注:
# - Epoch 5: 召回率应 >60%
# - Epoch 10: 召回率应 >80%, 完整匹配 >40%
# - Epoch 20: 召回率应 >85%, 完整匹配 >60%
```

### 预期训练时间
- GPU服务器 (2×RTX 4090): **1.5-2小时**
- 60000样本 × 150轮 ≈ 120分钟

## 📋 成功标准

训练成功的标志:
- ✅ 召回率 ≥ 85% (从37%提升)
- ✅ 精确率 ≥ 90% (保持高位)
- ✅ 完整匹配 ≥ 70% (从0%提升)
- ✅ F1-score ≥ 87%

## 🎓 技术原理深入

### 为什么加权有效？

#### 梯度分析
标准BCE梯度:
```
∂L/∂w = (y_pred - y_true)
```

加权BCE梯度:
```
∂L/∂w = pos_weight × (y_pred - y_true)  当y_true=1
∂L/∂w = (y_pred - y_true)                当y_true=0
```

**效果:**
- 正类错误 → 梯度×3 → 更新更大 → 更快学习
- 负类错误 → 梯度×1 → 正常更新 → 保持稳定

#### 优化轨迹
```
标准BCE:
  模型倾向 → 最小化总损失
          → 优先学习占比90%的负类
          → 忽略占比10%的正类
          → 召回率低

加权BCE:
  模型倾向 → 正类损失×3
          → 正负类损失权重平衡
          → 同时关注正负类
          → 召回率高
```

## 📝 实验记录

### 实验1: 标准BCE (已完成)
```
配置: pos_weight=1.0 (标准BCE等价)
结果: 召回率37%, 完整匹配0%
结论: 失败 - 类别不平衡导致模型退化
```

### 实验2: 加权BCE pos_weight=3.0 (进行中)
```
配置: WeightedBinaryCrossentropy(pos_weight=3.0)
预期: 召回率85%+, 完整匹配70%+
状态: 待GPU服务器验证
```

### 实验3: 备选方案 (如需要)
```
配置1: pos_weight=4.0 (更激进)
配置2: pos_weight=2.0 (更保守)
配置3: Focal Loss + 加权 (组合策略)
```

## 🔗 相关文档

- [TRAINING_OPTIMIZATION_2026-01-30_FINAL.md](TRAINING_OPTIMIZATION_2026-01-30_FINAL.md) - 之前的优化记录
- [focal_loss.py](../caocrvfy/focal_loss.py) - Focal Loss实现（已弃用）
- [model_enhanced.py](../caocrvfy/model_enhanced.py) - 模型定义
- [train.py](../caocrvfy/train.py) - 训练脚本

---

**最后更新**: 2026-01-31 09:20
**状态**: 等待GPU服务器验证
**优先级**: 🔴 紧急 - 解决训练不学习问题
