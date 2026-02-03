# GPU服务器错误修复说明

## 错误信息
```
TypeError: compile_model() got an unexpected keyword argument 'use_focal_loss'
```

## 问题原因
`train_v4.py` 在 `USE_ENHANCED_MODEL=False` 时调用了 `core.model.compile_model()`，但传入了只有 `model_enhanced.compile_model()` 才支持的参数：
- `use_focal_loss`
- `pos_weight`
- `focal_gamma`

## 修复方案

### 方法1：使用增强模型（推荐用于实验Focal Loss）
修改 [train_v4.py](../train_v4.py) 第35行：
```python
USE_ENHANCED_MODEL = True  # 改为True使用增强版模型
```

### 方法2：添加参数兼容性（已实施）✅
在 `core.model.compile_model()` 中添加兼容性参数，自动忽略Focal Loss相关参数。

**修改内容：**
```python
def compile_model(model, learning_rate=None, use_lr_schedule=False, 
                  use_focal_loss=False, pos_weight=None, focal_gamma=None, **kwargs):
    # 如果调用了Focal Loss相关参数，给出警告
    if use_focal_loss or pos_weight or focal_gamma:
        print("⚠️  注意：当前使用core.model，不支持Focal Loss")
        print("    如需使用Focal Loss，请设置 USE_ENHANCED_MODEL = True")
        print("    将使用标准BinaryCrossentropy损失函数\n")
    
    # ... 继续使用标准损失函数
```

## 已修复的问题

### ✅ 问题1：compile_model参数不兼容
**文件**: `caocrvfy/core/model.py`
**修复**: 添加 `use_focal_loss`、`pos_weight`、`focal_gamma`、`**kwargs` 参数支持

### ✅ 问题2：train_v4.py代码语法错误
**文件**: `caocrvfy/train_v4.py` 第66行
**原代码**: `model_dir = os.path.dirname(mode` (不完整)
**修复后**: `model_dir = os.path.dirname(model_dir)`

## 验证测试

### 测试1：参数兼容性 ✅
```bash
cd caocrvfy
python -c "from core.model import create_cnn_model, compile_model; \
m = create_cnn_model(); \
m = compile_model(m, use_focal_loss=True, pos_weight=3.5, focal_gamma=2.0); \
print('✓ 兼容性参数支持正常')"
```

**预期输出**:
```
⚠️  注意：当前使用core.model，不支持Focal Loss
    如需使用Focal Loss，请设置 USE_ENHANCED_MODEL = True
    将使用标准BinaryCrossentropy损失函数

✓ 兼容性参数支持正常
```

### 测试2：train_v4.py语法检查 ✅
```bash
python -m py_compile train_v4.py
```

## GPU服务器部署步骤

### 1. 同步代码到GPU服务器
```bash
# 在本地
scp -r caocrvfy/ ubuntu@your-gpu-server:/home/ubuntu/tensorflowcatpache/

# 或使用git
cd /home/ubuntu/tensorflowcatpache
git pull
```

### 2. 选择训练模式

#### 模式A：使用9层卷积模型（默认，已修复）
```python
# train_v4.py 第35行
USE_ENHANCED_MODEL = False
```
- ✅ 9层卷积网络
- ✅ Adam自适应学习率
- ✅ 标准BinaryCrossentropy损失
- ✅ 数字纠正逻辑

#### 模式B：使用增强模型 + Focal Loss
```python
# train_v4.py 第35行
USE_ENHANCED_MODEL = True
```
- ✅ 5层卷积 + 更大FC层
- ✅ Focal Loss处理困难样本
- ✅ pos_weight加权
- ✅ 数据增强

### 3. 启动训练
```bash
cd /home/ubuntu/tensorflowcatpache/caocrvfy
conda activate tensorflow_env
python train_v4.py
```

### 4. 监控训练进度
```bash
# 实时查看日志
tail -f logs/run_*/events.out.tfevents.*

# 或使用TensorBoard
tensorboard --logdir=logs/ --port=6006
```

## 性能对比

### 9层卷积模型（core.model）
- **参数量**: 13.1M (50MB)
- **训练速度**: 中等
- **准确率**: 预期80-85%
- **特点**: 稳定、通用

### 增强模型（model_enhanced）
- **参数量**: 较少
- **训练速度**: 较快
- **准确率**: 预期75-80%（Focal Loss加成）
- **特点**: 针对困难样本优化

## 建议配置（GPU服务器）

### RTX 4090 (43GB显存)
```python
# config.py
BATCH_SIZE = 256  # 可以增大到256
EPOCHS = 500
LEARNING_RATE = 0.001

# train_v4.py回调配置
max_steps = 200000
checkpoint_save_step = 500
validation_steps = 300
```

### 训练预估
- **每步耗时**: ~0.05-0.1秒
- **总步数**: 200000步
- **预计时间**: 3-5小时
- **检查点数量**: ~400个（每500步保存）

## 常见问题

### Q1: 为什么有两个模型？
**A**: 
- `core.model` (9层卷积): 更深的网络，适合复杂验证码
- `model_enhanced` (5层+Focal): 更快的训练，针对困难样本

### Q2: 如何选择模型？
**A**:
- 初次训练：使用 `core.model`（更稳定）
- 准确率不理想：尝试 `model_enhanced + Focal Loss`
- 0/O混淆严重：使用 `core.model`（已内置数字纠正）

### Q3: 训练中断怎么办？
**A**: 
```python
# 从最近的checkpoint恢复
model = keras.models.load_model('models/checkpoint_step_150000.keras')
# 继续训练
```

## 文件修改清单

### 已修改的文件
1. ✅ `caocrvfy/core/model.py` - 添加Focal Loss参数兼容性
2. ✅ `caocrvfy/train_v4.py` - 修复第66行语法错误

### 测试状态
- ✅ 本地测试通过
- ⏳ GPU服务器待验证

## 下一步操作

1. ✅ 将修复后的代码部署到GPU服务器
2. ⏳ 选择训练模式（建议先用core.model）
3. ⏳ 启动训练
4. ⏳ 监控训练进度和准确率
5. ⏳ 根据结果调整参数

---

**修复日期**: 2026年2月3日  
**测试环境**: Windows 10 + TensorFlow 2.x  
**部署目标**: Ubuntu + RTX 4090 + TensorFlow 2.x
