# 优化后的模型使用指南

## 快速开始

### 1. 生成验证码（4位长度）
```bash
cd captcha
python generate_captcha.py
```
生成的验证码：
- ✅ 纯数字：4位 (例如: 3060, 9491)
- ✅ 数字+字母混合：4位 (例如: MdNr, FN6H)

### 2. 测试模型和纠正逻辑
```bash
python test_9layer_model.py
```

### 3. 训练模型
```bash
cd caocrvfy
python train.py
```

## 模型特性

### ✅ 9层卷积架构
- 32 → 64 → 64 → 128 → 128 → 256 → 256 → 512 → 512
- 卷积层使用批归一化（BN）
- **卷积层不使用dropout**
- 全连接层使用0.5的dropout

### ✅ 自适应学习率
- Adam优化器（beta_1=0.9, beta_2=0.999）
- 结合AdaptiveLearningRate回调
- 自动根据训练情况调整

### ✅ 智能数字纠正
- 针对4位验证码
- 当有3位是数字时，自动纠正第4位
- 解决常见混淆：O→0, I→1, Z→2, B→8, S→5, G→6

## 模型信息

```
模型名称：captcha_cnn9
输入形状：(60, 200, 3)
输出形状：(252,) = 4位 × 63个字符
参数量：13,095,548 (约50MB)
```

## 使用示例

### Python代码
```python
from caocrvfy.core import model, utils
import numpy as np

# 1. 加载模型
cnn_model = model.create_cnn_model()
cnn_model = model.compile_model(cnn_model)
cnn_model.load_weights('path/to/weights.keras')

# 2. 预测验证码
image = load_your_image()  # 形状: (60, 200, 3)
image = np.expand_dims(image, axis=0)  # 添加batch维度

prediction = cnn_model.predict(image)

# 3. 解码结果（自动应用数字纠正）
result = utils.vector_to_text(prediction[0], apply_correction=True)
print(f"识别结果: {result}")

# 4. 不使用纠正
result_no_correction = utils.vector_to_text(prediction[0], apply_correction=False)
print(f"原始结果: {result_no_correction}")
```

## 训练建议

### 1. 数据准备
- 确保验证码都是4位长度
- 包含纯数字和混合类型
- 建议至少10000张训练样本

### 2. 训练参数
```python
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 0.001  # Adam初始学习率
```

### 3. 回调配置
```python
from caocrvfy.core.callbacks import (
    AdaptiveLearningRate,
    DelayedEarlyStopping,
    TrainingProgress
)

callbacks = [
    AdaptiveLearningRate(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    DelayedEarlyStopping(
        start_epoch=50,
        patience=35,
        monitor='val_loss'
    ),
    TrainingProgress(val_data=(X_val, y_val))
]
```

## 性能优化

### GPU加速
确保安装了GPU版本的TensorFlow：
```bash
pip install tensorflow-gpu
```

### 混合精度训练
```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## 常见问题

### Q1: 如何处理0和O的混淆？
A: 模型会自动应用数字纠正逻辑。当识别到4位验证码中有3位是数字时，会将字母O纠正为数字0。

### Q2: 如何关闭数字纠正？
A: 在调用vector_to_text时设置 `apply_correction=False`

### Q3: 模型参数太大怎么办？
A: 可以减少卷积层的滤波器数量或减少全连接层的单元数

### Q4: 训练速度慢？
A: 
- 使用GPU训练
- 增大BATCH_SIZE
- 减少图片尺寸
- 使用混合精度训练

## 文件结构

```
tensorflow_cnn_captcha/
├── captcha/
│   ├── generate_captcha.py    # 验证码生成器（4位）
│   └── img/                    # 生成的验证码图片
├── caocrvfy/
│   └── core/
│       ├── model.py            # 9层卷积模型
│       ├── utils.py            # 包含数字纠正逻辑
│       ├── config.py           # 配置（MAX_CAPTCHA=4）
│       └── callbacks.py        # 训练回调
├── test_9layer_model.py        # 测试脚本
└── docs/
    └── OPTIMIZATION_2026-02-03.md  # 详细优化文档
```

## 更新日志

### 2026-02-03
- ✅ 将验证码长度统一为4位
- ✅ 简化为纯数字和混合两种类型
- ✅ 模型改为9层卷积架构
- ✅ 卷积层移除dropout
- ✅ 使用Adam自适应学习率
- ✅ 新增4位验证码数字纠正逻辑
- ✅ 优化配置参数（MAX_CAPTCHA=4, OUTPUT_SIZE=252）
