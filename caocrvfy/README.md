# 验证码识别训练系统

基于 TensorFlow + Keras 的验证码识别模型训练代码。

## 目录结构

```
caocrvfy/
├── config.py          # 配置文件
├── utils.py           # 工具函数
├── data_loader.py     # 数据加载器
├── model.py           # CNN模型定义
├── train.py           # 训练脚本
├── predict.py         # 预测脚本
├── models/            # 保存训练好的模型
├── logs/              # TensorBoard日志
└── README.md          # 本文件
```

## 环境要求

```bash
# Python 3.10.x (推荐使用conda环境)
conda create -n tensorflow python=3.10
conda activate tensorflow

# 安装依赖
pip install tensorflow==2.16.1
pip install pillow
pip install numpy
pip install scikit-learn
```

## 快速开始

### 1. 生成训练数据

首先需要生成验证码图片作为训练数据：

```bash
# 进入验证码生成目录
cd ../captcha

# 生成大量验证码图片（建议至少1000张）
python generate_captcha.py
```

### 2. 训练模型

```bash
# 进入训练目录
cd ../caocrvfy

# 开始训练
python train.py
```

训练过程会：
- 自动加载 `captcha/img/` 目录下的验证码图片
- 按 80/20 比例划分训练集和验证集
- 训练 CNN 模型
- 保存最优模型到 `models/best_model.h5`
- 生成 TensorBoard 日志到 `logs/`

### 3. 预测验证码

训练完成后，可以使用预测脚本：

```bash
# 交互式预测
python predict.py

# 预测单张图片
python predict.py --image ../captcha/img/abc123-hash.png

# 预测整个目录
python predict.py --dir ../captcha/img
```

## 模型架构

### CNN 结构

```
输入: (60, 200, 3) RGB图片
    ↓
卷积层1: 32个 3×3 filters + ReLU
    ↓
最大池化: 2×2
    ↓
卷积层2: 64个 3×3 filters + ReLU
    ↓
最大池化: 2×2
    ↓
卷积层3: 64个 3×3 filters + ReLU
    ↓
最大池化: 2×2
    ↓
Dropout: 0.25
    ↓
展平层
    ↓
全连接层: 1024 units + ReLU
    ↓
输出层: 496 units (8×62) + Sigmoid
```

### 参数说明

- **输入尺寸**: 200×60 像素 RGB 图片
- **字符集**: 62 个字符 (0-9 + A-Z + a-z)
- **最大长度**: 8 个字符
- **输出**: 496 维向量 (8 个位置 × 62 个字符的 one-hot 编码)

## 配置参数

所有配置参数在 `config.py` 中定义：

```python
# 图片参数
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
IMAGE_CHANNELS = 3

# 字符集
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
MAX_CAPTCHA = 8

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
```

## 使用示例

### Python 脚本中使用

```python
from predict import CaptchaPredictor

# 创建预测器
predictor = CaptchaPredictor()

# 预测单张图片
text = predictor.predict_image('path/to/captcha.png')
print(f"预测结果: {text}")

# 批量预测
predictions, accuracy = predictor.predict_directory('captcha/img')
```

### 训练自定义模型

```python
from data_loader import CaptchaDataLoader
from model import create_cnn_model, compile_model
from train import train_model, create_callbacks

# 1. 加载数据
loader = CaptchaDataLoader()
loader.load_data()
train_data, val_data = loader.prepare_dataset()

# 2. 创建模型
model = create_cnn_model()
model = compile_model(model, learning_rate=0.0005)

# 3. 训练
callbacks = create_callbacks()
history = train_model(model, train_data, val_data, callbacks=callbacks)
```

## TensorBoard 可视化

训练过程中会生成 TensorBoard 日志，可以实时查看训练进度：

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 在浏览器中打开
# http://localhost:6006
```

## 性能优化

### 数据增强

在 `data_loader.py` 中可以添加数据增强：

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

### GPU 加速

训练脚本会自动检测并使用 GPU：

```python
# 检查GPU
import tensorflow as tf
print("GPU可用:", tf.config.list_physical_devices('GPU'))
```

## 常见问题

### Q: 训练准确率很低怎么办？

A: 可能的原因和解决方案：
1. **数据量不足**: 建议至少 1000 张训练图片
2. **验证码太复杂**: 调整 captcha 生成器减少干扰
3. **学习率不合适**: 尝试调整 `config.LEARNING_RATE`
4. **训练轮数不够**: 增加 `config.EPOCHS`

### Q: 如何提高预测速度？

A: 可以：
1. 使用批量预测而不是单张预测
2. 减小模型尺寸（减少卷积层或神经元数量）
3. 使用模型量化或剪枝
4. 使用 GPU 推理

### Q: 支持哪些类型的验证码？

A: 目前支持：
- 纯数字 (0-9)
- 纯字母 (A-Z, a-z)
- 数字+字母混合
- 数学算术题

## 文件说明

### config.py
全局配置文件，包含所有超参数和路径配置。

### utils.py
工具函数：
- `parse_filename()`: 从文件名提取验证码文本
- `text_to_vector()`: 文本转 one-hot 向量
- `vector_to_text()`: 向量转文本
- `load_image()`: 加载并预处理图片
- `calculate_accuracy()`: 计算准确率

### data_loader.py
数据加载器：
- 自动扫描验证码目录
- 解析文件名提取标签
- 划分训练集和验证集
- 提供批次生成器
- 统计数据分布

### model.py
模型定义：
- CNN 架构定义
- 模型编译
- 模型摘要打印

### train.py
训练脚本：
- 完整的训练流程
- 回调函数配置（检查点、早停、TensorBoard）
- 模型评估
- 模型保存

### predict.py
预测脚本：
- 单张图片预测
- 批量预测
- 目录预测
- 交互式预测模式

## 进阶用法

### 自定义字符集

修改 `config.py`：

```python
# 只识别数字
CHAR_SET = "0123456789"

# 只识别大写字母
CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 自定义字符集
CHAR_SET = "0123456789ABCDEF"  # 十六进制
```

### 修改验证码长度

```python
# config.py
MAX_CAPTCHA = 6  # 改为6位验证码
```

### 调整模型复杂度

```python
# config.py
CONV_FILTERS = [16, 32, 32]  # 减小模型（更快但可能不够准确）
CONV_FILTERS = [64, 128, 128]  # 增大模型（更准确但更慢）

FC_UNITS = 512  # 减小全连接层
FC_UNITS = 2048  # 增大全连接层
```

## 许可证

本项目代码仅供学习和研究使用。

## 更新日志

- **v1.0.0** (2024): 初始版本
  - 支持多类型验证码识别
  - 三层卷积 CNN 架构
  - 完整的训练和预测流程
  - TensorBoard 可视化支持
