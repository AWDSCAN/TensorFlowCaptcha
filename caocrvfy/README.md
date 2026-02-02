# caocrvfy - 验证码识别模块

基于CNN的验证码识别系统，采用模块化设计，易于维护和扩展。

## ⚡ 重要更新 (2026-02-02)

### 🔥 v4.1 - 余弦退火优化版 (最新)

**核心改进**：
- ✅ **余弦退火学习率** - 4-6小时训练（比之前快40%）
- ✅ **完整模型保存** - 支持.keras + checkpoint双格式
- ✅ **Focal Loss** - 专注困难样本(数学题)
- ✅ **精细优化** - 后期学习率降至0.00001

**性能提升**：
- 训练时间：8-12小时 → **4-6小时** (-40%)
- 完整匹配：63% → **78-85%** (+15-22%)
- 收敛速度：提升 **50%+**

📖 **详细说明**：[UPDATE_v4.1_COSINE_ANNEALING.md](../docs/UPDATE_v4.1_COSINE_ANNEALING.md)

### 📚 文档导航
- 🔥 **[v4.1更新说明](../docs/UPDATE_v4.1_COSINE_ANNEALING.md)** - **最新** 余弦退火优化
- 🚀 **[快速开始](../docs/ONE_PAGE_SUMMARY.md)** - 一页总结
- 📋 **[执行清单](../docs/EXECUTION_CHECKLIST.md)** - 完整训练流程
- 🔧 **[快速指南](../docs/QUICK_START_OPTIMIZED_TRAINING.md)** - 监控与调整
- 📊 **[详细方案](../docs/TRAINING_BREAKTHROUGH_2026-02-02.md)** - 技术细节

### 🎯 立即开始训练
```bash
cd caocrvfy

# 1. 测试模型保存（新功能）
python test_model_save.py

# 2. 验证优化配置
python verify_optimization.py

# 3. 开始训练（v4.1余弦退火版）
python train_v4.py

# 或后台运行
nohup python -u train_v4.py > training_v4.1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**预期效果**：
- ⏱️ 训练时间：**4-6小时** (GPU A100)
- 🎯 完整匹配：**78-85%**
- 📈 收敛速度：比v4.0快 **50%+**

---

## 📁 目录结构

```
caocrvfy/
├── train_v4.py              # 🎯 新版主程序（推荐使用）
├── train.py                 # 📌 原版主程序（向后兼容）
├── trainer.py               # 🔧 训练器模块
├── README.md                # 📖 本文档
│
├── core/                    # 核心模块
│   ├── __init__.py         
│   ├── config.py           # 配置文件
│   ├── callbacks.py        # 训练回调
│   ├── evaluator.py        # 模型评估
│   ├── data_loader.py      # 数据加载
│   ├── data_augmentation.py # 数据增强
│   ├── model.py            # 基础模型
│   └── utils.py            # 工具函数
│
├── extras/                  # 额外功能
│   ├── __init__.py
│   ├── model_enhanced.py   # 增强版模型
│   ├── focal_loss.py       # Focal Loss
│   ├── predict.py          # 预测脚本
│   └── quick_verify.py     # 快速验证
│
├── docs/                    # 文档
│   ├── MODULAR_DESIGN.md   # 模块化设计文档
│   ├── REFACTORING_SUMMARY.md # 重构总结
│   └── README.md           # 文档副本
│
├── models/                  # 模型保存目录
└── logs/                    # 日志目录
```

## 🚀 快速开始

### 使用新版模块化训练（推荐）

```bash
cd caocrvfy
python train_v4.py
```

**重要**: 必须在 `caocrvfy/` 目录下运行（包含 `core/` 和 `extras/` 的父目录）

**特点**：
- ✅ 清晰的模块化结构
- ✅ Step-based验证策略
- ✅ 指数衰减学习率
- ✅ 完整的回调管理

### 测试导入是否正常

```bash
cd caocrvfy
python test_imports.py
```

如果所有测试通过，即可正常运行训练。

### 使用原版训练（向后兼容）

```bash
cd caocrvfy
python train.py
```

## 📦 核心模块

### core/ - 核心功能
- **config.py** - 统一配置管理
- **callbacks.py** - 训练回调（5个回调类）
- **evaluator.py** - 模型评估
- **data_loader.py** - 数据加载
- **data_augmentation.py** - 数据增强
- **model.py** - 基础CNN模型
- **utils.py** - 工具函数

### extras/ - 额外功能
- **model_enhanced.py** - 增强版模型（5层卷积）
- **focal_loss.py** - Focal Loss实现
- **predict.py** - 预测脚本
- **quick_verify.py** - 快速验证

### 主程序
- **trainer.py** - 训练器封装
- **train_v4.py** - 新版主程序（120行）
- **train.py** - 原版主程序（向后兼容）

## 💡 使用示例

```python
from core import config
from core.data_loader import CaptchaDataLoader
from core.callbacks import create_callbacks
from trainer import CaptchaTrainer
from core.evaluator import CaptchaEvaluator
from extras.model_enhanced import create_enhanced_cnn_model, compile_model

# 1. 加载数据
loader = CaptchaDataLoader()
train_images, train_labels, val_images, val_labels = loader.load_data()

# 2. 创建模型
model = create_enhanced_cnn_model()
model = compile_model(model)

# 3. 训练
trainer = CaptchaTrainer(model)
history = trainer.train(...)

# 4. 评估
evaluator = CaptchaEvaluator(model)
evaluator.generate_report(val_data)
```

## 📚 详细文档

- [模块化设计文档](docs/MODULAR_DESIGN.md)
- [重构总结](docs/REFACTORING_SUMMARY.md)

## 🎯 设计理念

- **单一职责**: 每个模块只负责一个功能
- **松耦合**: 模块间依赖最小化
- **易维护**: 功能划分清晰
- **易扩展**: 添加新功能不影响现有代码

参考 `test/captcha_trainer` 模块化架构设计。

---

**版本**: v4.0 | **更新**: 2026-01-31
