# 模型测试与转换工具使用指南

## 📦 工具说明

### 1. test_model.py - 模型测试脚本
用于本地验证训练效果，评估模型性能

### 2. convert_to_onnx.py - ONNX转换脚本
将Keras模型转换为ONNX格式，用于跨平台部署

---

## 🧪 test_model.py 使用方法

### 基础使用

```bash
# 测试默认模型（core/models/final_model.keras）
python test_model.py

# 测试指定模型
python test_model.py --model models/best_model.keras

# 测试GPU服务器训练的模型
python test_model.py --model /data/coding/caocrvfy/core/models/final_model.keras
```

### 高级功能

```bash
# 显示更多示例（默认20个）
python test_model.py --samples 50

# 只显示错误预测
python test_model.py --only-errors

# 详细错误分析
python test_model.py --analyze-errors

# 保存评估报告
python test_model.py --report evaluation_report.txt

# 组合使用
python test_model.py --samples 100 --only-errors --analyze-errors --report report.txt
```

### 输出内容

#### 1. 模型信息
```
📊 模型信息:
   输入形状: (None, 60, 200, 3)
   输出形状: (None, 504)
   参数量: 10,234,567
   文件大小: 336.37 MB
```

#### 2. 性能指标
```
📈 模型评估
验证集性能:
   loss: 0.007500
   binary_accuracy: 0.9987
   precision: 0.9508
   recall: 0.9649

✨ 完整匹配准确率: 0.7720 (77.20%)
```

#### 3. 预测示例
```
📝 预测示例
真实值              预测值              匹配        
--------------------------------------------------------------
NZlT47u             NZlT47u             ✓         
PCBEa4Fb            PCBEa Fb            ✗         
40577912            40577912            ✓         
4mjCR2vO            4micR2yO            ✗         
```

#### 4. 错误分析
```
🔍 错误分析
错误统计:
   总样本数: 20000
   错误数量: 4560
   错误率: 22.80%

错误类型分布:
   字符混淆: 2050 (45.0%)
   空格问题: 1350 (29.6%)
   字符丢失: 890 (19.5%)
   字符增加: 180 (3.9%)
   完全错误: 90 (2.0%)
```

---

## 🔄 convert_to_onnx.py 使用方法

### 依赖安装

```bash
pip install tf2onnx onnx onnxruntime
```

### 基础使用

```bash
# 转换默认模型
python convert_to_onnx.py

# 转换指定模型
python convert_to_onnx.py --model models/best_model.keras

# 转换GPU服务器模型
python convert_to_onnx.py --model /data/coding/caocrvfy/core/models/final_model.keras
```

### 高级选项

```bash
# 指定输出路径
python convert_to_onnx.py --model final_model.keras --output model.onnx

# 指定ONNX opset版本
python convert_to_onnx.py --model final_model.keras --opset 15

# 转换后测试推理
python convert_to_onnx.py --model final_model.keras --test
```

### ONNX Opset版本选择

| Opset | 特点 | 推荐场景 |
|-------|------|---------|
| 11 | 基础功能，兼容性最好 | 旧平台部署 |
| 13 | 平衡性能与兼容性 | **推荐默认** |
| 15 | 新特性，性能更好 | 新平台，追求性能 |
| 17+ | 最新特性 | 实验性质 |

### 输出内容

```
🔄 Keras → ONNX 模型转换
==================================================================

📥 加载Keras模型: core/models/final_model.keras
   ✓ Keras模型加载成功

📊 模型信息:
   输入形状: (None, 60, 200, 3)
   输出形状: (None, 504)
   参数量: 10,234,567

🔄 转换中... (opset=13)
   ✓ ONNX模型已保存: core/models/final_model.onnx

🔍 验证ONNX模型...
   ✓ ONNX模型验证通过

📦 文件大小对比:
   Keras: 336.37 MB
   ONNX:  338.12 MB
   差异:  +1.75 MB

✅ 转换成功！
```

---

## 📊 完整工作流程

### 场景1：本地测试GPU服务器训练的模型

```bash
# 1. 从GPU服务器下载模型（如果需要）
scp user@gpu-server:/data/coding/caocrvfy/core/models/final_model.keras ./models/

# 2. 测试模型
python test_model.py --model models/final_model.keras --analyze-errors

# 3. 查看详细错误
python test_model.py --model models/final_model.keras --only-errors --samples 100
```

### 场景2：转换模型用于生产部署

```bash
# 1. 测试模型性能
python test_model.py --model models/final_model.keras --report report.txt

# 2. 如果性能满意，转换为ONNX
python convert_to_onnx.py --model models/final_model.keras --test

# 3. ONNX模型已生成
# models/final_model.onnx
```

### 场景3：对比多个checkpoint性能

```bash
# 测试不同步数的checkpoint
python test_model.py --model models/checkpoint_step_145000.keras > step_145k.txt
python test_model.py --model models/checkpoint_step_148000.keras > step_148k.txt
python test_model.py --model models/checkpoint_step_150000.keras > step_150k.txt

# 对比完整匹配准确率
grep "完整匹配准确率" step_*.txt
```

---

## 🎯 使用场景

### test_model.py 适用于：
- ✅ 本地快速验证训练效果
- ✅ 分析模型错误类型
- ✅ 对比不同checkpoint性能
- ✅ 生成详细评估报告
- ✅ 展示给团队的演示

### convert_to_onnx.py 适用于：
- ✅ 跨平台部署（Windows/Linux/Mac）
- ✅ C++/Java等语言调用
- ✅ 移动端部署（需进一步转换）
- ✅ Web部署（ONNX.js）
- ✅ 优化推理性能

---

## 🔧 常见问题

### Q1: test_model.py报错"模型文件不存在"
**A**: 检查模型路径是否正确，使用绝对路径或相对于caocrvfy目录的路径

```bash
# 正确示例
python test_model.py --model core/models/final_model.keras
python test_model.py --model /data/coding/caocrvfy/core/models/final_model.keras
```

### Q2: convert_to_onnx.py报错"ModuleNotFoundError: No module named 'tf2onnx'"
**A**: 安装依赖

```bash
pip install tf2onnx onnx onnxruntime
```

### Q3: ONNX转换后文件变大
**A**: 正常现象，ONNX包含更多元数据用于跨平台兼容

### Q4: 想测试ONNX模型推理速度
**A**: 使用 `--test` 参数

```bash
python convert_to_onnx.py --model final_model.keras --test
```

### Q5: 如何在C++中使用ONNX模型？
**A**: 使用ONNX Runtime C++ API

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "captcha");
Ort::SessionOptions session_options;
Ort::Session session(env, "final_model.onnx", session_options);

// 准备输入数据
// 执行推理
// 处理输出
```

---

## 📈 性能基准

### test_model.py 性能
- 加载模型: ~2-3秒
- 评估20000样本: ~30-60秒（CPU）
- 评估20000样本: ~5-10秒（GPU）

### ONNX推理性能（单张图片）
- CPU (Intel i7): ~15-20ms
- GPU (RTX 3090): ~2-3ms
- ONNX优化后: 提升10-20%

---

## 💡 最佳实践

### 1. 定期测试模型
```bash
# 训练后立即测试
python test_model.py --model models/final_model.keras --analyze-errors

# 保存测试报告
python test_model.py --report reports/model_v1_$(date +%Y%m%d).txt
```

### 2. 对比不同训练策略
```bash
# baseline模型
python test_model.py --model models/baseline.keras > baseline_result.txt

# focal loss模型
python test_model.py --model models/focal_loss.keras > focal_result.txt

# 对比
diff baseline_result.txt focal_result.txt
```

### 3. 生产部署前检查
```bash
# 1. 完整测试
python test_model.py --model final_model.keras --analyze-errors --report report.txt

# 2. 转换ONNX并测试
python convert_to_onnx.py --model final_model.keras --test

# 3. 确认准确率满足要求后部署
```

---

## 📝 输出文件

### test_model.py 生成
- `evaluation_report.txt` - 评估报告（如果指定--report）

### convert_to_onnx.py 生成
- `*.onnx` - ONNX模型文件

---

**工具版本**: v1.0  
**更新日期**: 2026-01-31  
**适用模型**: Keras (.keras, .h5)
