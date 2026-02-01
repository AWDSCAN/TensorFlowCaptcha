# 数学题验证码命名方式更新（2026-02-01）

## 📋 更新概述

修改了数学题验证码的文件命名方式，从"答案-hash.png"改为"base64(题目)_答案_hash.png"，确保训练时图片内容与标签匹配。

---

## 🔴 原问题

### 旧命名方式的缺陷
```
文件名: 22-abc123def456.png
图片内容: "19+3=?"
训练标签: "22" (从文件名解析)

问题: 图片显示 "19+3=?" 但标签是 "22" → 完全不匹配！
后果: 模型无法学习，25%数据浪费
```

---

## ✅ 新解决方案

### 新命名方式
```
文件名: MTkrMz0/_22_abc123def456.png
       ↑        ↑   ↑
       |        |   └─ 16位hash
       |        └───── 答案 (22)
       └────────────── base64编码的题目 (19+3=?)

图片内容: "19+3=?"
训练标签: "19+3=?" (从base64部分解析)

优势: 图片和标签完全匹配！✓
```

### 命名格式说明
- **数学题**: `base64(题目)_答案_16位hash.png`
  - base64部分: 题目内容编码（避免文件名特殊字符问题）
  - 答案部分: 数学运算结果（可选，用于验证）
  - hash部分: 唯一标识（缩短为16位）

- **普通类型**: `内容-32位hash.png` (保持不变)
  - 内容部分: 验证码文本
  - hash部分: 唯一标识

---

## 🔧 代码修改

### 1. captcha/generate_captcha.py

#### 添加base64支持
```python
import base64  # 新增

# 更新文档说明
"""
数学题命名格式: base64(数学运算题)_运算结果_随机hash.png
例如: MTkrMz0/_22_abc123def456.png 表示 "19+3=?" 答案是 22
"""
```

#### 修改文件名生成逻辑
```python
def generate_filename(self, text, answer=None):
    """
    生成文件名
    
    普通类型格式：验证码内容-32位hash.png
    数学题格式：base64(题目)_答案_16位hash.png
    """
    if self.captcha_type == 'math' and answer is not None:
        # 数学题：base64编码题目_答案_hash
        text_base64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        file_hash = self.generate_hash(text + str(answer))[:16]
        return f"{text_base64}_{answer}_{file_hash}.png"
    else:
        # 普通类型：原有格式
        file_hash = self.generate_hash(text)
        return f"{text}-{file_hash}.png"
```

#### 更新调用逻辑
```python
def generate_captcha(self, text=None, save_path=None):
    # ... 生成图片 ...
    
    # 生成文件名
    if self.captcha_type == 'math':
        # 数学题：使用base64(题目)_答案_hash格式
        filename = self.generate_filename(text, answer)
    else:
        # 普通类型：使用text-hash格式
        filename = self.generate_filename(text)
```

### 2. caocrvfy/core/utils.py

#### 添加base64解析
```python
import base64  # 新增

def parse_filename(filename):
    """
    解析验证码文件名，提取验证码文本
    
    支持两种格式:
    1. 普通格式: 验证码内容-32位hash.png
    2. 数学题格式: base64(题目)_答案_16位hash.png
    
    返回:
        - 普通类型: 返回文本内容（如 "abc123"）
        - 数学题类型: 返回解码后的题目（如 "19+3=?"）
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # 检查是否为数学题格式（包含下划线且有3部分）
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        if len(parts) == 3:
            try:
                # base64解码第一部分
                base64_text = parts[0]
                decoded_text = base64.b64decode(base64_text.encode('utf-8')).decode('utf-8')
                return decoded_text
            except Exception as e:
                print(f"警告: base64解码失败 {filename}: {e}")
                pass
    
    # 普通格式: 使用'-'分割
    captcha_text = name_without_ext.split('-')[0]
    return captcha_text
```

### 3. caocrvfy/core/config.py

#### 扩展字符集支持数学运算符
```python
# 数学运算符（用于数学题类型）
MATH_OPERATORS = '+-*=?'  # 加、减、乘、等号、问号

# 完整字符集（数字+大小写字母+填充字符+数学运算符）
CHAR_SET = DIGITS + ALPHA_ALL + PADDING_CHAR + MATH_OPERATORS

# 字符集大小
CHAR_SET_LEN = len(CHAR_SET)  # 68个字符 (原63 + 5个运算符)

# 验证码最大长度
MAX_CAPTCHA = 8  # 支持1-8位（数学题如"19+3=?"是6位）
```

---

## 🧪 测试验证

### 测试脚本1: test_math_naming.py

验证命名和解析逻辑：
```bash
python test_math_naming.py
```

**输出示例**:
```
测试1: 生成数学题验证码
第 1 个验证码:
  题目: 3+4=?
  答案: 7
  文件名: Mys0PT8=_7_02c7d8b5d92d8066.png
    ✓ 格式正确: base64=Mys0PT8=... / 答案=7
    ✓ base64解码正确: 3+4=?

测试2: 解析文件名
文件名: Mys0PT8=_7_02c7d8b5d92d8066.png
  原始题目: 3+4=?
  解析结果: 3+4=?
  ✓ 解析正确

字符集测试:
  ✓ '+' 在字符集中 (索引: 63)
  ✓ '-' 在字符集中 (索引: 64)
  ✓ '*' 在字符集中 (索引: 65)
  ✓ '=' 在字符集中 (索引: 66)
  ✓ '?' 在字符集中 (索引: 67)
```

### 测试脚本2: demo_math_training.py

演示完整训练流程：
```bash
python demo_math_training.py
```

**输出要点**:
```
步骤 2: 分析数据类型
✓ 数学题类型: 3 张
✓ 普通类型: 9 张

数学题样本:
  • 文件名: Myo5PT8=_27_9c904175eba40e39.png
    标签: 3*9=?
    ✓ 所有字符都在字符集中

步骤 3: 标签向量化
示例: 2*4=?
向量维度: (544,) = (8 × 68)
字符编码:
  位置 0: '2' → 字符集索引 2
  位置 1: '*' → 字符集索引 65
  位置 2: '4' → 字符集索引 4
  位置 3: '=' → 字符集索引 66
  位置 4: '?' → 字符集索引 67
✓ 向量化和反向量化正确
```

### 测试脚本3: verify_training_data.py

验证数据加载：
```bash
python verify_training_data.py
```

**输出**:
```
总样本数: 12
短标签数量: 0 (0.0%)
✓ 未发现短标签，数据集正常
```

---

## 📊 命名格式对比

| 类型 | 旧格式 | 新格式 | 优势 |
|------|--------|--------|------|
| **数学题** | `22-hash.png` | `MTkrMz0/_22_hash.png` | ✅ 文件名包含题目和答案 |
| **普通** | `abc123-hash.png` | `abc123-hash.png` | - 保持不变 |

### 解析示例

#### 数学题文件名
```
文件名: MTkrMz0/_22_abc123def456.png

解析步骤:
1. 去除扩展名: MTkrMz0/_22_abc123def456
2. 按'_'分割: ['MTkrMz0/', '22', 'abc123def456']
3. base64解码第1部分: MTkrMz0/ → "19+3=?"
4. 第2部分是答案: 22
5. 第3部分是hash: abc123def456

训练标签: "19+3=?"
```

#### 普通文件名
```
文件名: abc123-xyz789.png

解析步骤:
1. 去除扩展名: abc123-xyz789
2. 按'-'分割: ['abc123', 'xyz789']
3. 第1部分是内容: abc123

训练标签: "abc123"
```

---

## 🎯 训练效果

### 训练目标变化

**旧方式（错误）**:
```
输入: 图片显示 "19+3=?"
标签: "22"
问题: 图片和标签不匹配 → 无法学习
```

**新方式（正确）**:
```
输入: 图片显示 "19+3=?"
标签: "19+3=?"
目标: 识别数学运算题本身 → 可以正常学习
```

### 模型能力

训练后模型可以：
1. ✅ 识别数学题的完整内容（包括数字和运算符）
2. ✅ 作为OCR使用，识别任意数学表达式
3. ✅ 后续可通过`eval()`计算答案
4. ✅ 与普通验证码识别统一架构

### 预期效果

| 指标 | 旧方式 | 新方式 | 改进 |
|------|--------|--------|------|
| 数学题准确率 | ~0% | 85-90% | +85-90% |
| 整体准确率 | 78% | 90-95% | +12-17% |
| 数据利用率 | 75% | 100% | +25% |

---

## 📝 使用说明

### 生成验证码

```python
from captcha.generate_captcha import CaptchaGenerator

# 生成数学题验证码
generator = CaptchaGenerator(captcha_type='math')
image, text, answer, filename = generator.generate_captcha()

print(f"题目: {text}")        # "19+3=?"
print(f"答案: {answer}")      # "22"
print(f"文件名: {filename}")  # "MTkrMz0/_22_hash.png"
```

### 训练模型

```python
from caocrvfy.core.data_loader import CaptchaDataLoader
from caocrvfy.core import utils

# 加载数据
loader = CaptchaDataLoader()
loader.load_data()

# 解析文件名（自动处理新格式）
for image_path, label_text in zip(loader.image_paths, loader.labels):
    filename = os.path.basename(image_path)
    parsed_text = utils.parse_filename(filename)
    
    # parsed_text 会自动从base64解码（如果是数学题）
    # 或直接返回文本（如果是普通类型）
```

### 批量生成

```bash
# 生成包含数学题的完整数据集
cd captcha
python generate_captcha.py --count 20000

# 数学题会自动使用新命名格式
```

---

## 🚀 部署步骤

### 本地环境

```bash
# 1. 测试新命名方式
python test_math_naming.py

# 2. 演示训练流程
python demo_math_training.py

# 3. 生成训练数据
cd captcha
python generate_captcha.py --count 20000

# 4. 验证数据质量
cd ..
python verify_training_data.py

# 5. 开始训练
cd caocrvfy
python train_v4.py
```

### GPU服务器

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 生成训练数据
cd captcha
python generate_captcha.py --count 20000

# 3. 验证数据
cd ..
python verify_training_data.py

# 4. 开始训练
cd caocrvfy
tmux new -s training_math
python train_v4.py
```

---

## ⚠️ 重要注意事项

### 1. 向后兼容性
- ✅ 新代码兼容旧的普通格式文件名
- ✅ `parse_filename()`自动识别格式类型
- ⚠️ 旧的数学题文件名无法正确解析（需要重新生成）

### 2. 字符集更新
- 原字符集: 63个（数字+字母+空格）
- 新字符集: 68个（+5个运算符 `+-*=?`）
- 影响: 输出维度从 `8×63=504` 变为 `8×68=544`
- ⚠️ 需要重新训练模型（旧模型不兼容）

### 3. 数据集管理
- 旧数据集包含错误的数学题标签，建议删除
- 使用新代码重新生成完整数据集
- 备份命令: `mv captcha/img captcha/img_old`

### 4. 训练建议

#### 选项A: 统一训练（推荐）
```bash
# 生成包含所有类型的数据集
python generate_captcha.py --count 20000

# 训练一个通用模型
python train_v4.py
```

**优势**: 一个模型识别所有类型  
**劣势**: 可能需要更多训练时间

#### 选项B: 分开训练
```bash
# 生成纯数学题数据集
python generate_captcha.py --count 5000 --types math

# 训练数学题专用模型
python train_v4.py --model-name math_model

# 生成普通验证码数据集
python generate_captcha_fixed.py --count 15000

# 训练普通验证码模型
python train_v4.py --model-name normal_model
```

**优势**: 每个模型专注一种类型，可能准确率更高  
**劣势**: 需要维护两个模型

---

## 📈 预期改进

| 方面 | 改进 | 说明 |
|------|------|------|
| 数据利用率 | 100% | 所有数据都可正确训练 |
| 数学题准确率 | 85-90% | 从接近0%提升 |
| 整体准确率 | 90-95% | 从78%提升 |
| 模型能力 | ✓ OCR | 可识别数学表达式 |
| 扩展性 | ✓ 更多符号 | 可添加更多运算符 |

---

## 🔄 后续优化

### 短期
1. ✅ 完成新命名方式实现
2. ✅ 更新字符集支持运算符
3. ✅ 创建测试脚本验证
4. ⏳ 重新生成训练数据集
5. ⏳ 训练并评估新模型

### 中期
1. 优化数学题图片生成（去除干扰，便于识别）
2. 增加更多运算符支持（/, ^, (, )等）
3. 支持更复杂的数学表达式
4. 分离数学题和普通验证码模型

### 长期
1. 支持多行数学题
2. 支持分数、根号等复杂符号
3. 集成计算引擎（自动验证答案）
4. Web API接口

---

## 📚 相关文档

- [训练瓶颈分析报告](TRAINING_BOTTLENECK_ANALYSIS_2026-02-01.md)
- [训练修复指南](TRAINING_FIX_GUIDE.md)
- [模型测试指南](MODEL_TESTING_GUIDE.md)

---

**更新日期**: 2026-02-01  
**版本**: v2.0  
**状态**: ✅ 已完成并测试
