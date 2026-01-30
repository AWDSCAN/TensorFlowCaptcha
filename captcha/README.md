# 验证码生成器

## 功能特性

- ✅ **多类型支持**：纯数字、纯字母、数字+字母混合、数学算术题
- ✅ **字符集丰富**：数字(0-9) + 大写字母(A-Z) + 小写字母(a-z)
- ✅ **不定长验证码**：支持4-8位随机长度
- ✅ **智能干扰**：
  - 普通验证码：三层干扰系统（底层+中层穿透+顶层覆盖）
  - 数学题：无干扰线，保持简洁便于识别
- ✅ **字符变换**：随机旋转(-30°~30°)、颜色变化
- ✅ **智能命名**：验证码内容-hash.png（基于时间戳+内容的MD5）
- ✅ **跨平台**：自动识别Windows/Linux系统字体

## 环境要求

```bash
# 激活conda环境（Python 3.10.x）
conda activate TensorFlow

# 安装依赖
pip install Pillow numpy captcha
```

## 快速使用

### 直接运行

```bash
# 生成测试验证码（每种类型3张）
python generate_captcha.py

# 图片自动保存在 captcha/img 目录
```

### Python代码使用

```python
from generate_captcha import CaptchaGenerator

# 创建生成器
gen = CaptchaGenerator(captcha_type='mixed')  # digit, alpha, mixed, math

# 生成一张验证码
image, text, answer, filename = gen.generate_captcha(save_path='img')
print(f"文件: {filename}, 内容: {text}")

# 批量生成
for i in range(10):
    gen.generate_captcha(save_path='img')
```

## 验证码类型

| 类型 | 参数 | 说明 | 示例 | 干扰效果 |
|------|------|------|------|---------|
| 纯数字 | `digit` | 0-9随机组合 | 123456 | ✅ 强干扰 |
| 纯字母 | `alpha` | 大小写字母混合 | AbCdEf | ✅ 强干扰 |
| 混合模式 | `mixed` | 数字+字母组合 | a1B2c3 | ✅ 强干扰 |
| 数学题 | `math` | 加减乘运算 | 3+5=? | ❌ 无干扰（便于识别）|

## 干扰系统说明

### 普通验证码（digit/alpha/mixed）

采用三层干扰系统：

```
第一层：底层干扰（背景）
├─ 6-10条随机线
└─ 1000-1500个噪点

第二层：中间层干扰（穿透字符）
├─ 4-7条穿过验证码中心的粗线
└─ 显著增加机器识别难度

第三层：顶层干扰（覆盖）
├─ 3-6条覆盖线
├─ 2-4条随机弧线
└─ 40%概率高斯模糊
```

### 数学题验证码（math）

**无干扰设计**：为了确保用户能清晰看到算术题，数学题类型不添加任何干扰线和噪点，只保留：
- 清晰的背景
- 字符随机旋转
- 基本的字符颜色变化

## 文件命名规则

格式：`验证码内容-hash.png`

- **验证码内容**：实际的验证码字符（数学题使用答案）
- **hash**：32位MD5哈希值（时间戳+内容+随机数），确保唯一性

示例：
```
普通验证码: abc123-f3b1c8e8adeaeae20f26913b53bbc9d8.png
数学题: 8-191651f9c522daec3ec256f15787e0f0.png  (问题是3+5=?，答案是8)
```

**优势**：32位完整哈希 + 随机数，即使生成数百万张图片也不会出现命名冲突。

## API文档

### CaptchaGenerator 类

```python
class CaptchaGenerator:
    def __init__(self, width=200, height=60, mode='pil', captcha_type='mixed'):
        """
        初始化验证码生成器
        
        参数:
            width (int): 宽度，默认200
            height (int): 高度，默认60
            mode (str): 生成模式
                - 'pil': PIL自定义绘制（推荐）
                - 'captcha': captcha库生成
            captcha_type (str): 验证码类型
                - 'digit': 纯数字
                - 'alpha': 纯字母
                - 'mixed': 数字+字母混合（默认）
                - 'math': 数学算术题（无干扰）
        """
    
    def generate_captcha(self, text=None, save_path=None):
        """
        生成验证码
        
        参数:
            text (str, optional): 指定验证码内容，None为随机生成
            save_path (str, optional): 保存路径（文件或目录）
        
        返回:
            tuple: (image, text, answer, filename)
                - image: PIL Image对象
                - text: 验证码显示的文本
                - answer: 验证码答案
                - filename: 生成的文件名
        """
```

## 目录结构

```
captcha/
├── generate_captcha.py     # 主程序（唯一代码文件）
├── README.md               # 使用文档
└── img/                    # 输出目录
    └── *.png              # 生成的验证码图片
```

## 使用示例

### 示例1: 生成纯数字验证码

```python
from generate_captcha import CaptchaGenerator

gen = CaptchaGenerator(captcha_type='digit')
for i in range(100):
    image, text, answer, filename = gen.generate_captcha(save_path='img')
    print(f"生成: {filename} -> {text}")
```

### 示例2: 生成数学题（无干扰）

```python
from generate_captcha import CaptchaGenerator

gen = CaptchaGenerator(captcha_type='math')
labels = []
for i in range(100):
    image, text, answer, filename = gen.generate_captcha(save_path='img')
    labels.append({
        'filename': filename,
        'question': text,
        'answer': answer
    })
    print(f"{filename}: {text} = {answer}")

# 保存标签
import json
with open('img/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)
```

### 示例3: 自定义尺寸

```python
from generate_captcha import CaptchaGenerator

# 生成更大的验证码
gen = CaptchaGenerator(width=300, height=80, captcha_type='mixed')
gen.generate_captcha(save_path='img')
```

## 常见问题

**Q: 为什么数学题没有干扰线？**  
A: 为了确保用户能清晰识别算术题目，数学题类型特意设计为无干扰，便于人眼识别。

**Q: 图片保存在哪里？**  
A: 默认保存在 `captcha/img` 目录下，也可以通过参数指定其他目录。

**Q: 如何调整验证码长度？**  
A: 修改 `get_random_text(min_len=4, max_len=8)` 的参数。

**Q: 如何修改字符集？**  
A: 修改类中的 `self.charset` 变量。

**Q: captcha库是必需的吗？**  
A: 不是必需的。如果未安装，程序会自动使用PIL模式，功能完全不受影响。

## 技术栈

- **Python**: 3.10.x
- **PIL/Pillow**: 图像生成和处理
- **NumPy**: 数值计算（可选）
- **Captcha**: 可选的captcha库支持
- **环境**: Conda TensorFlow

## 更新日志

### v3.2.0 (2026-01-30)
- ✅ 优化文件命名：hash长度从12位增加到32位
- ✅ 增加随机数因子，确保大批量生成时无命名冲突
- ✅ 支持数百万级别图片生成而不重复

### v3.1.0 (2026-01-30)
- ✅ 优化目录结构，清理冗余文件
- ✅ 数学题类型取消干扰线（保持简洁）
- ✅ 统一图片保存路径为 captcha/img
- ✅ 精简代码，保留核心功能
- ✅ 更新文档，更加简洁清晰

### v3.0.0 (2026-01-30)
- 新增4种验证码类型支持
- 增强PIL模式三层干扰系统
- 优化文件命名规则

### v2.0.0 (2026-01-30)
- 双模式支持（PIL + Captcha）
- 智能文件命名
- 跨平台字体识别

### v1.0.0
- 基础验证码生成功能

