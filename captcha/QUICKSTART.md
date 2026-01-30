# 验证码生成器 - 快速开始

## ⚡ 5秒开始

```bash
# 1. 激活环境
conda activate TensorFlow

# 2. 生成验证码
cd C:\Users\admin\Documents\company\CompanyToolDevelopment\tensorflow_cnn_captcha\captcha
python generate_captcha.py

# 3. 查看结果
# 图片保存在: captcha/img/*.png
```

## 📋 验证码类型

生成的12张图片包括：

1. **纯数字** (3张) - 带强干扰线
2. **纯字母** (3张) - 带强干扰线  
3. **混合模式** (3张) - 带强干扰线
4. **数学题** (3张) - 无干扰线（清晰易读）

## 🎨 干扰效果对比

| 类型 | 干扰线 | 噪点 | 模糊 | 用途 |
|------|--------|------|------|------|
| 纯数字/字母/混合 | ✅ 13-23条 | ✅ 1000+ | ✅ 40% | 深度学习训练 |
| 数学题 | ❌ 无 | ❌ 无 | ❌ 无 | 人机验证 |

## 💻 Python API

```python
from generate_captcha import CaptchaGenerator

# 生成纯数字（带干扰）
gen = CaptchaGenerator(captcha_type='digit')
img, text, ans, file = gen.generate_captcha(save_path='img')

# 生成数学题（无干扰）
gen = CaptchaGenerator(captcha_type='math')
img, text, ans, file = gen.generate_captcha(save_path='img')
# text: "3+5=?"
# ans: "8"
```

## 📁 目录结构

```
captcha/
├── generate_captcha.py   # 主程序（唯一代码文件）
├── README.md             # 完整文档
└── img/                  # 输出目录
    └── *.png            # 验证码图片
```

**简洁清爽，一个文件搞定所有功能！** ✨
