---
name: TensorFlow2-skills
description: TensorFlow2 ruler
---

# TensorFlow2 ruler

以TensorFlow2.16.1为基础的验证码识别程序变成如下形式：引入了大量用户可控参数以备精准控制验证码识别过程；使用了卷积神经网络（CNN）来识别验证码，使用三层卷积 + 全连接层的经典CNN架构。框架选用TensorFlow 2.x + Keras，激活函数选用ReLU，池化层选用 MaxPooling 2×2，正则化选用 Dropout(0.25)，损失函数选用 Sigmoid 交叉熵，优化器选用 Adam (lr=0.001)。
其详细过程如下：1.  输入验证码图像（200×50）。
2.  [卷积层1] 32个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。
3.  [卷积层2] 64个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。
4.  [卷积层3] 64个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。
5.  [展平层] → 展平后得到 11200 维特征向量。
6.  [全连接层] 1024个神经元 + Dropout。
7.  [输出层] max_captcha × char_set_len 个神经元。
TensorFlow2.x 在验证码识别率达到95%之后，会生成4个文件，这其中就有识别验证码的模型。我们以一本书做比喻，对这4个文件的作用加深理解：文件打比方crack_captcha_model.keras完整的一本书（封面+目录+内容）ckpt-1.data-00000-of-00001只有正文内容的纯文本ckpt-1.index这本书的目录指引checkpoint记录“当前看到书的第几章”
# 验证码类型
验证码类型包括数字+大写字母、字符旋转、字符大小变换、字符黏连、大量的干扰线、随机背景颜色、字符颜色变换、字体扭曲等