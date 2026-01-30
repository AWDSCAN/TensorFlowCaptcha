卷积神经网络与TensorFlow识别复杂验证码的实战效果测试



**Part2 技术研究过程** 

- ### **基础设施准备**

经过不断地优化，以TensorFlow2.x为基础的验证码识别程序变成如下形式：引入了大量用户可控参数以备精准控制验证码识别过程；使用了卷积神经网络（CNN）来识别验证码，使用三层卷积 + 全连接层的经典CNN架构。框架选用**TensorFlow 2.x + Keras**，激活函数选用**ReLU**，池化层选用 **MaxPooling 2×2**，正则化选用 **Dropout(0.25)**，损失函数选用 **Sigmoid** 交叉熵，优化器选用 **Adam (lr=0.001)**。

![图片](https://mmbiz.qpic.cn/mmbiz_png/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXjzVM1ygpbhoBEickNxVibK16u92eXia4kArFgiaylt7FwaCjyHCcIjSOcQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)



其详细过程如下：

**1.** 输入验证码图像（200×50）。

**2.** [卷积层1] 32个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。

**3.** [卷积层2] 64个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。

**4.** [卷积层3] 64个3×3过滤器 → ReLU → MaxPool(2×2) → Dropout(0.25)。

**5.** [展平层] → 展平后得到 11200 维特征向量。

**6.** [全连接层] 1024个神经元 + Dropout。

**7.** [输出层] max_captcha × char_set_len 个神经元。

接下来运行命令：

- 

```
python tensorflow_cnn_train.py --charset digits --max_captcha 4 --batch_size 128 --target_accuracy 0.95
```



TensorFlow2.x 在验证码识别率达到95%之后，会生成4个文件，这其中就有识别验证码的模型。我们以一本书做比喻，对这4个文件的作用加深理解：

| 文件                       | 打比方                         |
| -------------------------- | ------------------------------ |
| crack_captcha_model.keras  | 完整的一本书（封面+目录+内容） |
| ckpt-1.data-00000-of-00001 | 只有正文内容的纯文本           |
| ckpt-1.index               | 这本书的目录指引               |
| checkpoint                 | 记录“当前看到书的第几章”       |



和先前DGA域名识别一样，首先选择一下算力环境。RTX 4090的GPU环境肯定更好，但是考虑性价比，还是退而求其次选择了RTX 3090。

![图片](https://mmbiz.qpic.cn/mmbiz_png/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXMG05kY3Oya1HEsLIJJwJRvcfDQef8ulibaBCvJAncV6KDXDX1c708Mw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

###  

- ### **卷积神经网络知识回顾**

卷积神经网络识别验证码/图像的流程如下，在上一篇文章做了详细讲解，这里不过多叙述。对于验证码识别，其输入层如下：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXKH6RicyZiajcTyaNQFCBxs9miaanP2YycgNvKSktTd3okcyANG5IRsPAQ/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)



从左到右，依次是输入层、卷积层、最大池化层、全连接层、输出层。

![图片](https://mmbiz.qpic.cn/mmbiz_png/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfX4T9Z2mV7Nc2ZyWNJ5oreS1RXic1W3GtOHCSRCuMfvLWeX5aTnXiaPF9g/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)



- ### **识别复杂验证码测试1**

首先拿出一个复杂的验证码样本做测试，大概有几万多个验证码图片样本。从图中可以看出，该类型验证码加入了数字+大写字母、字符旋转、字符大小变换、字符黏连、大量的干扰线、随机背景颜色、字符颜色变换等，如果通过编程算法来识别，难度非常大。接下来我们看一下卷积神经网络算法的识别效果。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXmT1gXxMWwKHMibg6f23u9ic4WCQIRZCp3pwjNUJjaSrNf3JibpNMOSccg/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)



如下图所示，**在经过10分钟零几秒的时间计算，验证码识别率就达到了95%**，比我们自己研究验证码识别算法效率要高的太多了。将程序训练过程中生成的数据制作成Accuracy–Step 曲线图，通过观察分析可以看出：

**1.** 当 Step 在 0–3000 步区间时：准确率基本处于极低水平，≈0.02 - 0.04，说明模型尚未有效学习或学习率过低。

**2.** 当 Step ≈3500 步开始：准确率出现明显跳跃式提升。

**3.** 当 Step 在 4000–5000区间时：准确率快速攀升，从 0.2 → 0.6 → 0.85。

**4.** 当 Step 在 5000 之后：准确率趋近于收敛，最终稳定在 ≈0.95 左右。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXalyaF8Pwn9bv6hPjnzcbMVoze9o6b6ZSGPicMnibLHoO1ibECVCVvibmWQ/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)



- ### **识别企鹅验证码测试2**

这是企鹅曾经使用过的验证码，这个验证码就很复杂了，最难的是加入了字体的扭曲。当然最复杂扭曲的是谷歌搜索引擎曾经使用的验证码，那个验证码太强了。对于这种带有扭曲的验证码，如果我们自己编写算法来识别非常困难。此外此套验证码加入了干扰图像（比干扰线更难）、颜色变换、字符的空心、字符的粗细变化，字符粘连等。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXPTMUvusf7LiaicHH7JdzbdYiantLU7FzymxeibC6jgicLnVkFZIyOnw8DEQ/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)



最后**TensorFlow 2.x 仅花了32分钟零几秒的时间，识别率就达到了95%，太强了**。接下来生成Accuracy 随 Step 变化的曲线图，从图中可以看出：

**1.** 当step 小于3000 时，验证码识别的准确率几乎徘徊在极低值，说明模型尚未有效进行机器学习。

**2.** 当step 在 3000 - 6000 时，准确率开始出现明显跃升。

**3.** 当step 在 6000–10000 时，曲线图进入平稳提升区间，模型逐渐趋近收敛状态。

**4.** 当step 在16000 之后准确率接近 0.93–0.96，接近收敛。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/OAz0RNU450ChfFbgjr5ahDVP3a2icuJfXQ1Pz8ZCINHgCrszHm5UyySeNWagkHRqtAf0ZctoOYSUYl8JYdIQdaA/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)



##  **Part3 总结** 

**1.** 技术总是在不断进步的，先前的验证码识别的一些编程、算法类的尝试，在CNN卷积神经网络面前，都变得意义不大。

**2.** 对于图像验证码的识别，TensorFlow2.x + 卷积神经网络CNN 就足够用了；而对于滑块验证码的识别，我们下期在做探讨。