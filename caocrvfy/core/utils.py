#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
功能：提供文件名解析、标签转换等通用工具函数
"""

import os
import base64
import numpy as np
from PIL import Image, ImageEnhance
from . import config

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: opencv-python未安装，图片预处理将使用基础模式。可通过 'pip install opencv-python' 安装获得更好效果")


def parse_filename(filename):
    """
    解析验证码文件名，提取验证码文本
    
    支持两种格式:
    1. 普通格式: 验证码内容-32位hash.png (如 "abc123-f3b1c8e8adeaeae20f26913b53bbc9d8.png")
    2. 数学题格式: base64(题目)_答案_16位hash.png (如 "MTkrMz0/_22_abc123def456.png")
    
    参数:
        filename: 文件名
    
    返回:
        验证码文本
        - 普通类型: 返回文本内容（如 "abc123"）
        - 数学题类型: 返回解码后的题目（如 "19+3=?"）
    """
    # 去除扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 检查是否为数学题格式（包含下划线且有3部分）
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        if len(parts) == 3:
            # 数学题格式: base64_answer_hash
            try:
                # 尝试base64解码第一部分
                base64_text = parts[0]
                decoded_text = base64.b64decode(base64_text.encode('utf-8')).decode('utf-8')
                return decoded_text
            except Exception as e:
                # 如果解码失败，可能是普通格式的文件名包含下划线，继续尝试普通解析
                print(f"警告: base64解码失败 {filename}: {e}")
                pass
    
    # 普通格式: 使用'-'分割
    captcha_text = name_without_ext.split('-')[0]
    return captcha_text


def text_to_vector(text):
    """
    将验证码文本转换为one-hot编码向量
    
    参数:
        text: 验证码文本，如 "abc123"
    
    返回:
        numpy数组，形状为 (MAX_CAPTCHA × CHAR_SET_LEN,)
    """
    vector = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN, dtype=np.float32)
    
    # 将文本填充到MAX_CAPTCHA长度（短验证码用空格填充）
    padded_text = text.ljust(config.MAX_CAPTCHA, config.PADDING_CHAR)
    
    for i, char in enumerate(padded_text[:config.MAX_CAPTCHA]):
        # 查找字符在字符集中的索引
        if char in config.CHAR_SET:
            char_idx = config.CHAR_SET.index(char)
            # 设置对应位置为1（one-hot编码）
            vector[i * config.CHAR_SET_LEN + char_idx] = 1.0
    
    return vector


def vector_to_text(vector, threshold=0.5):
    """
    将one-hot编码向量转换回验证码文本
    
    参数:
        vector: numpy数组，形状为 (MAX_CAPTCHA × CHAR_SET_LEN,) 或 (MAX_CAPTCHA, CHAR_SET_LEN)
        threshold: 置信度阈值，低于此值的预测将被忽略
    
    返回:
        验证码文本字符串
    """
    # 确保向量是一维的
    if len(vector.shape) == 2:
        vector = vector.flatten()
    
    # reshape 为 (MAX_CAPTCHA, CHAR_SET_LEN)
    vector = vector.reshape((config.MAX_CAPTCHA, config.CHAR_SET_LEN))
    
    text = []
    for i in range(config.MAX_CAPTCHA):
        # 找到每个位置概率最大的字符
        max_prob = np.max(vector[i])
        char_idx = np.argmax(vector[i])
        
        # 只有置信度超过阈值才使用，否则用填充字符
        if max_prob >= threshold:
            char = config.CHAR_SET[char_idx]
        else:
            char = config.PADDING_CHAR  # 低置信度用空格
        
        text.append(char)
    
    # 去除尾部的填充字符（空格）
    result = ''.join(text).rstrip(config.PADDING_CHAR)
    return result


def preprocess_captcha_with_cv2(img):
    """
    使用OpenCV进行验证码预处理：去除干扰线和噪点
    
    处理步骤:
    1. 灰度化
    2. CLAHE对比度增强
    3. 自适应阈值二值化
    4. 形态学开运算去噪
    5. 转回RGB格式
    
    参数:
        img: PIL.Image对象
    
    返回:
        PIL.Image对象（预处理后）
    """
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 1. 转为灰度图
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. CLAHE对比度增强（拉伸字符与背景的差异）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 自适应阈值二值化（去除背景和干扰线）
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # 4. 形态学操作：开运算去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. 转回RGB格式（复制3通道）
    rgb_preprocessed = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb_preprocessed)


def preprocess_captcha_with_pil(img):
    """
    使用PIL进行基础验证码预处理（当OpenCV不可用时）
    
    处理步骤:
    1. 转为灰度
    2. 增强对比度
    3. 锐化
    4. 转回RGB
    
    参数:
        img: PIL.Image对象
    
    返回:
        PIL.Image对象（预处理后）
    """
    # 1. 转为灰度
    gray = img.convert('L')
    
    # 2. 增强对比度
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    
    # 3. 锐化
    from PIL import ImageFilter
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    
    # 4. 转回RGB
    rgb = sharpened.convert('RGB')
    
    return rgb


def load_image(image_path, use_preprocessing=True):
    """
    加载并预处理验证码图像
    
    参数:
        image_path: 图像文件路径
        use_preprocessing: 是否使用预处理去除干扰（默认True）
    
    返回:
        numpy数组，形状为 (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        值范围 [0, 1]
    """
    # 加载图像
    img = Image.open(image_path)
    
    # 确保图像是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 去干扰预处理
    if use_preprocessing:
        if CV2_AVAILABLE:
            img = preprocess_captcha_with_cv2(img)
        else:
            img = preprocess_captcha_with_pil(img)
    
    # 调整图像尺寸
    img = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组并归一化到[0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def calculate_accuracy(y_true, y_pred):
    """
    计算验证码识别准确率（完全匹配）
    
    参数:
        y_true: 真实标签（文本列表或向量数组）
        y_pred: 预测标签（文本列表或向量数组）
    
    返回:
        准确率（0-1之间的浮点数）
    """
    # 如果是列表，直接处理
    if isinstance(y_true, (list, tuple)):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if len(y_true) > 0 else 0.0
    
    # 如果是numpy数组
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(1, -1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)
    
    batch_size = y_true.shape[0]
    correct = 0
    
    for i in range(batch_size):
        true_text = vector_to_text(y_true[i])
        pred_text = vector_to_text(y_pred[i])
        
        if true_text == pred_text:
            correct += 1
    
    return correct / batch_size


def get_char_position_accuracy(y_true, y_pred):
    """
    计算每个字符位置的准确率
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    
    返回:
        字典，包含每个位置的准确率
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(1, -1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)
    
    batch_size = y_true.shape[0]
    position_correct = np.zeros(config.MAX_CAPTCHA)
    
    for i in range(batch_size):
        true_vector = y_true[i].reshape((config.MAX_CAPTCHA, config.CHAR_SET_LEN))
        pred_vector = y_pred[i].reshape((config.MAX_CAPTCHA, config.CHAR_SET_LEN))
        
        for pos in range(config.MAX_CAPTCHA):
            true_idx = np.argmax(true_vector[pos])
            pred_idx = np.argmax(pred_vector[pos])
            
            if true_idx == pred_idx:
                position_correct[pos] += 1
    
    position_accuracy = position_correct / batch_size
    
    return {f'position_{i+1}': acc for i, acc in enumerate(position_accuracy)}


# 测试工具函数
if __name__ == '__main__':
    print("测试工具函数...")
    print("=" * 80)
    
    # 测试文件名解析
    test_filename = "abc123-f3b1c8e8adeaeae20f26913b53bbc9d8.png"
    text = parse_filename(test_filename)
    print(f"文件名: {test_filename}")
    print(f"解析结果: {text}")
    print()
    
    # 测试文本到向量转换
    print(f"文本: {text}")
    vector = text_to_vector(text)
    print(f"向量形状: {vector.shape}")
    print(f"向量中非零元素数量: {np.sum(vector)}")
    print()
    
    # 测试向量到文本转换
    reconstructed_text = vector_to_text(vector)
    print(f"重建文本: {reconstructed_text}")
    print(f"转换是否正确: {text == reconstructed_text}")
    print()
    
    print("=" * 80)
    print("✓ 工具函数测试完成")
