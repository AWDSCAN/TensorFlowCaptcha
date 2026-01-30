#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
功能：提供文件名解析、标签转换等通用工具函数
"""

import os
import numpy as np
from PIL import Image
import config


def parse_filename(filename):
    """
    解析验证码文件名，提取验证码文本
    文件名格式: 验证码内容-32位hash.png
    
    参数:
        filename: 文件名，如 "abc123-f3b1c8e8adeaeae20f26913b53bbc9d8.png"
    
    返回:
        验证码文本，如 "abc123"
    """
    # 去除扩展名
    name_without_ext = os.path.splitext(filename)[0]
    # 分割获取验证码内容（hash前面的部分）
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
    
    for i, char in enumerate(text):
        if i >= config.MAX_CAPTCHA:
            break
        
        # 查找字符在字符集中的索引
        if char in config.CHAR_SET:
            char_idx = config.CHAR_SET.index(char)
            # 设置对应位置为1（one-hot编码）
            vector[i * config.CHAR_SET_LEN + char_idx] = 1.0
    
    return vector


def vector_to_text(vector):
    """
    将one-hot编码向量转换回验证码文本
    
    参数:
        vector: numpy数组，形状为 (MAX_CAPTCHA × CHAR_SET_LEN,) 或 (MAX_CAPTCHA, CHAR_SET_LEN)
    
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
        # 找到每个位置概率最大的字符索引
        char_idx = np.argmax(vector[i])
        # 如果该位置的最大概率很小，说明该位置没有字符
        if vector[i][char_idx] < 0.5:
            continue
        char = config.CHAR_SET[char_idx]
        text.append(char)
    
    return ''.join(text)


def load_image(image_path):
    """
    加载并预处理验证码图像
    
    参数:
        image_path: 图像文件路径
    
    返回:
        numpy数组，形状为 (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        值范围 [0, 1]
    """
    # 加载图像
    img = Image.open(image_path)
    
    # 确保图像是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 调整图像尺寸
    img = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组并归一化到[0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def calculate_accuracy(y_true, y_pred):
    """
    计算验证码识别准确率（完全匹配）
    
    参数:
        y_true: 真实标签，形状 (batch_size, MAX_CAPTCHA × CHAR_SET_LEN)
        y_pred: 预测标签，形状 (batch_size, MAX_CAPTCHA × CHAR_SET_LEN)
    
    返回:
        准确率（0-1之间的浮点数）
    """
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
