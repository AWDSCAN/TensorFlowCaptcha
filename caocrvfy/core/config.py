#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件 - 验证码识别模型参数配置
功能：统一管理所有超参数和路径配置
"""

import os
import string

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 验证码图片目录
# CAPTCHA_DIR = os.path.join(PROJECT_ROOT, 'captcha', 'img')
CAPTCHA_DIR = '/data/coding/captcha/img'
# 模型保存目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# 日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# ==================== 图像参数 ====================
# 验证码图像尺寸（与 generate_captcha.py 保持一致）
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
IMAGE_CHANNELS = 3  # RGB彩色图像

# ==================== 验证码字符集 ====================
# 支持的字符集（与 generate_captcha.py 保持一致）
DIGITS = string.digits  # 0-9
ALPHA_UPPER = string.ascii_uppercase  # A-Z
ALPHA_LOWER = string.ascii_lowercase  # a-z
ALPHA_ALL = string.ascii_letters  # A-Z + a-z

# 数学运算符（用于数学题类型）
MATH_OPERATORS = '+-*=?'  # 加、减、乘、等号、问号

# 完整字符集（数字+大小写字母+填充字符+数学运算符）
PADDING_CHAR = ' '  # 使用空格作为填充字符
CHAR_SET = DIGITS + ALPHA_ALL + PADDING_CHAR + MATH_OPERATORS  # 0-9 + A-Z + a-z + ' ' + +-*=?

# 字符集大小
CHAR_SET_LEN = len(CHAR_SET)  # 68个字符 (原63 + 5个运算符)

# 验证码最大长度
MAX_CAPTCHA = 8  # 支持1-8位不定长验证码（数学题如"19+3=?"是6位）

# ==================== 模型架构参数 ====================
# 卷积层配置（三层卷积架构）
CONV_FILTERS = [32, 64, 64]  # 每层的过滤器数量
CONV_KERNEL_SIZE = (3, 3)  # 卷积核大小
POOL_SIZE = (2, 2)  # 池化层大小
DROPOUT_CONV = 0.25  # 卷积层Dropout（增强正则化，减少过拟合）

# 全连接层配置
FC_UNITS = 2048  # 全连接层神经元数量（增加至2048，提升表达能力）
DROPOUT_FC = 0.5  # 全连接层Dropout（提高至0.5，参考trains.py）

# 输出层配置
OUTPUT_SIZE = MAX_CAPTCHA * CHAR_SET_LEN  # 8 × 63 = 504

# ==================== 训练参数 ====================
# 批次大小
BATCH_SIZE = 128  # 充分利用GPU内存

# 训练轮数
EPOCHS = 300  # 足够的训练轮数

# 学习率配置
LEARNING_RATE = 0.0008  # 初始学习率（降至0.0008，更精细的优化）
WARMUP_EPOCHS = 15  # Warmup轮数（增加至15，更平滑的启动）
WARMUP_LR_START = 0.00005  # Warmup起始学习率（更小的起点）
LR_DECAY_FACTOR = 0.6  # 学习率衰减因子（0.6更平滑）
LR_DECAY_PATIENCE = 12  # 学习率衰减耐心值（增加至12，更稳定）

# 验证集比例
VALIDATION_SPLIT = 0.2

# 早停策略（延迟启动避免过早停止）
EARLY_STOPPING_PATIENCE = 35  # 耐心值
EARLY_STOPPING_START_EPOCH = 50  # 从第50轮开始监控

# ==================== 数据增强参数 ====================
# 是否使用数据增强
USE_DATA_AUGMENTATION = True  # 启用数据增强，提升模型泛化能力和准确率

# ==================== 其他配置 ====================
# 是否使用 GPU
USE_GPU = True

# 随机种子（保证可复现性）
RANDOM_SEED = 42

# 模型保存格式
MODEL_FORMAT = 'keras'  # TensorFlow 2.x 推荐格式

# Checkpoint 保存配置
CHECKPOINT_SAVE_BEST_ONLY = True  # 只保存最优模型
CHECKPOINT_MONITOR = 'val_accuracy'  # 监控验证集准确率
CHECKPOINT_MODE = 'max'  # 准确率越高越好

# ==================== 打印配置信息 ====================
def print_config():
    """打印配置信息"""
    print("=" * 80)
    print(" " * 30 + "配置信息")
    print("=" * 80)
    print(f"验证码图片目录: {CAPTCHA_DIR}")
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"图像尺寸: {IMAGE_WIDTH}×{IMAGE_HEIGHT}×{IMAGE_CHANNELS}")
    print(f"字符集: {CHAR_SET[:20]}... (共{CHAR_SET_LEN}个字符)")
    print(f"最大验证码长度: {MAX_CAPTCHA}")
    print(f"输出维度: {OUTPUT_SIZE} ({MAX_CAPTCHA} × {CHAR_SET_LEN})")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"卷积层配置: {CONV_FILTERS}")
    print(f"全连接层单元: {FC_UNITS}")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
