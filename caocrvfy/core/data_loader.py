#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载模块
功能：加载验证码图片，生成训练和验证数据集
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from . import config
from . import utils


class CaptchaDataLoader:
    """验证码数据加载器"""
    
    def __init__(self, captcha_dir=None, validation_split=None):
        """
        初始化数据加载器
        
        参数:
            captcha_dir: 验证码图片目录
            validation_split: 验证集比例
        """
        self.captcha_dir = captcha_dir or config.CAPTCHA_DIR
        self.validation_split = validation_split or config.VALIDATION_SPLIT
        
        # 数据存储
        self.image_paths = []
        self.labels = []
        
        # 数据集划分
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
    
    def load_data(self):
        """
        从目录加载所有验证码图片
        
        返回:
            加载的图片数量
        """
        print(f"正在从 {self.captcha_dir} 加载验证码图片...")
        
        if not os.path.exists(self.captcha_dir):
            raise FileNotFoundError(f"验证码目录不存在: {self.captcha_dir}")
        
        # 获取所有png图片
        image_files = [f for f in os.listdir(self.captcha_dir) 
                      if f.endswith('.png')]
        
        if len(image_files) == 0:
            raise ValueError(f"验证码目录中没有图片: {self.captcha_dir}")
        
        print(f"找到 {len(image_files)} 张验证码图片")
        
        # 加载每张图片和对应的标签
        for filename in image_files:
            image_path = os.path.join(self.captcha_dir, filename)
            
            # 解析文件名获取验证码文本
            captcha_text = utils.parse_filename(filename)
            
            # 过滤超长验证码
            if len(captcha_text) > config.MAX_CAPTCHA:
                print(f"跳过超长验证码: {filename} (长度: {len(captcha_text)})")
                continue
            
            # 验证字符是否都在字符集中
            if not all(c in config.CHAR_SET for c in captcha_text):
                print(f"跳过包含非法字符的验证码: {filename}")
                continue
            
            self.image_paths.append(image_path)
            self.labels.append(captcha_text)
        
        print(f"✓ 成功加载 {len(self.image_paths)} 张有效验证码")
        
        return len(self.image_paths)
    
    def prepare_dataset(self):
        """
        准备训练集和验证集
        
        返回:
            (train_images, train_labels, val_images, val_labels)
        """
        if len(self.image_paths) == 0:
            raise ValueError("请先调用 load_data() 加载数据")
        
        print("正在准备数据集...")
        
        # 加载所有图片
        images = []
        labels = []
        
        for i, (image_path, label_text) in enumerate(zip(self.image_paths, self.labels)):
            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i+1}/{len(self.image_paths)}")
            
            # 加载图片
            img = utils.load_image(image_path)
            images.append(img)
            
            # 转换标签为向量
            label_vector = utils.text_to_vector(label_text)
            labels.append(label_vector)
        
        # 转换为numpy数组
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        print(f"图片数组形状: {images.shape}")
        print(f"标签数组形状: {labels.shape}")
        
        # 划分训练集和验证集
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels,
            test_size=self.validation_split,
            random_state=config.RANDOM_SEED
        )
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        
        print(f"训练集大小: {len(train_images)}")
        print(f"验证集大小: {len(val_images)}")
        
        return train_images, train_labels, val_images, val_labels
    
    def get_batch_generator(self, batch_size=None, is_training=True):
        """
        创建批次数据生成器
        
        参数:
            batch_size: 批次大小
            is_training: 是否为训练集
        
        返回:
            数据生成器
        """
        batch_size = batch_size or config.BATCH_SIZE
        
        if is_training:
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.val_images
            labels = self.val_labels
        
        num_samples = len(images)
        indices = np.arange(num_samples)
        
        while True:
            # 随机打乱数据（仅训练集）
            if is_training:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]
                
                yield batch_images, batch_labels
    
    def get_statistics(self):
        """
        获取数据集统计信息
        
        返回:
            统计信息字典
        """
        if len(self.labels) == 0:
            return {}
        
        # 验证码长度分布
        length_dist = {}
        for label in self.labels:
            length = len(label)
            length_dist[length] = length_dist.get(length, 0) + 1
        
        # 字符频率统计
        char_freq = {}
        for label in self.labels:
            for char in label:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        return {
            'total_samples': len(self.labels),
            'length_distribution': length_dist,
            'char_frequency': char_freq,
            'unique_chars': len(char_freq)
        }
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()
        
        print("=" * 80)
        print(" " * 30 + "数据集统计")
        print("=" * 80)
        print(f"总样本数: {stats['total_samples']}")
        print()
        
        print("验证码长度分布:")
        for length, count in sorted(stats['length_distribution'].items()):
            percentage = count / stats['total_samples'] * 100
            print(f"  长度 {length}: {count} 张 ({percentage:.1f}%)")
        print()
        
        print(f"唯一字符数: {stats['unique_chars']}")
        print("字符频率前10:")
        sorted_chars = sorted(stats['char_frequency'].items(), 
                            key=lambda x: x[1], reverse=True)
        for char, freq in sorted_chars[:10]:
            print(f"  '{char}': {freq} 次")
        print("=" * 80)


# 测试数据加载器
if __name__ == '__main__':
    print("测试数据加载器...")
    print()
    
    # 创建数据加载器
    loader = CaptchaDataLoader()
    
    # 加载数据
    loader.load_data()
    print()
    
    # 打印统计信息
    loader.print_statistics()
    print()
    
    # 准备数据集
    train_images, train_labels, val_images, val_labels = loader.prepare_dataset()
    print()
    
    # 测试批次生成器
    print("测试批次生成器...")
    gen = loader.get_batch_generator(batch_size=4, is_training=True)
    batch_images, batch_labels = next(gen)
    print(f"批次图片形状: {batch_images.shape}")
    print(f"批次标签形状: {batch_labels.shape}")
    
    # 显示第一个样本
    print(f"\n第一个样本的标签向量转换:")
    sample_text = utils.vector_to_text(batch_labels[0])
    print(f"验证码文本: {sample_text}")
    
    print()
    print("=" * 80)
    print("✓ 数据加载器测试完成")
