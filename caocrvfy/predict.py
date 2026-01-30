#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证码预测模块
功能：使用训练好的模型预测验证码
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import config
import utils


class CaptchaPredictor:
    """验证码预测器"""
    
    def __init__(self, model_path=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型文件路径
        """
        self.model_path = model_path or os.path.join(config.MODEL_DIR, 'best_model.keras')
        self.model = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("✓ 模型加载成功")
    
    def predict_image(self, image_path, verbose=True):
        """
        预测单张验证码图片
        
        参数:
            image_path: 图片路径
            verbose: 是否打印详细信息
        
        返回:
            预测的验证码文本
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 加载和预处理图片
        img = utils.load_image(image_path)
        
        # 添加批次维度
        img_batch = np.expand_dims(img, axis=0)
        
        # 预测
        prediction = self.model.predict(img_batch, verbose=0)
        
        # 解码
        predicted_text = utils.vector_to_text(prediction[0])
        
        if verbose:
            print(f"图片: {os.path.basename(image_path)}")
            print(f"预测: {predicted_text}")
            
            # 如果文件名包含真实标签，显示对比
            try:
                true_text = utils.parse_filename(os.path.basename(image_path))
                match = "✓" if true_text == predicted_text else "✗"
                print(f"真实: {true_text}")
                print(f"匹配: {match}")
            except:
                pass
        
        return predicted_text
    
    def predict_batch(self, image_paths, verbose=True):
        """
        批量预测多张验证码图片
        
        参数:
            image_paths: 图片路径列表
            verbose: 是否打印详细信息
        
        返回:
            预测结果列表
        """
        predictions = []
        
        if verbose:
            print(f"批量预测 {len(image_paths)} 张图片...")
            print("-" * 80)
        
        for i, image_path in enumerate(image_paths):
            if verbose and (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{len(image_paths)}")
            
            try:
                predicted_text = self.predict_image(image_path, verbose=False)
                predictions.append({
                    'image_path': image_path,
                    'predicted_text': predicted_text
                })
            except Exception as e:
                print(f"预测失败: {image_path}, 错误: {e}")
                predictions.append({
                    'image_path': image_path,
                    'predicted_text': None,
                    'error': str(e)
                })
        
        if verbose:
            print("-" * 80)
            print(f"✓ 批量预测完成")
        
        return predictions
    
    def predict_directory(self, directory, verbose=True):
        """
        预测目录下所有验证码图片
        
        参数:
            directory: 图片目录
            verbose: 是否打印详细信息
        
        返回:
            预测结果列表和准确率
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 获取所有png图片
        image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        image_paths = [os.path.join(directory, f) for f in image_files]
        
        if len(image_paths) == 0:
            print(f"目录中没有图片: {directory}")
            return [], 0.0
        
        # 批量预测
        predictions = self.predict_batch(image_paths, verbose=verbose)
        
        # 计算准确率
        correct = 0
        total = 0
        
        for pred in predictions:
            if pred['predicted_text'] is None:
                continue
            
            try:
                true_text = utils.parse_filename(os.path.basename(pred['image_path']))
                if true_text == pred['predicted_text']:
                    correct += 1
                total += 1
            except:
                pass
        
        accuracy = correct / total if total > 0 else 0.0
        
        if verbose:
            print()
            print("=" * 80)
            print(" " * 30 + "预测结果统计")
            print("=" * 80)
            print(f"总图片数: {len(image_paths)}")
            print(f"成功预测: {len([p for p in predictions if p['predicted_text'] is not None])}")
            print(f"预测失败: {len([p for p in predictions if p['predicted_text'] is None])}")
            if total > 0:
                print(f"准确预测: {correct}/{total}")
                print(f"准确率: {accuracy*100:.2f}%")
            print("=" * 80)
        
        return predictions, accuracy
    
    def predict_from_array(self, image_array):
        """
        从numpy数组预测验证码
        
        参数:
            image_array: 图片数组 (H, W, C) 或 (N, H, W, C)
        
        返回:
            预测的验证码文本或文本列表
        """
        # 检查是否为批次数据
        if image_array.ndim == 3:
            # 单张图片，添加批次维度
            image_array = np.expand_dims(image_array, axis=0)
            single_image = True
        else:
            single_image = False
        
        # 预测
        predictions = self.model.predict(image_array, verbose=0)
        
        # 解码
        predicted_texts = [utils.vector_to_text(pred) for pred in predictions]
        
        # 返回单个或列表
        return predicted_texts[0] if single_image else predicted_texts


def interactive_predict():
    """交互式预测模式"""
    print("=" * 80)
    print(" " * 25 + "验证码识别 - 交互式预测")
    print("=" * 80)
    print()
    
    # 创建预测器
    predictor = CaptchaPredictor()
    print()
    
    while True:
        print("-" * 80)
        print("请选择操作:")
        print("  1. 预测单张图片")
        print("  2. 预测目录下所有图片")
        print("  3. 退出")
        print("-" * 80)
        
        choice = input("请输入选项 (1/2/3): ").strip()
        print()
        
        if choice == '1':
            # 预测单张图片
            image_path = input("请输入图片路径: ").strip()
            if image_path:
                try:
                    predicted_text = predictor.predict_image(image_path)
                    print()
                except Exception as e:
                    print(f"预测失败: {e}")
                    print()
        
        elif choice == '2':
            # 预测目录
            directory = input("请输入图片目录路径: ").strip()
            if directory:
                try:
                    predictions, accuracy = predictor.predict_directory(directory)
                    print()
                except Exception as e:
                    print(f"预测失败: {e}")
                    print()
        
        elif choice == '3':
            # 退出
            print("退出预测程序")
            break
        
        else:
            print("无效的选项，请重新选择")
            print()


# 测试预测功能
if __name__ == '__main__':
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == '--dir' and len(sys.argv) > 2:
            # 预测目录
            directory = sys.argv[2]
            predictor = CaptchaPredictor()
            predictions, accuracy = predictor.predict_directory(directory)
        
        elif sys.argv[1] == '--image' and len(sys.argv) > 2:
            # 预测单张图片
            image_path = sys.argv[2]
            predictor = CaptchaPredictor()
            predictor.predict_image(image_path)
        
        else:
            print("用法:")
            print("  python predict.py --dir <目录路径>    # 预测目录下所有图片")
            print("  python predict.py --image <图片路径>  # 预测单张图片")
            print("  python predict.py                    # 交互式模式")
    
    else:
        # 交互式模式
        interactive_predict()
