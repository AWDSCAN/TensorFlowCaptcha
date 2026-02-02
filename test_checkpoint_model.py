#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint模型测试验证脚本
功能：从checkpoint加载模型，对captcha/img目录下的验证码图片进行预测测试
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

from core import config, utils


class CheckpointCaptchaPredictor:
    """基于Checkpoint的验证码预测器"""
    
    def __init__(self, checkpoint_dir, model_structure=None):
        """
        初始化Checkpoint预测器
        
        参数:
            checkpoint_dir: checkpoint文件所在目录
            model_structure: 可选的模型结构（如果只有权重文件）
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        
        # 加载模型
        self._load_model(model_structure)
    
    def _load_model(self, model_structure=None):
        """加载checkpoint模型"""
        # 方法1: 尝试直接加载.keras文件（完整模型）
        keras_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.keras')]
        if keras_files:
            keras_path = os.path.join(self.checkpoint_dir, keras_files[0])
            print(f"正在加载完整模型: {keras_path}")
            
            # 尝试加载（可能需要自定义对象）
            try:
                # 首先尝试不带自定义对象
                self.model = keras.models.load_model(keras_path, compile=False)
                print("✓ 模型加载成功（标准加载）")
            except Exception as e1:
                print(f"标准加载失败: {e1}")
                # 尝试带自定义对象
                try:
                    from extras.model_enhanced import WeightedBinaryCrossentropy
                    custom_objects = {
                        'WeightedBinaryCrossentropy': WeightedBinaryCrossentropy
                    }
                    self.model = keras.models.load_model(keras_path, custom_objects=custom_objects, compile=False)
                    print("✓ 模型加载成功（带自定义对象）")
                except Exception as e2:
                    print(f"自定义对象加载也失败: {e2}")
                    raise
        else:
            # 方法2: 从checkpoint文件加载权重（需要先构建模型结构）
            checkpoint_path = os.path.join(self.checkpoint_dir, 'ckpt-1')
            if not os.path.exists(checkpoint_path + '.index'):
                raise FileNotFoundError(f"找不到checkpoint文件: {checkpoint_path}")
            
            if model_structure is None:
                raise ValueError("从checkpoint加载需要提供model_structure")
            
            print(f"正在从checkpoint加载权重: {checkpoint_path}")
            self.model = model_structure
            self.model.load_weights(checkpoint_path)
            print("✓ 权重加载成功")
        
        # 打印模型信息
        print(f"\n模型信息:")
        
        # 处理单输入或多输入的情况
        if isinstance(self.model.input, list):
            for idx, inp in enumerate(self.model.input):
                print(f"  输入{idx+1}形状: {inp.shape}")
        else:
            print(f"  输入形状: {self.model.input.shape}")
        
        # 处理单输出或多输出的情况
        if isinstance(self.model.output, list):
            for idx, out in enumerate(self.model.output):
                print(f"  输出{idx+1}形状: {out.shape}")
        else:
            print(f"  输出形状: {self.model.output.shape}")
        
        print(f"  参数数量: {self.model.count_params():,}")
        print()
    
    def predict_image(self, image_path, verbose=True):
        """
        预测单张验证码图片
        
        参数:
            image_path: 图片路径
            verbose: 是否打印详细信息
        
        返回:
            预测的验证码文本、置信度和推理时间
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 加载和预处理图片
        img = utils.load_image(image_path)
        
        # 添加批次维度
        img_batch = np.expand_dims(img, axis=0)
        
        # 模型推理
        start_time = time.time()
        prediction = self.model.predict(img_batch, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # 转为毫秒
        
        # 解码预测结果
        predicted_text = utils.vector_to_text(prediction[0])
        
        # 计算平均置信度
        prediction_reshaped = prediction[0].reshape((config.MAX_CAPTCHA, config.CHAR_SET_LEN))
        confidences = []
        for i in range(config.MAX_CAPTCHA):
            max_prob = np.max(prediction_reshaped[i])
            confidences.append(max_prob)
        avg_confidence = np.mean(confidences)
        
        if verbose:
            filename = os.path.basename(image_path)
            print(f"文件名: {filename}")
            print(f"预测结果: {predicted_text}")
            print(f"平均置信度: {avg_confidence:.4f}")
            print(f"推理时间: {inference_time:.2f} ms")
            
            # 如果文件名包含真实标签，显示对比
            try:
                true_text = utils.parse_filename(filename)
                match = "✓" if true_text == predicted_text else "✗"
                print(f"真实标签: {true_text}")
                print(f"匹配结果: {match}")
                
                # 计算字符级准确率
                if len(true_text) == len(predicted_text):
                    char_matches = sum(1 for a, b in zip(true_text, predicted_text) if a == b)
                    char_accuracy = char_matches / len(true_text) * 100
                    print(f"字符准确率: {char_accuracy:.1f}% ({char_matches}/{len(true_text)})")
                else:
                    print(f"长度不匹配: 真实={len(true_text)}, 预测={len(predicted_text)}")
            except Exception as e:
                print(f"(无法解析真实标签: {e})")
            
            print("-" * 80)
        
        return predicted_text, avg_confidence, inference_time
    
    def predict_directory(self, directory, verbose=True):
        """
        预测目录下所有验证码图片
        
        参数:
            directory: 图片目录
            verbose: 是否打印详细信息
        
        返回:
            统计结果字典
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 获取所有png图片
        image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.png')])
        
        if len(image_files) == 0:
            print(f"目录中没有PNG图片: {directory}")
            return {}
        
        print(f"="*80)
        print(f"开始测试 {len(image_files)} 张验证码图片")
        print(f"目录: {directory}")
        print(f"="*80)
        print()
        
        # 统计信息
        total_count = len(image_files)
        correct_count = 0
        error_count = 0
        inference_times = []
        confidences = []
        results = []
        
        # 逐个预测
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(directory, filename)
            
            if verbose:
                print(f"[{i}/{total_count}] ", end="")
            
            try:
                predicted_text, confidence, inference_time = self.predict_image(
                    image_path, 
                    verbose=verbose
                )
                
                # 尝试获取真实标签
                try:
                    true_text = utils.parse_filename(filename)
                    is_correct = (true_text == predicted_text)
                    
                    if is_correct:
                        correct_count += 1
                    
                    results.append({
                        'filename': filename,
                        'true_text': true_text,
                        'predicted_text': predicted_text,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'is_correct': is_correct
                    })
                except:
                    # 无法解析真实标签
                    results.append({
                        'filename': filename,
                        'true_text': None,
                        'predicted_text': predicted_text,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'is_correct': None
                    })
                
                inference_times.append(inference_time)
                confidences.append(confidence)
                
            except Exception as e:
                error_count += 1
                print(f"✗ 预测失败: {e}")
                print("-" * 80)
        
        # 打印统计结果
        print()
        print("="*80)
        print("测试统计结果")
        print("="*80)
        print(f"总测试数量: {total_count}")
        print(f"预测成功数: {total_count - error_count}")
        print(f"预测失败数: {error_count}")
        
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
            print(f"\n推理性能:")
            print(f"  平均推理时间: {avg_inference_time:.2f} ms")
            print(f"  FPS (帧率): {fps:.1f}")
            print(f"  最快: {min(inference_times):.2f} ms")
            print(f"  最慢: {max(inference_times):.2f} ms")
        
        if confidences:
            avg_confidence = np.mean(confidences)
            print(f"\n置信度统计:")
            print(f"  平均置信度: {avg_confidence:.4f}")
            print(f"  最高: {max(confidences):.4f}")
            print(f"  最低: {min(confidences):.4f}")
        
        # 准确率统计
        valid_results = [r for r in results if r['is_correct'] is not None]
        if valid_results:
            accuracy = correct_count / len(valid_results) * 100
            print(f"\n准确率统计:")
            print(f"  正确数量: {correct_count}/{len(valid_results)}")
            print(f"  准确率: {accuracy:.2f}%")
            
            # 显示错误案例
            error_cases = [r for r in valid_results if not r['is_correct']]
            if error_cases:
                print(f"\n错误案例 ({len(error_cases)} 个):")
                for idx, case in enumerate(error_cases[:10], 1):  # 显示前10个
                    print(f"  {idx}. {case['filename']}")
                    print(f"     真实: {case['true_text']}")
                    print(f"     预测: {case['predicted_text']}")
                    print(f"     置信度: {case['confidence']:.4f}")
                if len(error_cases) > 10:
                    print(f"  ... 还有 {len(error_cases) - 10} 个错误案例")
        
        print("="*80)
        
        return {
            'total_count': total_count,
            'correct_count': correct_count,
            'error_count': error_count,
            'accuracy': accuracy if valid_results else None,
            'avg_inference_time': np.mean(inference_times) if inference_times else None,
            'avg_confidence': np.mean(confidences) if confidences else None,
            'results': results
        }


def main():
    """主函数"""
    # Checkpoint目录
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        'test_models'
    )
    
    # 测试图片目录
    test_image_dir = os.path.join(
        os.path.dirname(__file__),
        'captcha',
        'img'
    )
    
    # 检查路径
    if not os.path.exists(checkpoint_dir):
        print(f"错误: Checkpoint目录不存在: {checkpoint_dir}")
        return
    
    if not os.path.exists(test_image_dir):
        print(f"错误: 测试图片目录不存在: {test_image_dir}")
        return
    
    print("="*80)
    print("Checkpoint模型测试验证")
    print("="*80)
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"测试图片目录: {test_image_dir}")
    print()
    
    # 创建预测器
    predictor = CheckpointCaptchaPredictor(checkpoint_dir)
    
    # 批量测试
    stats = predictor.predict_directory(test_image_dir, verbose=True)
    
    print("\n✓ 测试完成！")
    
    # 保存详细结果到文件
    if stats and stats.get('results'):
        import json
        result_file = 'checkpoint_test_results.json'
        
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_converted = convert_types(stats['results'])
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2)
        print(f"\n详细测试结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
