#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX模型测试验证脚本
功能：使用ONNX Runtime加载模型，对captcha/img目录下的验证码图片进行预测测试
"""

import os
import sys
import time
import string
import numpy as np
import onnxruntime as ort
from PIL import Image

# 添加caocrvfy到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'caocrvfy'))

from core import config, utils


class ONNXCaptchaPredictor:
    """基于ONNX Runtime的验证码预测器"""
    
    def __init__(self, onnx_model_path):
        """
        初始化ONNX预测器
        
        参数:
            onnx_model_path: ONNX模型文件路径
        """
        self.model_path = onnx_model_path
        
        # 创建ONNX Runtime推理会话
        print(f"正在加载ONNX模型: {onnx_model_path}")
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']  # 使用CPU推理
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        print(f"✓ ONNX模型加载成功")
        print(f"  输入名称: {self.input_name}")
        print(f"  输入形状: {input_shape}")
        print(f"  输出名称: {self.output_name}")
        print(f"  输出形状: {output_shape}")
        print()
    
    def predict_image(self, image_path, verbose=True):
        """
        预测单张验证码图片
        
        参数:
            image_path: 图片路径
            verbose: 是否打印详细信息
        
        返回:
            预测的验证码文本和置信度
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 加载和预处理图片
        img = utils.load_image(image_path)
        
        # 添加批次维度
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        
        # ONNX Runtime推理
        start_time = time.time()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: img_batch}
        )
        inference_time = (time.time() - start_time) * 1000  # 转为毫秒
        
        # 解码预测结果
        prediction = outputs[0][0]  # 取第一个样本的输出
        
        # 自动检测模型的输出维度
        output_dim = len(prediction)
        # 推断MAX_CAPTCHA和CHAR_SET_LEN
        # 尝试不同的MAX_CAPTCHA值
        detected_max_captcha = None
        detected_char_set_len = None
        for possible_max in [8, 7, 6, 5, 4]:
            if output_dim % possible_max == 0:
                detected_max_captcha = possible_max
                detected_char_set_len = output_dim // possible_max
                break
        
        if detected_max_captcha is None:
            raise ValueError(f"无法解析输出维度 {output_dim}")
        
        # 使用检测到的维度进行reshape
        prediction_reshaped = prediction.reshape((detected_max_captcha, detected_char_set_len))
        
        # 解码文本（使用相应的字符集）
        # 504 = 8 × 63（旧字符集，不含数学运算符）
        # 544 = 8 × 68（新字符集，含数学运算符）
        if detected_char_set_len == 63:
            # 旧字符集（不含+-*=?）
            char_set = string.digits + string.ascii_letters + ' '
        elif detected_char_set_len == 68:
            # 新字符集（含+-*=?）
            char_set = config.CHAR_SET
        else:
            # 使用默认字符集的前N个字符
            char_set = config.CHAR_SET[:detected_char_set_len]
        
        # 解码
        text = []
        confidences = []
        for i in range(detected_max_captcha):
            max_prob = np.max(prediction_reshaped[i])
            char_idx = np.argmax(prediction_reshaped[i])
            
            if max_prob >= 0.5:  # 置信度阈值
                char = char_set[char_idx]
            else:
                char = ' '  # 低置信度用空格
            
            text.append(char)
            confidences.append(max_prob)
        
        predicted_text = ''.join(text).rstrip(' ')
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
        
        # 准确率统计（仅统计能解析标签的图片）
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
                for case in error_cases[:5]:  # 只显示前5个
                    print(f"  {case['filename']}")
                    print(f"    真实: {case['true_text']}")
                    print(f"    预测: {case['predicted_text']}")
                    print(f"    置信度: {case['confidence']:.4f}")
                if len(error_cases) > 5:
                    print(f"  ... 还有 {len(error_cases) - 5} 个错误案例")
        
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
    # ONNX模型路径
    onnx_model_path = os.path.join(
        os.path.dirname(__file__),
        'test_models',
        'final_model2.onnx'
    )
    
    # 测试图片目录
    test_image_dir = os.path.join(
        os.path.dirname(__file__),
        'captcha',
        'img'
    )
    
    # 检查路径
    if not os.path.exists(onnx_model_path):
        print(f"错误: ONNX模型文件不存在: {onnx_model_path}")
        return
    
    if not os.path.exists(test_image_dir):
        print(f"错误: 测试图片目录不存在: {test_image_dir}")
        return
    
    # 创建预测器
    predictor = ONNXCaptchaPredictor(onnx_model_path)
    
    # 批量测试
    stats = predictor.predict_directory(test_image_dir, verbose=True)
    
    print("\n✓ 测试完成！")


if __name__ == '__main__':
    main()
