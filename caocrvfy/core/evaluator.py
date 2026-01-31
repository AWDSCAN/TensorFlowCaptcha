#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估模块（参考captcha_trainer模块化设计）
功能：评估模型性能，确保功能单一性
"""

from . import utils


class CaptchaEvaluator:
    """
    验证码模型评估器
    
    参考：captcha_trainer/validation.py
    职责：
    - 计算模型评估指标
    - 生成预测示例
    - 输出评估报告
    """
    
    def __init__(self, model):
        """
        初始化评估器
        
        参数:
            model: 训练好的Keras模型
        """
        self.model = model
    
    def evaluate(self, val_data, verbose=True):
        """
        评估模型性能
        
        参考：captcha_trainer/validation.py的accuracy_calculation
        
        参数:
            val_data: 验证数据 (X, y)
            verbose: 是否打印详细信息
        
        返回:
            评估指标字典
        """
        val_images, val_labels = val_data
        
        if verbose:
            print("\n" + "=" * 80)
            print(" " * 30 + "模型评估")
            print("=" * 80)
        
        # Keras内置评估
        results = self.model.evaluate(val_images, val_labels, verbose=0)
        
        keras_metrics = {
            'loss': results[0],
            'binary_accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        if verbose:
            print(f"验证集损失: {keras_metrics['loss']:.4f}")
            print(f"二进制准确率: {keras_metrics['binary_accuracy']:.4f}")
            print(f"精确率: {keras_metrics['precision']:.4f}")
            print(f"召回率: {keras_metrics['recall']:.4f}")
            print()
        
        # 完整匹配准确率评估
        if verbose:
            print("计算完整验证码匹配准确率...")
        
        full_match_acc = self._calculate_full_match_accuracy(val_images, val_labels)
        
        if verbose:
            print(f"完整匹配准确率: {full_match_acc:.4f} ({full_match_acc*100:.2f}%)")
            print()
        
        # 合并所有指标
        metrics = {**keras_metrics, 'full_match_accuracy': full_match_acc}
        
        return metrics
    
    def _calculate_full_match_accuracy(self, images, labels):
        """
        计算完整匹配准确率
        
        参考：captcha_trainer/validation.py的accuracy_calculation逻辑
        
        参数:
            images: 图像数据
            labels: 标签数据
        
        返回:
            完整匹配准确率
        """
        predictions = self.model.predict(images, verbose=0)
        
        # 确保预测是正确的形状 (batch_size, MAX_CAPTCHA * CHAR_SET_LEN)
        if len(predictions.shape) == 2:
            # 如果已经是2D，直接使用
            pass
        elif len(predictions.shape) == 3:
            # 如果是3D (batch_size, MAX_CAPTCHA, CHAR_SET_LEN)，需要flatten
            predictions = predictions.reshape(predictions.shape[0], -1)
        
        # 解码预测和真实标签
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in labels]
        
        # 计算准确率
        accuracy = utils.calculate_accuracy(true_texts, pred_texts)
        
        return accuracy
    
    def show_prediction_examples(self, val_data, num_examples=10):
        """
        显示预测示例
        
        参考：captcha_trainer的预测输出格式
        
        参数:
            val_data: 验证数据 (X, y)
            num_examples: 显示的示例数量
        """
        val_images, val_labels = val_data
        
        predictions = self.model.predict(val_images[:num_examples], verbose=0)
        
        # 解码
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in val_labels[:num_examples]]
        
        print("示例预测（前10个）:")
        print("-" * 80)
        print(f"{'真实值':<15} {'预测值':<15} {'匹配':<10}")
        print("-" * 80)
        
        for i in range(min(num_examples, len(true_texts))):
            match = "✓" if true_texts[i] == pred_texts[i] else "✗"
            print(f"{true_texts[i]:<15} {pred_texts[i]:<15} {match:<10}")
        
        print("=" * 80)
    
    def generate_report(self, val_data):
        """
        生成完整的评估报告
        
        参考：captcha_trainer的评估报告格式
        
        参数:
            val_data: 验证数据 (X, y)
        
        返回:
            评估指标字典
        """
        # 评估模型
        metrics = self.evaluate(val_data, verbose=True)
        
        # 显示预测示例
        self.show_prediction_examples(val_data, num_examples=10)
        
        return metrics
