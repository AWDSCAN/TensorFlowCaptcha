#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估模块（参考captcha_trainer模块化设计）
功能：评估模型性能，确保功能单一性

支持数学运算题三步验证流程：
1. 识别验证码中的数学运算题内容
2. 对识别出的内容进行计算
3. 比对计算结果和预期答案
"""

import os
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
    
    def __init__(self, model, image_paths=None):
        """
        初始化评估器
        
        参数:
            model: 训练好的Keras模型
            image_paths: 验证图片路径列表（可选，用于数学题验证）
        """
        self.model = model
        self.image_paths = image_paths
    
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
    
    def show_prediction_examples(self, val_data, num_examples=10, show_math_validation=True):
        """
        显示预测示例（支持数学题三步验证）
        
        参考：captcha_trainer的预测输出格式
        
        参数:
            val_data: 验证数据 (X, y)
            num_examples: 显示的示例数量
            show_math_validation: 是否显示数学题验证详情
        """
        val_images, val_labels = val_data
        
        predictions = self.model.predict(val_images[:num_examples], verbose=0)
        
        # 解码
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in val_labels[:num_examples]]
        
        print("示例预测（前10个）:")
        print("-" * 80)
        
        # 检查是否有数学题
        has_math = False
        math_validations = []
        
        if show_math_validation and self.image_paths is not None:
            for i in range(min(num_examples, len(true_texts))):
                if i < len(self.image_paths):
                    filename = os.path.basename(self.image_paths[i])
                    expected_answer = utils.extract_answer_from_filename(filename)
                    
                    if expected_answer is not None:
                        has_math = True
                        validation = utils.validate_math_captcha(pred_texts[i], expected_answer)
                        math_validations.append((i, filename, validation))
        
        if has_math:
            # 数学题模式：显示详细验证信息
            print(f"{'序号':<5} {'真实题目':<12} {'识别结果':<12} {'计算答案':<8} {'预期答案':<8} {'状态':<15}")
            print("-" * 80)
            
            for i in range(min(num_examples, len(true_texts))):
                true_text = true_texts[i]
                pred_text = pred_texts[i]
                
                # 查找对应的数学题验证结果
                math_val = None
                expected_ans = "N/A"
                for idx, fname, val in math_validations:
                    if idx == i:
                        math_val = val
                        expected_ans = utils.extract_answer_from_filename(fname) or "N/A"
                        break
                
                if math_val is not None:
                    # 数学题验证
                    calc_ans = math_val['step2_calculated'] or "计算失败"
                    
                    if math_val['is_correct']:
                        status = "✓ 完全正确"
                    elif math_val['step3_matched']:
                        status = "✓ 答案正确"
                    elif math_val['step2_calculated']:
                        status = f"✗ 答案错误"
                    elif math_val['step1_recognized']:
                        status = "✗ 计算失败"
                    else:
                        status = "✗ 识别失败"
                    
                    print(f"{i+1:<5} {true_text:<12} {pred_text:<12} {calc_ans:<8} {expected_ans:<8} {status:<15}")
                else:
                    # 普通验证码
                    match = "✓ 匹配" if true_text == pred_text else "✗ 不匹配"
                    print(f"{i+1:<5} {true_text:<12} {pred_text:<12} {'N/A':<8} {'N/A':<8} {match:<15}")
        
        else:
            # 普通模式：简单显示
            print(f"{'真实值':<15} {'预测值':<15} {'匹配':<10}")
            print("-" * 80)
            
            for i in range(min(num_examples, len(true_texts))):
                match = "✓" if true_texts[i] == pred_texts[i] else "✗"
                print(f"{true_texts[i]:<15} {pred_texts[i]:<15} {match:<10}")
        
        print("=" * 80)
    
    def evaluate_math_captcha(self, val_data, verbose=True):
        """
        专门针对数学题验证码的三步验证评估
        
        三步验证流程:
        1. 步骤1: 识别验证码中的数学运算题内容
        2. 步骤2: 对识别出的内容进行计算
        3. 步骤3: 比对计算结果和预期答案
        
        参数:
            val_data: 验证数据 (X, y)
            verbose: 是否打印详细信息
        
        返回:
            数学题评估指标字典
        """
        if self.image_paths is None:
            if verbose:
                print("警告: 未提供图片路径，无法进行数学题验证")
            return None
        
        val_images, val_labels = val_data
        
        if verbose:
            print("\n" + "=" * 80)
            print(" " * 25 + "数学题三步验证评估")
            print("=" * 80)
        
        # 获取预测
        predictions = self.model.predict(val_images, verbose=0)
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        
        # 统计数学题
        math_count = 0
        step1_success = 0  # 成功识别为数学表达式
        step2_success = 0  # 成功计算出结果
        step3_success = 0  # 计算结果正确
        
        math_details = []
        
        for i, pred_text in enumerate(pred_texts):
            if i >= len(self.image_paths):
                break
            
            filename = os.path.basename(self.image_paths[i])
            expected_answer = utils.extract_answer_from_filename(filename)
            
            # 只处理数学题类型
            if expected_answer is None:
                continue
            
            math_count += 1
            
            # 执行三步验证
            validation = utils.validate_math_captcha(pred_text, expected_answer)
            
            if validation['step1_recognized']:
                step1_success += 1
            if validation['step2_calculated'] is not None:
                step2_success += 1
            if validation['step3_matched']:
                step3_success += 1
            
            # 保存详情用于展示
            if len(math_details) < 20:  # 只保留前20个用于展示
                math_details.append({
                    'filename': filename,
                    'predicted': pred_text,
                    'expected_answer': expected_answer,
                    'validation': validation
                })
        
        if math_count == 0:
            if verbose:
                print("✓ 验证集中没有数学题类型的验证码")
                print("=" * 80)
            return None
        
        # 计算各步骤准确率
        step1_acc = step1_success / math_count
        step2_acc = step2_success / math_count
        step3_acc = step3_success / math_count
        
        if verbose:
            print(f"数学题验证码数量: {math_count}")
            print()
            print("三步验证准确率:")
            print(f"  步骤1 - 识别为数学表达式: {step1_acc:.4f} ({step1_acc*100:.2f}%) [{step1_success}/{math_count}]")
            print(f"  步骤2 - 成功计算结果:     {step2_acc:.4f} ({step2_acc*100:.2f}%) [{step2_success}/{math_count}]")
            print(f"  步骤3 - 答案完全正确:     {step3_acc:.4f} ({step3_acc*100:.2f}%) [{step3_success}/{math_count}]")
            print()
            
            # 显示详细示例
            if len(math_details) > 0:
                print("数学题验证详情（前10个）:")
                print("-" * 100)
                print(f"{'文件名':<35} {'识别结果':<12} {'计算答案':<10} {'预期答案':<10} {'验证状态':<20}")
                print("-" * 100)
                
                for detail in math_details[:10]:
                    val = detail['validation']
                    calc = val['step2_calculated'] or "失败"
                    
                    if val['is_correct']:
                        status = "✓ 三步全通过"
                    elif val['step3_matched']:
                        status = "✓ 答案正确"
                    elif val['step2_calculated']:
                        status = f"✗ 答案错误"
                    elif val['step1_recognized']:
                        status = "✗ 计算失败"
                    else:
                        status = "✗ 识别失败"
                    
                    print(f"{detail['filename']:<35} {detail['predicted']:<12} {calc:<10} {detail['expected_answer']:<10} {status:<20}")
                
                print("=" * 100)
        
        metrics = {
            'math_count': math_count,
            'step1_recognition_accuracy': step1_acc,
            'step2_calculation_accuracy': step2_acc,
            'step3_answer_accuracy': step3_acc,
            'math_details': math_details
        }
        
        return metrics
    
    def generate_report(self, val_data, include_math_validation=True):
        """
        生成完整的评估报告（包含数学题三步验证）
        
        参考：captcha_trainer的评估报告格式
        
        参数:
            val_data: 验证数据 (X, y)
            include_math_validation: 是否包含数学题验证
        
        返回:
            评估指标字典
        """
        # 评估模型
        metrics = self.evaluate(val_data, verbose=True)
        
        # 数学题三步验证
        if include_math_validation and self.image_paths is not None:
            math_metrics = self.evaluate_math_captcha(val_data, verbose=True)
            if math_metrics is not None:
                metrics['math_validation'] = math_metrics
        
        # 显示预测示例
        self.show_prediction_examples(val_data, num_examples=10, show_math_validation=include_math_validation)
        
        return metrics
