#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练回调模块（参考captcha_trainer模块化设计）
功能：定义所有训练回调类，确保功能单一性
"""

import os
import numpy as np
from tensorflow import keras
from . import utils


class DelayedEarlyStopping(keras.callbacks.EarlyStopping):
    """
    延迟早停回调：在指定轮次之前不触发早停
    
    参考：captcha_trainer的模块化设计思路
    用途：前期充分训练，后期启用早停监控
    """
    def __init__(self, start_epoch=85, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch
        self.delayed_mode = True  # 标记是否处于延迟模式
    
    def on_epoch_end(self, epoch, logs=None):
        # 只在达到start_epoch后才调用父类的早停逻辑
        if epoch >= self.start_epoch - 1:  # epoch从0开始，第85轮时epoch=84
            if self.delayed_mode:
                # 第一次启用早停时，打印提示信息
                print(f"\n⏰ 已达到第{self.start_epoch}轮，启用早停监控（耐心值: {self.patience}轮）")
                self.delayed_mode = False
            # 调用父类的早停逻辑
            super().on_epoch_end(epoch, logs)
        # 前85轮完全跳过早停检查


class BestFullMatchCheckpoint(keras.callbacks.Callback):
    """
    保存最佳完整匹配准确率模型
    
    参考：captcha_trainer/validation.py的准确率计算
    用途：跟踪并保存完整验证码匹配准确率最高的模型
    """
    def __init__(self, val_data, model_dir, check_interval=5):
        """
        参数:
            val_data: 验证数据 (X, y)
            model_dir: 模型保存目录
            check_interval: 检查间隔（每N轮计算一次）
        """
        super().__init__()
        self.val_images, self.val_labels = val_data
        self.best_full_match_acc = 0
        self.model_dir = model_dir
        self.check_interval = check_interval
    
    def on_epoch_end(self, epoch, logs=None):
        # 每check_interval轮计算一次完整匹配准确率
        if (epoch + 1) % self.check_interval != 0:
            return
        
        # 随机采样验证样本
        sample_size = min(2000, len(self.val_images))
        indices = np.random.choice(len(self.val_images), sample_size, replace=False)
        sample_images = self.val_images[indices]
        sample_labels = self.val_labels[indices]
        
        # 预测并计算准确率
        predictions = self.model.predict(sample_images, verbose=0)
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in sample_labels]
        full_match_acc = utils.calculate_accuracy(true_texts, pred_texts)
        
        # 保存最佳模型
        if full_match_acc > self.best_full_match_acc:
            self.best_full_match_acc = full_match_acc
            save_path = os.path.join(self.model_dir, 'best_full_match_model.keras')
            self.model.save(save_path)
            print(f"  ⭐ 完整匹配准确率提升至 {full_match_acc*100:.2f}%，模型已保存！")


class TrainingProgress(keras.callbacks.Callback):
    """
    训练进度监控回调
    
    参考：captcha_trainer/trains.py的训练日志
    用途：每轮打印详细的训练指标和完整匹配准确率
    """
    def __init__(self, val_data, sample_size=1000):
        """
        参数:
            val_data: 验证数据 (X, y)
            sample_size: 采样大小
        """
        super().__init__()
        self.val_images, self.val_labels = val_data
        self.sample_size = sample_size
        self.best_full_match_acc = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', 0)
        val_binary_acc = logs.get('val_binary_accuracy', 0)
        
        # 获取当前学习率（兼容不同Keras版本）
        try:
            current_lr = float(self.model.optimizer.learning_rate.numpy())
        except:
            try:
                import tensorflow.keras.backend as K
                current_lr = float(K.get_value(self.model.optimizer.lr))
            except:
                current_lr = 0.001  # 默认值
        
        # 计算完整匹配准确率（采样加快速度）
        sample_size = min(self.sample_size, len(self.val_images))
        indices = np.random.choice(len(self.val_images), sample_size, replace=False)
        sample_images = self.val_images[indices]
        sample_labels = self.val_labels[indices]
        
        predictions = self.model.predict(sample_images, verbose=0)
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in sample_labels]
        full_match_acc = utils.calculate_accuracy(true_texts, pred_texts)
        
        # 打印训练进度
        print(f"\n[Epoch {epoch+1}] 训练损失: {logs.get('loss', 0):.4f} | "
              f"验证损失: {val_loss:.4f} | "
              f"二进制准确率: {val_binary_acc:.4f} | "
              f"完整匹配: {full_match_acc*100:.2f}% | "
              f"学习率: {current_lr:.6f}")
        
        # 跟踪最佳完整匹配准确率
        if full_match_acc > self.best_full_match_acc:
            self.best_full_match_acc = full_match_acc
            print(f"    ⬆ 完整匹配准确率提升！当前: {full_match_acc*100:.2f}% "
                  f"(历史最佳: {self.best_full_match_acc*100:.2f}%)")


class StepBasedCallbacks(keras.callbacks.Callback):
    """
    Step-based训练策略（参考captcha_trainer/trains.py）
    
    核心功能：
    - 每save_step步保存checkpoint
    - 每validation_steps步验证
    - 多条件终止：accuracy AND loss AND steps
    - 自动清理旧checkpoint，只保留最近N个
    
    参考：captcha_trainer/trains.py的achieve_cond逻辑
    """
    def __init__(self, val_data, model_dir, save_step=100, validation_steps=500,
                 end_acc=0.95, end_loss=0.01, max_steps=50000, max_checkpoints=5):
        """
        参数:
            val_data: 验证数据 (X, y)
            model_dir: 模型保存目录
            save_step: 保存间隔（步）
            validation_steps: 验证间隔（步）
            end_acc: 目标准确率
            end_loss: 目标损失
            max_steps: 最大步数
            max_checkpoints: 最多保留的checkpoint数量（默认5个）
        """
        super().__init__()
        self.val_images, self.val_labels = val_data
        self.model_dir = model_dir
        self.save_step = save_step
        self.validation_steps = validation_steps
        self.end_acc = end_acc
        self.end_loss = end_loss
        self.max_steps = max_steps
        self.max_checkpoints = max_checkpoints
        self.current_step = 0
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.checkpoint_files = []  # 记录已保存的checkpoint文件
    
    def on_batch_end(self, batch, logs=None):
        """每个batch结束时调用"""
        self.current_step += 1
        logs = logs or {}
        
        # 每save_step步保存checkpoint
        if self.current_step % self.save_step == 0:
            checkpoint_path = os.path.join(self.model_dir, f'checkpoint_step_{self.current_step}.keras')
            self.model.save(checkpoint_path)
            print(f"\n  💾 Step {self.current_step}: 保存checkpoint (loss={logs.get('loss', 0):.4f})")
            
            # 记录checkpoint文件
            self.checkpoint_files.append(checkpoint_path)
            
            # 清理旧checkpoint，只保留最近的N个
            if len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                try:
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                        print(f"  🗑️  删除旧checkpoint: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"  ⚠️  删除checkpoint失败: {e}")
        
        # 每validation_steps步验证
        if self.current_step % self.validation_steps == 0:
            self._validate_and_check_termination()
    
    def _validate_and_check_termination(self):
        """
        执行验证并检查终止条件
        参考：captcha_trainer/trains.py的验证策略
        """
        # 采样1000个验证样本
        sample_size = min(1000, len(self.val_images))
        indices = np.random.choice(len(self.val_images), sample_size, replace=False)
        sample_images = self.val_images[indices]
        sample_labels = self.val_labels[indices]
        
        # 计算验证损失和准确率
        val_results = self.model.evaluate(sample_images, sample_labels, verbose=0)
        val_loss = val_results[0]
        val_binary_acc = val_results[1]
        
        # 计算完整匹配准确率
        predictions = self.model.predict(sample_images, verbose=0)
        pred_texts = [utils.vector_to_text(pred) for pred in predictions]
        true_texts = [utils.vector_to_text(label) for label in sample_labels]
        full_match_acc = utils.calculate_accuracy(true_texts, pred_texts)
        
        # 获取当前学习率
        current_lr = self._get_current_lr()
        
        # 打印验证结果
        print(f"\n  📊 Step {self.current_step} 验证结果:")
        print(f"      验证损失: {val_loss:.4f} | 二进制准确率: {val_binary_acc:.4f}")
        print(f"      完整匹配: {full_match_acc*100:.2f}% | 学习率: {current_lr:.6f}")
        
        # 更新最佳指标
        if full_match_acc > self.best_val_acc:
            self.best_val_acc = full_match_acc
            print(f"      ⬆ 最佳完整匹配准确率: {self.best_val_acc*100:.2f}%")
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"      ⬇ 最佳验证损失: {self.best_val_loss:.4f}")
        
        # 多条件终止检查（参考trains.py的achieve_cond）
        if self._should_terminate(full_match_acc, val_loss):
            self._print_termination_info(full_match_acc, val_loss)
            self.model.stop_training = True
    
    def _get_current_lr(self):
        """获取当前学习率"""
        try:
            return float(self.model.optimizer.learning_rate(self.current_step))
        except:
            try:
                return float(self.model.optimizer.learning_rate.numpy())
            except:
                return 0.001
    
    def _should_terminate(self, full_match_acc, val_loss):
        """
        判断是否应该终止训练
        参考：captcha_trainer/trains.py的achieve_cond
        """
        achieve_accuracy = full_match_acc >= self.end_acc
        achieve_loss = val_loss <= self.end_loss
        achieve_steps = self.current_step >= 10000  # 至少训练10000步
        over_max_steps = self.current_step > self.max_steps
        
        return (achieve_accuracy and achieve_loss and achieve_steps) or over_max_steps
    
    def _print_termination_info(self, full_match_acc, val_loss):
        """打印终止信息"""
        achieve_accuracy = full_match_acc >= self.end_acc
        achieve_loss = val_loss <= self.end_loss
        achieve_steps = self.current_step >= 10000
        over_max_steps = self.current_step > self.max_steps
        
        print(f"\n  🎯 满足终止条件:")
        print(f"      准确率达标: {achieve_accuracy} (>={self.end_acc:.2%})")
        print(f"      损失达标: {achieve_loss} (<={self.end_loss:.4f})")
        print(f"      步数达标: {achieve_steps} (>={10000})")
        print(f"      或超过最大步数: {over_max_steps} (>{self.max_steps})")
        print(f"\n  ✅ 提前终止训练！")


def create_callbacks(model_dir, log_dir, val_data, 
                     use_step_based=True, use_early_stopping=False,
                     checkpoint_save_step=500, validation_steps=500,
                     max_checkpoints_keep=5, end_acc=0.85, max_steps=150000):
    """
    创建训练回调函数（模块化设计）
    
    参考：captcha_trainer的模块化回调设计
    功能：根据配置组装所需的回调
    
    参数:
        model_dir: 模型保存目录
        log_dir: 日志保存目录
        val_data: 验证数据 (X, y)
        use_step_based: 是否使用step-based策略
        use_early_stopping: 是否使用早停（不建议与step-based同时使用）
        checkpoint_save_step: checkpoint保存间隔（步）- 默认500步（避免磁盘占满）
        validation_steps: 验证间隔（步）- 默认500步
        max_checkpoints_keep: 最多保留的checkpoint数量（默认5个）
        end_acc: 目标准确率（默认0.85即85%）
        max_steps: 最大训练步数（默认150000）
    
    返回:
        回调函数列表
    """
    import time
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = []
    
    # 1. 模型检查点：保存最优模型
    checkpoint_path = os.path.join(model_dir, 'best_model.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 2. TensorBoard：可视化训练过程
    tensorboard_log_dir = os.path.join(log_dir, f'run_{time.strftime("%Y%m%d_%H%M%S")}')
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    
    # 3. Step-based策略（参考captcha_trainer/trains.py）
    if use_step_based:
        step_based = StepBasedCallbacks(
            val_data=val_data,
            model_dir=model_dir,
            save_step=checkpoint_save_step,  # 使用配置的保存间隔
            validation_steps=validation_steps,
            end_acc=end_acc,  # 使用传入的目标准确率
            end_loss=0.05,
            max_steps=max_steps,  # 使用传入的最大步数
            max_checkpoints=max_checkpoints_keep  # 只保留N个checkpoint
        )
        callbacks.append(step_based)
        print(f"✓ 启用Step-based训练策略（每{validation_steps}步验证，每{checkpoint_save_step}步保存，保留{max_checkpoints_keep}个checkpoint）")
        print(f"  目标准确率: {end_acc:.1%} | 最大步数: {max_steps}")
    
    # 4. 早停（可选，不建议与step-based同时使用）
    if use_early_stopping and not use_step_based:
        early_stop = DelayedEarlyStopping(
            start_epoch=50,
            monitor='val_loss',
            mode='min',
            patience=35,
            verbose=1,
            restore_best_weights=True,
            min_delta=0.00005
        )
        callbacks.append(early_stop)
        print("✓ 启用延迟早停策略（第50轮后监控）")
    
    # 5. 最佳完整匹配模型保存
    if val_data is not None:
        best_match = BestFullMatchCheckpoint(
            val_data=val_data, 
            model_dir=model_dir,
            check_interval=5
        )
        callbacks.append(best_match)
    
    # 6. 训练进度监控
    if val_data is not None:
        progress = TrainingProgress(val_data=val_data, sample_size=1000)
        callbacks.append(progress)
    
    return callbacks
