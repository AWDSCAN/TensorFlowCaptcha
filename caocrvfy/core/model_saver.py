#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型保存模块
功能：完整保存模型，包括.keras格式和checkpoint格式
"""

import os
import tensorflow as tf
from tensorflow import keras


class ModelSaver:
    """
    模型保存器
    支持两种格式：
    1. .keras格式（完整模型）
    2. checkpoint格式（权重+元数据）
    """
    
    def __init__(self, model_dir):
        """
        初始化模型保存器
        
        参数:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_complete_model(self, model, model_name='crack_captcha_model'):
        """
        保存完整模型（.keras格式 + checkpoint格式）
        
        生成文件：
        - crack_captcha_model.keras  （完整模型）
        - checkpoint                  （checkpoint元数据）
        - ckpt-1.index               （变量索引）
        - ckpt-1.data-00000-of-00001 （变量数据）
        
        参数:
            model: Keras模型
            model_name: 模型名称（不含扩展名）
        
        返回:
            保存的文件路径列表
        """
        saved_files = []
        
        # 1. 保存为.keras格式（TensorFlow 2.x推荐格式）
        keras_path = os.path.join(self.model_dir, f'{model_name}.keras')
        model.save(keras_path)
        saved_files.append(keras_path)
        print(f"✓ 已保存 .keras 格式模型: {keras_path}")
        
        # 获取文件大小
        keras_size = os.path.getsize(keras_path) / (1024 ** 2)
        print(f"  文件大小: {keras_size:.2f} MB")
        
        # 2. 保存为checkpoint格式（兼容TensorFlow 1.x）
        ckpt_prefix = os.path.join(self.model_dir, 'ckpt')
        
        # 创建checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        ckpt_save_path = checkpoint.save(ckpt_prefix)
        
        print(f"✓ 已保存 checkpoint 格式:")
        
        # 列出生成的checkpoint文件
        checkpoint_files = []
        for filename in os.listdir(self.model_dir):
            if filename.startswith('ckpt-'):
                filepath = os.path.join(self.model_dir, filename)
                checkpoint_files.append(filepath)
                saved_files.append(filepath)
                file_size = os.path.getsize(filepath) / (1024 ** 2)
                print(f"  - {filename} ({file_size:.2f} MB)")
        
        # checkpoint元数据文件
        checkpoint_meta = os.path.join(self.model_dir, 'checkpoint')
        if os.path.exists(checkpoint_meta):
            saved_files.append(checkpoint_meta)
            print(f"  - checkpoint (元数据)")
        
        return saved_files
    
    def load_keras_model(self, model_name='crack_captcha_model'):
        """
        加载.keras格式模型
        
        参数:
            model_name: 模型名称
        
        返回:
            加载的模型
        """
        keras_path = os.path.join(self.model_dir, f'{model_name}.keras')
        
        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"模型文件不存在: {keras_path}")
        
        print(f"正在加载模型: {keras_path}")
        model = keras.models.load_model(keras_path)
        print(f"✓ 模型加载成功")
        
        return model
    
    def load_from_checkpoint(self, model, ckpt_name='ckpt-1'):
        """
        从checkpoint恢复模型权重
        
        参数:
            model: 已创建的模型实例
            ckpt_name: checkpoint名称（不含路径）
        
        返回:
            恢复权重后的模型
        """
        ckpt_path = os.path.join(self.model_dir, ckpt_name)
        
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(ckpt_path)
        
        print(f"✓ 已从 checkpoint 恢复权重: {ckpt_path}")
        
        return model
    
    def get_latest_checkpoint(self):
        """
        获取最新的checkpoint路径
        
        返回:
            最新checkpoint的路径，如果不存在返回None
        """
        return tf.train.latest_checkpoint(self.model_dir)
    
    def list_saved_models(self):
        """
        列出所有已保存的模型文件
        
        返回:
            模型文件列表
        """
        model_files = {
            'keras_models': [],
            'checkpoints': []
        }
        
        for filename in os.listdir(self.model_dir):
            filepath = os.path.join(self.model_dir, filename)
            
            if filename.endswith('.keras'):
                file_size = os.path.getsize(filepath) / (1024 ** 2)
                model_files['keras_models'].append({
                    'name': filename,
                    'path': filepath,
                    'size_mb': file_size
                })
            
            elif filename.startswith('ckpt-') or filename == 'checkpoint':
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath) / (1024 ** 2)
                    model_files['checkpoints'].append({
                        'name': filename,
                        'path': filepath,
                        'size_mb': file_size
                    })
        
        return model_files


def save_model_complete(model, model_dir, model_name='crack_captcha_model'):
    """
    便捷函数：保存完整模型
    
    参数:
        model: Keras模型
        model_dir: 保存目录
        model_name: 模型名称
    
    返回:
        保存的文件路径列表
    """
    saver = ModelSaver(model_dir)
    return saver.save_complete_model(model, model_name)


def load_model_from_keras(model_dir, model_name='crack_captcha_model'):
    """
    便捷函数：加载.keras格式模型
    
    参数:
        model_dir: 模型目录
        model_name: 模型名称
    
    返回:
        加载的模型
    """
    saver = ModelSaver(model_dir)
    return saver.load_keras_model(model_name)


if __name__ == '__main__':
    # 测试代码
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core import config
    from extras.model_enhanced import create_enhanced_cnn_model
    
    print("=" * 80)
    print("测试模型保存功能")
    print("=" * 80)
    
    # 创建测试模型
    print("\n1. 创建测试模型...")
    model = create_enhanced_cnn_model()
    
    # 保存模型
    print("\n2. 保存模型...")
    test_dir = os.path.join(config.MODEL_DIR, 'test_save')
    saver = ModelSaver(test_dir)
    saved_files = saver.save_complete_model(model, 'test_model')
    
    print(f"\n共保存 {len(saved_files)} 个文件")
    
    # 列出已保存的模型
    print("\n3. 列出已保存的模型...")
    models = saver.list_saved_models()
    
    print("\nKeras模型:")
    for m in models['keras_models']:
        print(f"  - {m['name']} ({m['size_mb']:.2f} MB)")
    
    print("\nCheckpoint文件:")
    for m in models['checkpoints']:
        print(f"  - {m['name']} ({m['size_mb']:.2f} MB)")
    
    # 测试加载
    print("\n4. 测试加载模型...")
    loaded_model = saver.load_keras_model('test_model')
    print("✓ 模型加载测试成功")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
