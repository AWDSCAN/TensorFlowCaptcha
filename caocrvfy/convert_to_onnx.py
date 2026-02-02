#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keras模型转ONNX格式
用于部署到不同平台（如C++、移动端等）

依赖安装：
    pip install tf2onnx onnx onnxruntime

使用方法：
    python convert_to_onnx.py --model models/final_model.keras
    python convert_to_onnx.py --model /data/coding/caocrvfy/core/models/final_model.keras
"""

import os
import sys
import argparse
import tensorflow as tf
import tf2onnx
import onnx

# 添加项目路径以导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_keras_to_onnx(keras_model_path, onnx_model_path=None, opset=13):
    """
    将Keras模型转换为ONNX格式
    
    参数:
        keras_model_path: Keras模型路径（.keras或.h5）
        onnx_model_path: ONNX输出路径（可选，默认与keras同名）
        opset: ONNX操作集版本（默认13，兼容性好）
    
    返回:
        onnx_model_path: 保存的ONNX模型路径
    """
    # 检查Keras模型是否存在
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"❌ Keras模型不存在: {keras_model_path}")
    
    print("=" * 70)
    print("🔄 Keras → ONNX 模型转换")
    print("=" * 70)
    
    # 设置ONNX输出路径
    if onnx_model_path is None:
        base_name = os.path.splitext(keras_model_path)[0]
        onnx_model_path = base_name + '.onnx'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(onnx_model_path) if os.path.dirname(onnx_model_path) else '.', exist_ok=True)
    
    print(f"\n📥 加载Keras模型: {keras_model_path}")
    
    try:
        # 导入自定义对象（如果存在）
        custom_objects = {}
        try:
            from caocrvfy.extras.model_enhanced import WeightedBinaryCrossentropy
            custom_objects['WeightedBinaryCrossentropy'] = WeightedBinaryCrossentropy
            print("   ✓ 已加载自定义损失函数: WeightedBinaryCrossentropy")
        except ImportError:
            pass
        
        try:
            from caocrvfy.extras.focal_loss import FocalLoss
            custom_objects['FocalLoss'] = FocalLoss
            print("   ✓ 已加载自定义损失函数: FocalLoss")
        except ImportError:
            pass
        
        # 加载Keras模型
        if custom_objects:
            keras_model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects, compile=False)
        else:
            keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
        print("   ✓ Keras模型加载成功")
        
        # 显示模型信息
        print(f"\n📊 模型信息:")
        print(f"   输入形状: {keras_model.input_shape}")
        print(f"   输出形状: {keras_model.output_shape}")
        
        # 获取模型参数量
        total_params = keras_model.count_params()
        print(f"   参数量: {total_params:,}")
        
        # 转换为ONNX
        print(f"\n🔄 转换中... (opset={opset})")
        
        # 使用tf2onnx进行转换
        spec = (tf.TensorSpec(keras_model.input_shape, tf.float32, name="input"),)
        
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=spec,
            opset=opset,
            output_path=onnx_model_path
        )
        
        print(f"   ✓ ONNX模型已保存: {onnx_model_path}")
        
        # 验证ONNX模型
        print(f"\n🔍 验证ONNX模型...")
        onnx_model_loaded = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model_loaded)
        print("   ✓ ONNX模型验证通过")
        
        # 显示文件大小对比
        keras_size = os.path.getsize(keras_model_path) / (1024 * 1024)
        onnx_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        
        print(f"\n📦 文件大小对比:")
        print(f"   Keras: {keras_size:.2f} MB")
        print(f"   ONNX:  {onnx_size:.2f} MB")
        print(f"   差异:  {onnx_size - keras_size:+.2f} MB")
        
        # 显示ONNX模型信息
        print(f"\n📋 ONNX模型详情:")
        print(f"   IR版本: {onnx_model_loaded.ir_version}")
        print(f"   Opset版本: {onnx_model_loaded.opset_import[0].version}")
        print(f"   生产者: {onnx_model_loaded.producer_name}")
        
        print("\n" + "=" * 70)
        print("✅ 转换成功！")
        print("=" * 70)
        
        return onnx_model_path
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_onnx_inference(onnx_model_path, test_input_shape=(1, 60, 200, 3)):
    """
    测试ONNX模型推理
    
    参数:
        onnx_model_path: ONNX模型路径
        test_input_shape: 测试输入形状
    """
    import numpy as np
    import onnxruntime as ort
    
    print("\n" + "=" * 70)
    print("🧪 测试ONNX模型推理")
    print("=" * 70)
    
    try:
        # 创建推理会话
        print(f"\n📥 加载ONNX模型: {onnx_model_path}")
        session = ort.InferenceSession(onnx_model_path)
        print("   ✓ 推理会话创建成功")
        
        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"\n📊 模型接口:")
        print(f"   输入名称: {input_name}")
        print(f"   输入形状: {session.get_inputs()[0].shape}")
        print(f"   输出名称: {output_name}")
        print(f"   输出形状: {session.get_outputs()[0].shape}")
        
        # 生成随机测试数据
        print(f"\n🎲 生成测试数据: {test_input_shape}")
        test_input = np.random.rand(*test_input_shape).astype(np.float32)
        
        # 执行推理
        print(f"\n⚡ 执行推理...")
        outputs = session.run([output_name], {input_name: test_input})
        
        print(f"   ✓ 推理成功")
        print(f"   输出形状: {outputs[0].shape}")
        print(f"   输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        # 简单性能测试
        import time
        
        print(f"\n⏱️  性能测试（100次推理）...")
        start_time = time.time()
        
        for _ in range(100):
            session.run([output_name], {input_name: test_input})
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 100 * 1000  # ms
        
        print(f"   平均推理时间: {avg_time:.2f} ms")
        print(f"   FPS: {1000 / avg_time:.1f}")
        
        print("\n" + "=" * 70)
        print("✅ ONNX模型推理测试通过！")
        print("=" * 70)
        
    except ImportError:
        print("\n⚠️  onnxruntime未安装，跳过推理测试")
        print("   安装命令: pip install onnxruntime")
    except Exception as e:
        print(f"\n❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Keras模型转ONNX格式')
    parser.add_argument('--model', type=str, 
                        default='core/models/final_model.keras',
                        help='Keras模型路径 (默认: core/models/final_model.keras)')
    parser.add_argument('--output', type=str, default=None,
                        help='ONNX输出路径 (可选，默认与输入同名)')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset版本 (默认: 13)')
    parser.add_argument('--test', action='store_true',
                        help='转换后测试ONNX推理')
    
    args = parser.parse_args()
    
    try:
        # 转换模型
        onnx_path = convert_keras_to_onnx(
            keras_model_path=args.model,
            onnx_model_path=args.output,
            opset=args.opset
        )
        
        # 可选：测试推理
        if args.test:
            test_onnx_inference(onnx_path)
        
        print(f"\n💡 使用建议:")
        print(f"   1. Python推理: 使用 onnxruntime")
        print(f"   2. C++推理: 使用 ONNX Runtime C++ API")
        print(f"   3. 移动端: 转换为TFLite或CoreML")
        print(f"\n📝 ONNX模型路径: {onnx_path}")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败")
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()
