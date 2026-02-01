#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版验证码生成脚本
移除数学题类型，只生成可训练的3种类型
"""

import os
import argparse
from generate_captcha import CaptchaGenerator


def main():
    """
    生成训练用验证码（移除数学题类型）
    """
    parser = argparse.ArgumentParser(description='生成验证码训练集（优化版）')
    parser.add_argument('--count', type=int, default=20000, help='生成数量')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'img')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(" " * 20 + "验证码生成器（优化版）")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"目标数量: {args.count}")
    print()
    print("⚠️  已移除数学题类型（避免训练时标签不匹配）")
    print("✓  生成类型: 纯数字、纯字母、混合模式")
    print("=" * 80)
    print()
    
    # 只生成3种类型（移除math类型）
    types_config = [
        ('digit', '纯数字', 0.33),
        ('alpha', '纯字母', 0.33),
        ('mixed', '数字+字母混合', 0.34),
    ]
    
    total_generated = 0
    
    for captcha_type, type_name, ratio in types_config:
        count_for_type = int(args.count * ratio)
        
        print(f"【{type_name}】正在生成 {count_for_type} 张...")
        
        generator = CaptchaGenerator(
            width=200,
            height=60,
            mode='pil',
            captcha_type=captcha_type
        )
        
        for i in range(count_for_type):
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i+1}/{count_for_type}")
            
            try:
                image, text, answer, filename = generator.generate_captcha(save_path=output_dir)
                total_generated += 1
            except Exception as e:
                print(f"  生成失败: {e}")
                continue
        
        print(f"  ✓ 完成 {count_for_type} 张")
        print()
    
    print("=" * 80)
    print(f"✅ 完成！共生成 {total_generated} 张验证码图片")
    print(f"📁 保存位置: {output_dir}")
    print("=" * 80)
    print()
    print("💡 验证码类型分布:")
    print(f"  • 纯数字: ~{int(args.count * 0.33)} 张 (33%)")
    print(f"  • 纯字母: ~{int(args.count * 0.33)} 张 (33%)")
    print(f"  • 混合模式: ~{int(args.count * 0.34)} 张 (34%)")
    print()
    print("⚠️  数学题类型已移除（原因：文件名是答案，图片是问题，导致标签不匹配）")
    print("=" * 80)


if __name__ == '__main__':
    main()
