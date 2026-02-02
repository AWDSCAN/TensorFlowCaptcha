#!/usr/bin/env python
# -*- coding: utf-8 -*-
import binascii

texts = ['2*8=?', '19+3=?', '5*7=?']

print('16进制编解码测试:')
print('='*80)

for text in texts:
    # 编码
    encoded = binascii.hexlify(text.encode('utf-8')).decode('utf-8')
    
    # 解码
    decoded = binascii.unhexlify(encoded.encode('utf-8')).decode('utf-8')
    
    print(f'原文: {text}')
    print(f'  16进制编码: {encoded}')
    print(f'  解码还原: {decoded}')
    print(f'  是否一致: {"✓" if text == decoded else "✗"}')
    print(f'  文件名示例: {encoded}_答案_hash.png')
    print()

print('优势分析:')
print('-'*80)
print('✓ 只包含0-9a-f字符，完全避免特殊字符')
print('✓ 无需填充字符，编解码简单')
print('✓ 跨平台兼容性好，无文件路径问题')
print('✓ 易于阅读和调试（相比base64）')

