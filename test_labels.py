#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'caocrvfy')
from core import utils

# 测试新旧格式
test_files = [
    # 新格式（16进制编码，只包含0-9a-f）
    '322a383d3f_16_abc123.png',      # 2*8=? (16进制编码)
    '31392b333d3f_22_def456.png',    # 19+3=? (16进制编码)
    '352a373d3f_35_xyz789.png',      # 5*7=? (16进制编码)
    # 旧格式（base64，向后兼容测试）
    'Mio0PT8=_8_7f4c.png',           # 2*4=? (标准base64)
    'MTkrMz0/_22_abc.png',           # 19+3=? (标准base64，包含/)
]

print('训练标签验证（支持16进制和base64）:')
print('='*60)
for f in test_files:
    try:
        label = utils.parse_filename(f)
        # 判断是新格式还是旧格式
        if f.startswith(('3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')) and '_' in f:
            format_type = '新格式(16进制)'
        else:
            format_type = '旧格式(base64)'
        print(f'✓ [{format_type}] {f}')
        print(f'  训练标签: {label}')
    except Exception as e:
        print(f'✗ 文件名: {f}')
        print(f'  错误: {e}')
    print()


