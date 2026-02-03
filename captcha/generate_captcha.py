#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复杂验证码图片生成器
包含：数字+大小写字母、字符旋转、大小变换、干扰线、噪点、随机背景色、字符颜色变换等
支持基于captcha库和PIL的双模式生成

数学题命名格式: hex(数学运算题)_运算结果_随机hash.png
例如: 31392b333d3f_22_abc123def456.png 表示 "19+3=?" 答案是 22
注: 使用16进制编码题目，避免特殊字符问题
"""

import os
import random
import string
import time
import hashlib
import platform
import base64  # 保留用于向后兼容旧格式
import binascii
from PIL import Image, ImageDraw, ImageFont, ImageFilter
try:
    from captcha.image import ImageCaptcha
    CAPTCHA_AVAILABLE = True
except ImportError:
    CAPTCHA_AVAILABLE = False
    print("警告: captcha库未安装，将使用PIL模式。可通过 'pip install captcha' 安装")


class CaptchaGenerator:
    """复杂验证码生成器"""
    
    def __init__(self, width=200, height=60, mode='pil', captcha_type='mixed'):
        """
        初始化验证码生成器
        :param width: 验证码宽度
        :param height: 验证码高度
        :param mode: 生成模式 'captcha' 或 'pil'
        :param captcha_type: 验证码类型 'digit'(纯数字), 'alpha'(纯字母), 'mixed'(混合)
        """
        self.width = width
        self.height = height
        self.mode = mode
        self.captcha_type = captcha_type
        
        # 字符集定义
        self.digits = string.digits  # 0-9
        self.alpha_upper = string.ascii_uppercase  # A-Z
        self.alpha_lower = string.ascii_lowercase  # a-z
        self.alpha_all = string.ascii_letters  # A-Z + a-z
        self.charset = string.digits + string.ascii_letters  # 完整字符集
        
        # 初始化captcha库
        if mode == 'captcha' and CAPTCHA_AVAILABLE:
            self.captcha_gen = ImageCaptcha(width=width, height=height)
        elif mode == 'captcha':
            print("captcha库不可用，切换到PIL模式")
            self.mode = 'pil'
    
    def get_system_font(self):
        """自动获取系统字体路径"""
        system = platform.system()
        if system == 'Windows':
            fonts = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
            ]
            for font in fonts:
                if os.path.exists(font):
                    return font
        elif system == 'Linux':
            fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
            for font in fonts:
                if os.path.exists(font):
                    return font
        return None
    
    def get_random_color(self, min_val=0, max_val=255):
        """生成随机颜色"""
        return (
            random.randint(min_val, max_val),
            random.randint(min_val, max_val),
            random.randint(min_val, max_val)
        )
    
    def get_random_text(self, min_len=4, max_len=8):
        """
        根据类型生成随机验证码文本
        新规则：验证码长度统一为4位
        :return: text 验证码文本
        """
        length = 4  # 统一长度为4位
        
        if self.captcha_type == 'digit':
            # 纯数字：4位
            text = ''.join(random.choices(self.digits, k=length))
            return text
            
        elif self.captcha_type == 'mixed':
            # 数字+字母混合：4位
            text = ''.join(random.choices(self.charset, k=length))
            return text
        
        else:
            # 默认混合模式：4位
            text = ''.join(random.choices(self.charset, k=length))
            return text
    
    def generate_hash(self, text):
        """
        生成文件名hash (32位MD5)
        格式：时间戳 + 验证码内容 + 随机数的MD5完整哈希
        """
        timestamp = str(int(time.time() * 1000000))  # 微秒级时间戳
        random_str = str(random.randint(100000, 999999))  # 6位随机数
        content = timestamp + text + random_str
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return hash_obj.hexdigest()  # 返回完整的32位哈希
    
    def generate_filename(self, text):
        """
        生成文件名
        
        格式：验证码内容-32位hash.png
        
        参数:
            text: 验证码文本
        """
        # 普通类型：原有格式
        file_hash = self.generate_hash(text)
        return f"{text}-{file_hash}.png"
    
    def generate_captcha_with_lib(self, text):
        """使用captcha库生成验证码"""
        if not CAPTCHA_AVAILABLE:
            raise RuntimeError("captcha库不可用")
        
        # 生成验证码图片
        image = self.captcha_gen.generate_image(text)
        return image
    
    def generate_captcha_with_pil(self, text):
        """使用PIL生成复杂验证码"""
        # 创建背景（渐变或纯色）
        bg_color = self.get_random_color(230, 255)
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # 绘制底层干扰线（背景层）
        for _ in range(random.randint(6, 10)):
            line_color = self.get_random_color(100, 200)
            draw.line([
                (random.randint(0, self.width), random.randint(0, self.height)),
                (random.randint(0, self.width), random.randint(0, self.height))
            ], fill=line_color, width=random.randint(1, 2))
        
        # 绘制噪点（增加数量）
        for _ in range(random.randint(1000, 1500)):
            draw.point(
                (random.randint(0, self.width), random.randint(0, self.height)),
                fill=self.get_random_color(150, 255)
            )
        
        # 获取系统字体
        font_path = self.get_system_font()
        if not font_path:
            raise RuntimeError("未找到系统字体，请手动指定字体路径")
        
        base_font_size = 36
        
        # 绘制每个字符（独立旋转）
        num_chars = len(text)
        char_spacing = (self.width - 40) // (num_chars + 1)
        
        for i, char in enumerate(text):
            # 随机字体大小
            font_size = base_font_size + random.randint(-3, 3)
            font = ImageFont.truetype(font_path, font_size)
            
            # 创建字符临时图像
            char_img = Image.new('RGBA', (50, 70), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            
            # 随机字符颜色（深色）
            char_color = self.get_random_color(20, 100)
            char_draw.text((10, 15), char, font=font, fill=char_color)
            
            # 随机旋转
            rotation_angle = random.randint(-30, 30)
            char_img = char_img.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0, 0))
            
            # 计算粘贴位置
            x = 20 + i * char_spacing + random.randint(-3, 3)
            y = (self.height - char_img.height) // 2 + random.randint(-5, 5)
            
            # 粘贴字符
            image.paste(char_img, (x, y), char_img)
        
        # 绘制中间层干扰线（穿过字符）
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(4, 7)):
            line_color = self.get_random_color(80, 180)
            # 绘制穿过验证码中间区域的线条
            x1 = random.randint(0, self.width)
            y1 = random.randint(self.height // 4, self.height * 3 // 4)
            x2 = random.randint(0, self.width)
            y2 = random.randint(self.height // 4, self.height * 3 // 4)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.randint(1, 3))
        
        # 绘制顶层干扰线
        for _ in range(random.randint(3, 6)):
            line_color = self.get_random_color(120, 200)
            draw.line([
                (random.randint(0, self.width), random.randint(0, self.height)),
                (random.randint(0, self.width), random.randint(0, self.height))
            ], fill=line_color, width=1)
        
        # 添加随机干扰弧线
        for _ in range(random.randint(2, 4)):
            arc_color = self.get_random_color(100, 190)
            start_angle = random.randint(0, 360)
            end_angle = start_angle + random.randint(30, 120)
            bbox = [
                random.randint(0, self.width // 2),
                random.randint(0, self.height),
                random.randint(self.width // 2, self.width),
                random.randint(0, self.height)
            ]
            try:
                draw.arc(bbox, start_angle, end_angle, fill=arc_color, width=random.randint(1, 2))
            except:
                pass
        
        # 应用模糊滤镜
        if random.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
        
        return image
    
    def generate_captcha(self, text=None, save_path=None):
        """
        生成验证码图片
        :param text: 验证码文本，如果为None则随机生成
        :param save_path: 保存路径，如果为None则不保存
        :return: (image, text, filename) 图片对象、验证码文本和文件名
        """
        # 获取验证码文本
        if text is None:
            text = self.get_random_text()
        
        # 根据模式生成验证码
        if self.mode == 'captcha':
            image = self.generate_captcha_with_lib(text)
        else:
            image = self.generate_captcha_with_pil(text)
        
        # 生成文件名
        filename = self.generate_filename(text)
        
        # 保存图片
        if save_path:
            # 如果save_path是目录，则拼接文件名
            if os.path.isdir(save_path):
                filepath = os.path.join(save_path, filename)
            else:
                filepath = save_path
            image.save(filepath)
        
        return image, text, filename


if __name__ == '__main__':
    """
    主函数：生成所有类型的验证码测试图片
    图片保存在 captcha/img 目录下
    """
    import os
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'img')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(" " * 25 + "验证码生成器")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print()
    
    # 生成各类型验证码（统一4位长度）
    types_config = [
        ('digit', '纯数字', 5),
        ('mixed', '数字+字母混合', 5),
    ]
    
    total = 0
    for captcha_type, type_name, count in types_config:
        print(f"【{type_name}】正在生成 {count} 张...")
        
        generator = CaptchaGenerator(
            width=200,
            height=60,
            mode='pil',
            captcha_type=captcha_type
        )
        
        for i in range(count):
            image, text, filename = generator.generate_captcha(save_path=output_dir)
            total += 1
            print(f"  [{i+1}/{count}] {filename:<35} | 内容: {text}")
        print()
    
    print("=" * 80)
    print(f"✅ 完成！共生成 {total} 张验证码图片")
    print(f"📁 保存位置: {output_dir}")
    print("=" * 80)
    print()
    print("💡 验证码类型说明:")
    print("  • 纯数字: 仅包含0-9（统一4位长度）")
    print("  • 混合模式: 数字+字母组合（统一4位长度，带强干扰）")
    print("=" * 80)
