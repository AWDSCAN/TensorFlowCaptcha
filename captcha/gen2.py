from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import os
import hashlib

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


def random_captcha_text(char_set=None, captcha_size=None):
    if char_set is None:
        char_set = number + alphabet + ALPHABET
    if captcha_size is None:
        captcha_size = random.randint(4, 6)  # 随机长度从4到6
    captcha_text = random.choices(char_set, k=captcha_size)
    return captcha_text


def gen_captcha_text_and_image():
    # 随机选择难度参数，例如图片宽度和高度
    width, height = random.randint(100, 200), random.randint(50, 100)
    image = ImageCaptcha(width=width, height=height)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    hash_obj = hashlib.sha256(captcha_text.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    filename = f"{captcha_text}_{hash_hex}.jpg"
    file_path = os.path.join("dist", filename)

    captcha = image.generate(captcha_text)
    # 写到文件
    with open(file_path, 'wb') as f:
        f.write(captcha.getbuffer())

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    if not os.path.exists('dist'):
        os.makedirs('dist')

    for _ in range(10):
        gen_captcha_text_and_image()
