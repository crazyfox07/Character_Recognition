# -*- coding:utf-8 -*-
"""
File Name: gen_img
Version:
Description:
Author: liuxuewen
Date: 2018/1/18 15:02
"""
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import os
characters = string.ascii_uppercase

height,width, n_len, n_class =  60, 160,4, len(characters)

path=r'D:\project\图像识别\Character_Recognition\img'
def gen_img(num=100):
    for i in range(num):
        text=''.join(random.sample(characters,n_len))
        generator = ImageCaptcha(width=width, height=height)
        img = generator.generate_image(text)
        img_name='{}_{}.png'.format(''.join(random.sample(string.digits,5)),text)
        img_path=os.path.join(path,img_name)
        img.save(img_path)
        if i%1000==0:
            print(i,text)


if __name__ == '__main__':
    gen_img(10000)