# -*- coding:utf-8 -*-


import string
from PIL import Image, ImageFont, ImageDraw
import random
import  os

chars=string.ascii_uppercase
digits=string.digits

text = "hello"
path='D:\project\图像识别\Character_Recognition\img_simple'

def gen_img(num=10000):
    for i in range(num):
        text=''.join(random.sample(chars,4))
        img_name='{}_{}.png'.format(''.join(random.sample(digits,5)),text)
        img_path=os.path.join(path,img_name)
        im = Image.new("RGB", (160, 60), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype('C:\Windows\Fonts\msyh.ttf',32)
        dr.text((10, 5), text, font=font, fill="#000000")
        im.save(img_path)


if __name__ == '__main__':
    gen_img()
