# -*- coding:utf-8 -*-
"""
File Name: img_handle
Version:
Description:
Author: liuxuewen
Date: 2018/1/18 15:21
"""
import os
import random

from keras.utils import np_utils
from skimage import io,color
import numpy as np
import string
characters = string.ascii_uppercase
HEIGHT,WIDTH, label_len, n_class =  60, 160,4, len(characters)
char2num={v:k for k,v in enumerate(characters)}
num2char={k:v for k,v in enumerate(characters)}

path=r'D:\project\图像识别\Character_Recognition\img'
def img2array(img_path='ABCJZ_4tkvhd.png'):
    img=io.imread(img_path)
    #大小调整
    # img=transform.resize(img,output_shape=(HEIGHT,WIDTH),mode = 'constant')
    #彩色转灰度
    gray=color.rgb2gray(img)
    #截长补短
    gray_extend = np.ones(shape=(HEIGHT, WIDTH))
    if gray.shape[1] <= WIDTH:
        gray_extend[:, :gray.shape[1]] = gray

    else:
        gray_extend[:, :] = gray[:, :WIDTH]

    #灰度转二值化
    # thresh = filters.threshold_otsu(gray)
    # binary=(gray<thresh)*1.0
    #图片保存
    # io.imsave('binary.png',gray_extend)
    return gray_extend

def label2onehot(label):
    label2index=list()
    for item in label:
        if item in characters:
            label2index.append(char2num[item])
        else:
            label2index.append(0)
    onehot=np_utils.to_categorical(label2index,num_classes=n_class)
    onehot=np.reshape(onehot,newshape=(len(label)*n_class,))
    return onehot

img_all=os.listdir(path)

def get_next_batch(path=path, batch_size=32):
    items = random.sample(img_all, batch_size)
    x_img = np.zeros(shape=(batch_size, HEIGHT,WIDTH))
    y_label = [np.zeros(shape=(batch_size, n_class)) for _ in range(label_len)]
    for i, item in enumerate(items):
        if item.endswith('png'):
            img_path = os.path.join(path, item)
            img_array = img2array(img_path)
            x_img[i] = img_array
            label_text = item.split('_')[-1].replace('.png', '')
            for j, ch in enumerate(label_text):
                y_label[j][i] = label2onehot(ch)
    x_img = x_img.reshape(batch_size,HEIGHT, WIDTH,1)
    return x_img, y_label