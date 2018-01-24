# -*- coding:utf-8 -*-
"""
File Name: cnn_keras_predict
Version:
Description:
Author: liuxuewen
Date: 2018/1/18 16:50
"""
import numpy as np

from cnn_keras_train import model_path
from utils.img_handle import num2char, img2array, HEIGHT,WIDTH
from keras.models import load_model

model = load_model(model_path)


def predict(img_path='utils/01234_NGPQ.png'):
    x_img=img2array(img_path)
    x_img=np.reshape(x_img,newshape=(1,HEIGHT,WIDTH,1))
    pred = model.predict(x_img,batch_size=1)
    pred = np.argmax(np.array(pred), axis=-1)#pred转换尾numpy后shape=(label_len,batch_size, n_class)
    pre = pred[:, 0]
    result = ''.join([num2char[num] for num in pre])
    print('result:{}'.format(result))


if __name__ == '__main__':
    predict()