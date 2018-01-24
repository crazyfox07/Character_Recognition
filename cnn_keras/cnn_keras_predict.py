# -*- coding:utf-8 -*-
"""
File Name: cnn_keras_predict
Version:
Description:
Author: liuxuewen
Date: 2018/1/19 9:46
"""
from keras.models import load_model

from cnn_keras_train import model_path
from util.img_handle import img2array, HEIGHT, WIDTH, label_len, n_class, num2char, path
import numpy as np
import os

print(model_path)
print(path)
model_path2='model\cnn_resnet.h5'
model=load_model(model_path)

c={'ne':0}
def predict(img_path=r'D:\project\图像识别\Character_Recognition\img_simple\01234_NGPQ.png'):
    img=img2array(img_path)
    img=np.reshape(img,newshape=(1,HEIGHT,WIDTH,1))
    pred=model.predict(img,batch_size=1)
    pred=np.reshape(pred,newshape=(label_len,n_class))
    pred=np.argmax(pred, axis=-1)
    pred=''.join([num2char[num] for num in pred])
    real=img_path.split('_')[-1].replace('.png','')
    if real!=pred:
        c['ne']+=1
    print(img_path.split('_')[-1],pred)


if __name__ == '__main__':
    # path='D:\project\图像识别\Character_Recognition\img'
    items=os.listdir(path)[0:100]
    for item in items:
        img_path=os.path.join(path,item)
        predict(img_path)
    print(c)