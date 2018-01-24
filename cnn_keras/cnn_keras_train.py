# -*- coding:utf-8 -*-
"""
File Name: cnn_,mnist
Version:
Description:
Author: liuxuewen
Date: 2017/12/14 17:19
"""
import numpy as np
from keras.optimizers import Adam

from util.img_handle import get_next_batch
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import  MaxPooling2D,Conv2D
import string
from util.img_handle import HEIGHT,WIDTH, label_len, n_class

# 全局变量
batch_size = 32

epochs = 12

chars=string.ascii_uppercase
char2num={char:num for num,char in enumerate(chars)}
num2char={num:char for num,char in enumerate(chars)}
nb_classes = len(chars)

# input image dimensions
input_shape = (HEIGHT,WIDTH,  1)

model_path=r'model\cnn_Chinese.h5'

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def cnn_model():
    # 构建模型
    model = Sequential()
    """ 
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], 
                            border_mode='same', 
                            input_shape=input_shape)) 
    """
    model.add(Conv2D(nb_filters, kernel_size,
                            padding='same',
                            input_shape=input_shape))  # 卷积层1
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size,padding='valid'))  # 池化层
    # model.add(Dropout(0.2))  # 神经元随机失活

    model.add(Conv2D(nb_filters*2, kernel_size,padding='same'))  # 卷积层2
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size,padding='valid'))  # 池化层
    # model.add(Dropout(0.2))  # 神经元随机失活

    model.add(Conv2D(nb_filters * 4, kernel_size,padding='same'))  # 卷积层3
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
    # model.add(Dropout(0.2))  # 神经元随机失活

    model.add(Flatten())  # 拉成一维数据
    # model.add(Dense(1024))  # 全连接层1
    # model.add(Activation('relu'))  # 激活层
    # model.add(Dropout(0.2))  # 随机失活
    model.add(Dense(label_len*n_class))  # 全连接层2
    model.add(Activation('softmax'))  # Softmax评分
    return model

def train_model():
    model=cnn_model()
    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    for step in range(100000):
        X_train,Y_train=get_next_batch(batch_size=128)
        # 训练模型
        loss, accuracy=model.train_on_batch(X_train,Y_train)
        print(step,loss,accuracy)
        if step%10==0:
            # 评估模型
            X_test, Y_test=get_next_batch()
            score = model.evaluate(X_test, Y_test, verbose=0)
            print('step:',step,'Test score:', score[0],'Test accuracy:', score[1])


        if step%200==0:
            model.save(model_path)

#
#
if __name__ == '__main__':
    train_model()
