# -*- coding:utf-8 -*-
"""
File Name: cnn_,mnist
Version:
Description:
Author: liuxuewen
Date: 2017/12/14 17:19
"""
import numpy as np
from keras import Input
from keras.engine import Model
from keras.optimizers import Adam

from util.img_handle import get_next_batch, characters

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, add
from keras.layers import MaxPooling2D, Conv2D
import string
from util.img_handle import HEIGHT, WIDTH, label_len

# 全局变量
batch_size = 32

epochs = 12

chars = characters
char2num = {char: num for num, char in enumerate(chars)}
num2char = {num: char for num, char in enumerate(chars)}
nb_classes = len(chars)

# input image dime        nsions
input_shape = (HEIGHT, WIDTH, 1)

model_path = r'model\cnn_resnet.h5'

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    #在使用BatchNormalization时一直未收敛，还不知道原因，todo
    # x = BatchNormalization(axis=-1, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    # x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    #直接返回x，不使用残差网络会发现损失值一直不会收敛
    # return x
    #当with_conv_shortcut为True时，可以改变输入维度，在add的时候避免维度冲突
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def cnn_model():
    # 构建模型
    input_tensor = Input(shape=(HEIGHT, WIDTH, 1))
    x = input_tensor
    # conv1
    x = Conv2d_BN(x, nb_filter=nb_filters, kernel_size=kernel_size, strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same')(x)

    # conv2
    x = identity_Block(x, nb_filter=nb_filters * 2, kernel_size=kernel_size, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=nb_filters * 2, kernel_size=kernel_size)
    x = identity_Block(x, nb_filter=nb_filters * 2, kernel_size=kernel_size)
    x = MaxPooling2D(pool_size=pool_size)(x)

    # conv3
    x = identity_Block(x, nb_filter=nb_filters * 4, kernel_size=kernel_size, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=nb_filters * 4, kernel_size=kernel_size)
    x = identity_Block(x, nb_filter=nb_filters * 4, kernel_size=kernel_size)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Flatten()(x)
    x = Dense(label_len * nb_classes, activation='softmax')(x)
    model = Model(input=input_tensor, output=x)
    return model


def train_model():
    model = cnn_model()
    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    for step in range(100000):
        X_train, Y_train = get_next_batch(batch_size=32)
        # 训练模型
        loss, accuracy = model.train_on_batch(X_train, Y_train)
        print(step, loss, accuracy)

        if step % 10 == 0:
            # 评估模型
            X_test, Y_test = get_next_batch()
            score = model.evaluate(X_test, Y_test, verbose=0)
            print('step:', step, 'Test score:', score[0], 'Test accuracy:', score[1])

        if step % 1000 == 0:
            model.save(model_path)


if __name__ == '__main__':
    train_model()
