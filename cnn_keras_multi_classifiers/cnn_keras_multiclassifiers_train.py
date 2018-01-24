# -*- coding:utf-8 -*-
"""
File Name: cnn_keras_train
Version:
Description:
Author: liuxuewen
Date: 2018/1/18 15:01
"""
import time
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
from utils.img_handle import HEIGHT, WIDTH, label_len, n_class, get_next_batch, num2char

model_path = 'multi_classifiers.h5'

HIDDEN_NUM=32
kernel_size=(3,3)
pool_size=(2,2)


def cnn_model():
    input_tensor=Input(shape=(HEIGHT,WIDTH,1))
    x=input_tensor

    #第一层卷积
    x=Conv2D(HIDDEN_NUM,kernel_size,strides=(1,1),padding='same',activation='relu')(x)
    x=MaxPooling2D(pool_size,padding='valid')(x)
    # 第二层卷积
    x = Conv2D(HIDDEN_NUM*2, kernel_size, strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size, padding='valid')(x)
    # 第三层卷积
    x = Conv2D(HIDDEN_NUM*4, kernel_size, strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size, padding='valid')(x)

    x=Flatten()(x)
    x = Dropout(0.25)(x)
    x=[Dense(n_class, activation='softmax', name='c{}'.format(i+1))(x) for i in range(label_len)]
    model = Model(inputs=[input_tensor], output=x)
    return model

def train():
    # 编译模型
    # if os.path.exists(model_path):
    #     model = load_model(model_path)
    # else:
    #     model = cnn_model()
    model=cnn_model()
    model.compile(loss='categorical_crossentropy',
                  # optimizer='adadelta',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    step = 1
    while True:
        X_train, Y_train = get_next_batch(batch_size=32)

        # print('x_train shape: {}，y_train shape: {}'.format(X_train.shape,Y_train[0].shape))
        # 训练模型
        outputs = model.train_on_batch(X_train, Y_train)
        # print('step', step, outputs)
        if step % 10 == 0:
            # 评估模型
            X_test, Y_test = get_next_batch(batch_size=32)
            score = model.evaluate(X_test, Y_test, verbose=0)
            print('step', step, 'score:', score)

            if min(score[-label_len:]) > 0.95:
                print('step', step, 'score:', score, 'over')
                model.save(model_path)
                break

        if step % 6000 == 0:
            model.save(model_path)

        if step == 30001:
            break
        step += 1

def test():
    batch_size = 32
    model = load_model(model_path)
    X_test, Y_test = get_next_batch(batch_size=batch_size)
    pred = model.predict(X_test, batch_size=batch_size)

    pred = np.argmax(np.array(pred), axis=-1)#pred转换尾numpy后shape=(label_len,batch_size, n_class)
    real = np.argmax(np.array(Y_test), axis=-1)
    for i in range(batch_size):
        pre = pred[:, i]
        rea = real[:, i]
        p = ''.join([num2char[num] for num in pre])
        r = ''.join([num2char[num] for num in rea])
        print('pred:{},real:{}'.format(p, r))



if __name__ == '__main__':
    begin = time.time()
    #train()
    test()
    # cnn_model()
    # img2array()
    end = time.time()
    print("time use: {}".format(end - begin))