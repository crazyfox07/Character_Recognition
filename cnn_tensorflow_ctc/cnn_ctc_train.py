# -*- coding:utf-8 -*-
"""
File Name: cnn_ctc
Version:
Description:
Author: liuxuewen
Date: 2018/1/16 15:43
"""
import tensorflow as tf
from math import ceil
import numpy as np
from utils.img_handle import HEIGHT, WIDTH, characters, get_next_batch, decode_sparse_tensor, img2array

num_classes=len(characters)+1+1# char_list + blank + ctc blank

keep_prob=tf.placeholder(tf.float32)
inputs = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH])
# 定义ctc_loss需要的稀疏矩阵
targets = tf.sparse_placeholder(tf.int32)
# 1维向量 序列长度 [batch_size,]
seq_len = tf.placeholder(tf.int32, [None])
learning_rate = 0.001
MODELS_PATH='model_cnn_ctc'

IMAGE_HEIGHT=HEIGHT
IMAGE_WIDTH=WIDTH
w_alpha = 0.1
b_alpha = 0.1
batch_s=32
#第一个卷积层的隐藏点数
HIDDEN_NUM=32

#卷积层
def conv2d(input_layer,w_shape):
    w = tf.Variable(w_alpha * tf.random_normal(w_shape))
    b = tf.Variable(b_alpha * tf.random_normal([w_shape[-1]]))
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_layer, w, strides=[1, 1, 1, 1], padding='SAME'), b))
    conv = tf.nn.dropout(conv, keep_prob=keep_prob)
    return conv

#全连接层
def Dense(input_layer,w_shape):
    W = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[w_shape[-1]]), name="b")
    logits = tf.matmul(input_layer, W) + b
    return logits

def cnn_model(batch_size=batch_s):
    # 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，第2、3维对应图片的宽和高，最后一维对应颜色通道的数目。
    x = tf.reshape(inputs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_shape前两维是patch的大小，第三维时输入通道的数目，最后一维是输出通道的数目。我们对每个输出通道加上了偏置(bias)
    # 第一层
    conv1 = conv2d(input_layer=x, w_shape=[3, 3, 1, HIDDEN_NUM])
    layer1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    # layer1=conv1
    # 第二层
    conv2 = conv2d(input_layer=layer1, w_shape=[3, 3, HIDDEN_NUM, HIDDEN_NUM * 2])
    layer2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    # 第三层
    conv3 = conv2d(input_layer=layer2, w_shape=[3, 3, HIDDEN_NUM * 2, HIDDEN_NUM * 4])
    layer3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第四层
    conv4 = conv2d(input_layer=layer3, w_shape=[3, 3, HIDDEN_NUM * 4, HIDDEN_NUM * 8])
    layer4=tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # layer4 = conv4

    conv_reshape = tf.reshape(layer4, [-1, HIDDEN_NUM * 8])   #shape=[batch_size*(HEIGHT/4)*(WIDTH/4),HIDDEN_NUM * 8]
    logits=Dense(input_layer=conv_reshape,w_shape=[HIDDEN_NUM * 8,num_classes])#shape=[batch_size*(HEIGHT/4)*(WIDTH/4),num_classes]
    logits = tf.reshape(logits, [batch_size, -1, num_classes])#仿照lstm的输出
    # 转置矩阵，第0和第1列互换位置=>[max_timesteps,batch_size,num_classes]
    logits = tf.transpose(logits, (1, 0, 2))
    return logits

#因为用了n次池化，所以高度和宽度都除以n次2
seq_length=ceil(HEIGHT/16)*ceil(WIDTH/16)

def train():
    logits = cnn_model(batch_size=batch_s)
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(MODELS_PATH)
        if checkpoint:
            saver.restore(sess, checkpoint)  # 从模型中读取数据，可以充分利用之前的经验
            current_step = int(checkpoint.split('-')[-1])
            print(checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
            current_step=0
        step=0
        while True:
            step+=1
            train_inputs, train_targets = get_next_batch(batch_size=batch_s)

            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: [seq_length]*batch_s,
                        keep_prob:0.8}


            train_cost,_=sess.run([cost,optimizer],feed_dict=val_feed)

            if step % 10 == 0:
                train_cost = sess.run([cost], feed_dict=val_feed)
                msg = "step {}, cost: {}".format(step, train_cost)
                print(msg)

            if step%10000==0:
                saver.save(sess,'{}/cnn_ctc'.format(MODELS_PATH),global_step=step+current_step)

def test(batch_size = 30):
    logits = cnn_model(batch_size=batch_size)
    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(MODELS_PATH)
        print(checkpoint)
        if checkpoint:
            saver.restore(sess, checkpoint)  # 从模型中读取数据
            print(checkpoint)
        else:
            print('no model')
            return

        test_inputs, test_targets = get_next_batch(batch_size=batch_size)

        val_feed = {inputs: test_inputs,
                    targets: test_targets,
                    seq_len: [seq_length]*batch_size,
                    keep_prob: 1}
        #train_decoded的类型为sparse
        test_decoded = sess.run(decoded[0], feed_dict=val_feed)

        predict=decode_sparse_tensor(test_decoded)
        real = decode_sparse_tensor(test_targets)
        result=list(map(lambda x,y:(x,y),predict,real))
        print(result)


if __name__ == '__main__':
    # train()
    test()
