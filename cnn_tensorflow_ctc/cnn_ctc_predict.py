# -*- coding:utf-8 -*-
"""
File Name: cnn_ctc_predict
Version:
Description:
Author: liuxuewen
Date: 2018/1/18 14:45
"""
import tensorflow as tf

from cnn_ctc_train import cnn_model, seq_len, MODELS_PATH, inputs, seq_length, keep_prob
from utils.img_handle import img2array, HEIGHT, WIDTH, decode_sparse_tensor
import numpy as np


sess=tf.Session()
logits = cnn_model(batch_size=1)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

checkpoint = tf.train.latest_checkpoint(MODELS_PATH)
saver = tf.train.Saver()
saver.restore(sess, checkpoint)


def predict(img_path='utils/ABCJZ_4tkvhd.png'):
    img=img2array(img_path=img_path)
    predict_inputs=np.reshape(img,newshape=[1,HEIGHT,WIDTH])
    val_feed = {inputs: predict_inputs,
                seq_len: [seq_length] * 1,
                keep_prob: 1}
    predict_decoded = sess.run(decoded[0], feed_dict=val_feed)
    result = decode_sparse_tensor(predict_decoded)
    print(result)

if __name__ == '__main__':
    for i in range(10):
        predict()