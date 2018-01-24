# -*- coding:utf-8 -*-
"""
File Name: img_handle
Version:
Description:
Author: liuxuewen
Date: 2018/1/16 16:02
"""
from skimage import io,transform,color,filters
from scipy import misc
import random
import os
import numpy as np
import string

characters = string.digits+string.ascii_lowercase
char2num = {v: k for k, v in enumerate(characters)}
num2char = {k: v for k, v in enumerate(characters)}


HEIGHT=60
WIDTH=160
img_path=r'D:\tmp\tmp\tmp'
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

img_list=os.listdir(img_path)
def get_next_batch(batch_size=1):
    img_chosen=random.sample(img_list,batch_size)
    x_img=np.zeros(shape=(batch_size,HEIGHT,WIDTH))
    text_list = list()

    for i,img_name in enumerate(img_chosen):
        text = img_name.split('_')[-1].replace(".png", "")
        img= img2array(os.path.join(img_path, img_name))
        x_img[i] = img
        text_list.append(list(text))
    y_sparse_targets = sparse_tuple_from(text_list)
    return x_img,y_sparse_targets




def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representation of x.
    if sequences = [['ab'], ['a']]
    the output will be
    indexes = [[0,0],[0,1],[1,0]]
    values = [1,2,1]
    dense_shape = [2,2] (两个数字串，最大长度为2)

    :param sequences: a list of lists of type dtype where each element is a sequence
    :param dtype:
    :return:   A tuple with (indices, values, shape)
        indices:二维int64的矩阵，代表非0的坐标点
        values:二维tensor，代表indexes位置的数据值
        dense_shape:一维，代表稀疏矩阵的大小
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        for ch in seq:
            values.append(char2num[ch])

    # print("values", values)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    """
    将sparse_tuple_from得到的返回值解码成原来的输入值
    输入：
        indexes = [[0,0],[0,1],[1,0]]
        values = [1,2,1]
        dense_shape = [2,2]
    输出
        sequences =[[1,2], [1]]
    :param sparse_tensor: A tuple with (indices, values, shape)
           indices:二维int64的矩阵，代表非0的坐标点
           values:二维tensor，代表indexes位置的数据值
           dense_shape:一维，代表稀疏矩阵的大小
    :return:
    """
    decoded_indexes = []
    current_i = 0
    current_seq = []

    for offset, (i, index) in enumerate(sparse_tensor[0]):
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = []
        current_seq.append(offset)
    decoded_indexes.append(current_seq)

    # print("decoded_indexes", decoded_indexes)

    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        s = num2char[spars_tensor[1][m]]
        decoded.append(s)
    return ''.join(decoded)



if __name__ == '__main__':
    # img2array()
    x,y=get_next_batch()
    print(x.shape)
    print(y)

