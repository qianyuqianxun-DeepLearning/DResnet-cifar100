#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#tensoflow 1.x 版本

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tflearn
import numpy as np
from tflearn.layers.conv import conv_2d

#cifar-10数据集下载
from tflearn.datasets import cifar100
(x,y), (testx,testy)=cifar100.load_data()

#cifar10数据集进行噪声添加
x=x+np.random.random((50000,32,32,3))*0.1
testx=testx+np.random.random((10000,32,32,3))*0.1

#cifar10数据集中标签转换
y=tflearn.data_utils.to_categorical(y,100)
testy=tflearn.data_utils.to_categorical(testy,100)

def residual_shrinkage_block(incoming,nb_blocks,out_channels,downsample=False,
                            downsample_strides=2,activation='relu',batch_norm=True,
                            bias=True, weights_init='variance_scaling',
                            bias_init='zeros',regularizer='L2',weight_decay=0.0001,
                            trainable=True,restore=True,reuse=False,scope=None,name="ResidualBlock"):

    #具有通道阈值的残差收缩块
    residual=incoming
    in_channels=incoming.get_shape().as_list()[-1]
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)
    with vscope as scope:
        name=scope.name

        for i in range(nb_blocks):

            identity=residual

            if  not downsample:
                downsample_strides=1

            if batch_norm:
                residual=tflearn.batch_normalization(residual)
            residual=tflearn.activation(residual,activation)
            residual=conv_2d(residual,out_channels,3,
                            downsample_strides,'same','linear',
                            bias,weights_init,bias_init,
                            regularizer,weight_decay,trainable,
                            restore)

            if batch_norm:
                residual=tflearn.batch_normalization(residual)
            residual=tflearn.activation(residual,activation)
            residual=conv_2d(residual,out_channels,3,1,
                            'same','linear',
                            bias,weights_init,bias_init,
                            regularizer,weight_decay,trainable,
                            restore)
            #得到软阈值函数的阈值并且进行阈值
            abs_mean = tf.reduce_mean(tf.reduce_mean(tf.abs(residual), axis=2, keep_dims=True), axis=1, keep_dims=True)
            scales = tflearn.fully_connected(abs_mean, out_channels // 4, activation='linear', regularizer='L2',
                                             weight_decay=0.0001, weights_init='variance_scaling')
            scales = tflearn.fully_connected(abs_mean, out_channels//4, activation='linear',regularizer='L2',weight_decay=0.0001,weights_init='variance_scaling')
            scales = tflearn.batch_normalization(scales)
            scales = tflearn.activation(scales, 'relu')
            scales = tflearn.fully_connected(scales, out_channels, activation='linear',regularizer='L2',weight_decay=0.0001,weights_init='variance_scaling')
            scales = tf.expand_dims(tf.expand_dims(scales,axis=1),axis=1)
            thres = tf.multiply(abs_mean,tflearn.activations.sigmoid(scales))
            # soft thresholding
            residual = tf.multiply(tf.sign(residual), tf.maximum(tf.abs(residual) - thres, 0))

            #下采样
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, 1,
                                               downsample_strides)
            #投射到新维度
            if in_channels != out_channels:
                if (out_channels - in_channels) % 2 == 0:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch]])
                else:
                    ch = (out_channels - in_channels) // 2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch + 1]])
                in_channels = out_channels

            residual = residual + identity

        return residual
#实时数据存储
img_prep=tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
#实时数据扩充
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

#建立具有三个残差块的残差网络
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = residual_shrinkage_block(net, 1, 16)
net = residual_shrinkage_block(net, 1, 32, downsample=True)
net = residual_shrinkage_block(net, 1, 32, downsample=True)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

#模型回归
net = tflearn.fully_connected(net, 100, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=20000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='model_cifar100',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)
model.fit(x, y, n_epoch=200, snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=100, shuffle=True, run_id='model_cifar100')

training_acc = model.evaluate(x, y)[0]
validation_acc = model.evaluate(testx, testy)[0]