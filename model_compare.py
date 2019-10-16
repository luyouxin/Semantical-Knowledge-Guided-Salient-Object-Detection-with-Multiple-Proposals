#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

from keras import layers
from keras.applications import VGG16
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import getProposals
import numpy as np

from attention_module import attach_attention_module


def CGScom(data):
    # 编码部分使用VGG166，不包含全连接层，输入图片大小定为224*224，最后一层是block5_pool(MaxPooling2D)大小为7*7*512
    # base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 13), input_tensor=data)
    # # 设置VGG模型部分为可训练
    # base.trainable = True

    x0 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv')(data)
    y = attach_attention_module(x0, 'cbam_block')
    fusion = layers.add([x0, y])
    x0 = layers.Activation('relu')(fusion)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x0)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x4)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # ##自己写的

    # # 14*14*512
    x = layers.Deconv2D(512, (4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv1')(x5)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Concatenate()([x, x4])
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='deconv1_conv1')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='deconv1_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='deconv1_conv3')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    xx1 = layers.Activation('relu')(fusion)
    # x = layers.Conv2D(512, (1, 1), padding='same', activation='relu', name='deconv1_conv4')(x)
    # y = attach_attention_module(x, 'cbam_block')
    # x = layers.add([x, y])
    # x = layers.Activation('relu')(x)

    # 28*28*256
    x = layers.Deconv2D(256, (4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv2')(xx1)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Concatenate()([x, x3])
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='deconv2_conv1')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='deconv2_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='deconv2_conv3')(x)
    # x = layers.Conv2D(256, (1, 1), padding='same', activation='relu', name='deconv2_conv4')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    xx2 = layers.Activation('relu')(fusion)

    # 56*56*128
    x = layers.Deconv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv3')(xx2)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Concatenate()([x, x2])
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='deconv3_conv1')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='deconv3_conv2')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='deconv3_conv3')(x)
    # x = layers.Conv2D(128, (1, 1), padding='same', activation='relu', name='deconv3_conv4')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    xx3 = layers.Activation('relu')(fusion)

    # 112*112*64
    x = layers.Deconv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv4')(xx3)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Concatenate()([x, x1])
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='deconv4_conv1')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='deconv4_conv2')(x)
    # x = layers.Conv2D(64, (1, 1), padding='same', activation='relu', name='deconv4_conv4')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    xx4 = layers.Activation('relu')(fusion)

    # 224*224*32
    x = layers.Deconv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv5')(xx4)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Concatenate()([x, x0])
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='deconv5_conv1')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='deconv5_conv2')(x)
    # x = layers.Conv2D(32, (1, 1), padding='same', activation='relu', name='deconv5_conv4')(x)
    y = attach_attention_module(x, 'cbam_block')
    fusion = layers.add([x, y])
    x = layers.Activation('relu')(fusion)

    # 224*224*1 结果1
    x0 = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='conv6_end')(x)

    # # train
    # return [x0, x0]
    # # return [x, fusion]

    # test
    m = Model(data, x0, name='CGScom')
    return m

