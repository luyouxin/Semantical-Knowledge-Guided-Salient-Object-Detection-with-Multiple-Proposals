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


def CGS(data):
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
    x0 = layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='conv6_end')(x)

    # concat
    d2 = layers.Lambda(lambda x: x[:, :, :, 3])(data)
    y7 = layers.Lambda(lambda x: K.expand_dims(x, axis=3))(d2)
    f = layers.Concatenate()([y7, x0])

    # optimization
    fusion1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='fusion1_conv1')(f)
    y = attach_attention_module(fusion1, 'cbam_block')
    fusion1 = layers.add([fusion1, y])
    fusion1 = layers.Activation('relu')(fusion1)
    fusion1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='fusion1_conv2')(fusion1)
    y = attach_attention_module(fusion1, 'cbam_block')
    fusion1 = layers.add([fusion1, y])
    fusion1 = layers.Activation('relu')(fusion1)
    fusion1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='fusion1_maxpooling')(fusion1)

    fusion2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='fusion2_conv1')(fusion1)
    y = attach_attention_module(fusion2, 'cbam_block')
    fusion2 = layers.add([fusion2, y])
    fusion2 = layers.Activation('relu')(fusion2)
    fusion2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='fusion2_conv2')(fusion2)
    y = attach_attention_module(fusion2, 'cbam_block')
    fusion2 = layers.add([fusion2, y])
    fusion2 = layers.Activation('relu')(fusion2)
    fusion2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='fusion2_maxpooling')(fusion2)

    fusion3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='fusion3_conv1')(fusion2)
    y = attach_attention_module(fusion3, 'cbam_block')
    fusion3 = layers.add([fusion3, y])
    fusion3 = layers.Activation('relu')(fusion3)
    fusion3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='fusion3_conv2')(fusion3)
    y = attach_attention_module(fusion3, 'cbam_block')
    fusion3 = layers.add([fusion3, y])
    fusion3 = layers.Activation('relu')(fusion3)
    fusion3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='fusion3_conv3')(fusion3)
    y = attach_attention_module(fusion3, 'cbam_block')
    fusion3 = layers.add([fusion3, y])
    fusion3 = layers.Activation('relu')(fusion3)
    fusion3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='fusion3_maxpooling')(fusion3)

    fusion4 = layers.Deconv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu', name='fusion4_deconv')(fusion3)
    y = attach_attention_module(fusion4, 'cbam_block')
    fusion4 = layers.add([fusion4, y])
    fusion4 = layers.Activation('relu')(fusion4)
    fusion4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='fusion4_conv1')(fusion4)
    y = attach_attention_module(fusion4, 'cbam_block')
    fusion4 = layers.add([fusion4, y])
    fusion4 = layers.Activation('relu')(fusion4)
    fusion4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='fusion4_conv2')(fusion4)
    y = attach_attention_module(fusion4, 'cbam_block')
    fusion4 = layers.add([fusion4, y])
    fusion4 = layers.Activation('relu')(fusion4)
    fusion4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='fusion4_conv3')(fusion4)
    y = attach_attention_module(fusion4, 'cbam_block')
    fusion4 = layers.add([fusion4, y])
    fusion4 = layers.Activation('relu')(fusion4)

    fusion5 = layers.Deconv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu', name='fusion5_deconv')(fusion4)
    y = attach_attention_module(fusion5, 'cbam_block')
    fusion5 = layers.add([fusion5, y])
    fusion5 = layers.Activation('relu')(fusion5)
    fusion5 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='fusion5_conv1')(fusion5)
    y = attach_attention_module(fusion5, 'cbam_block')
    fusion5 = layers.add([fusion5, y])
    fusion5 = layers.Activation('relu')(fusion5)
    fusion5 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='fusion5_conv2')(fusion5)
    y = attach_attention_module(fusion5, 'cbam_block')
    fusion5 = layers.add([fusion5, y])
    fusion5 = layers.Activation('relu')(fusion5)

    fusion6 = layers.Deconv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu', name='fusion6_deconv')(fusion5)
    y = attach_attention_module(fusion6, 'cbam_block')
    fusion6 = layers.add([fusion6, y])
    fusion6 = layers.Activation('relu')(fusion6)
    fusion6 = layers.Conv2D(32, (1, 1), padding='same', activation='relu', name='fusion6_conv1')(fusion6)
    y = attach_attention_module(fusion6, 'cbam_block')
    fusion6 = layers.add([fusion6, y])
    fusion6 = layers.Activation('relu')(fusion6)
    fusion6 = layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='fusion6_conv2')(fusion6)
    # y = attach_attention_module(fusion6, 'cbam_block')
    # fusion6 = layers.add([fusion6, y])
    # fusion6 = layers.Activation('relu')(fusion6)
    fusion = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='fusion')(fusion6)

    # # train
    # return [fusion, fusion]
    # # return [x, fusion]

    # test
    m = Model(data, fusion, name='CGS')
    return m

