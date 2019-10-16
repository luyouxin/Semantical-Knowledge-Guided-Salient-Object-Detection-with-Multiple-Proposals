#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@gmail.com

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
import cv2


def Visualization():

    model = load_model('/home/dorothy/project/CGS/model/0.3094.h5')
    model.summary()

    img_path = '/home/dorothy/project/CGS/dataset/test/8814.jpg'

    img = np.zeros((1, 224, 224, 13), dtype=np.float32)
    im = cv2.imread(img_path)
    result_img = cv2.resize(im, (224, 224))
    img[0, :, :, :3] = result_img

    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/0.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 3] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/1.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 4] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/2.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 5] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/3.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 6] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/4.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 7] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/5.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 8] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/6.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 9] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/7.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 10] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/8.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 11] = flag
    ff = cv2.imread('/home/dorothy/projectData/pro/test_p/8814/9.jpg')
    ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
    flag = cv2.resize(ff, (224, 224))
    img[0, :, :, 12] = flag

    # img_tensor /= 255.

    print(img.shape)

    # plt.imshow(img[0])
    # plt.show()

    layer_outputs = model.layers[531].output
    # layer_outputs = [layer.output for layer in model.layers[:17]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img)
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    re = first_layer_activation[:, :, 0]

    re = cv2.resize(re, (400, 300))
    re = (re - np.min(re)) / (np.max(re) - np.min(re))
    re = re * 255
    cv2.imwrite('/home/dorothy/projectResult/original.png', re)

    plt.matshow(re)
    plt.show()


Visualization()
