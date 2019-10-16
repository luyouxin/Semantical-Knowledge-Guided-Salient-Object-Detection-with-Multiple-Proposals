#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

import os
import cv2
import numpy as np
import cfg


def postprocess_prediction(prediction, shape1, shape2):
    re = cv2.resize(prediction, (shape2, shape1))
    re = (re - np.min(re)) / (np.max(re) - np.min(re))
    re = re * 255
    return re


def getFirRes(m):
    if not os.path.exists(cfg.first_train_res):
        os.makedirs(cfg.first_train_res)
    print 'Loading weights of model...'
    m.load_weights(os.path.join(cfg.model_path, '0.5929.h5'))
    num = 0
    img_name = sorted(os.listdir(cfg.train_img_path))
    for i in img_name[0:6000]:
        i_path = cfg.train_img_path + '/' + i
        original_img = cv2.imread(i_path)
        img = cv2.resize(original_img, (224, 224))
        print 'testing:', num + 1
        num += 1
        t = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        t[0, :, :, :] = img
        pre = m.predict(t)
        prediction = pre[0, :]
        re = postprocess_prediction(prediction, original_img.shape[0], original_img.shape[1])
        cv2.imwrite(cfg.first_train_res + '/%s.png' % i[:-4], re)
