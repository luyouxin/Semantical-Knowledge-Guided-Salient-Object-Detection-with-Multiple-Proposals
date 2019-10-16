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


def test(m):
    # if not os.path.exists(cfg.result_save_path):
    #     os.makedirs(cfg.result_save_path)
    if not os.path.exists(cfg.msra_test):
        os.makedirs(cfg.msra_test)
    print 'Loading weights of model...'
    # m.load_weights('/home/dorothy/project/CGS/model/final-epoch10-15/0/0.3148.h5')
    m.load_weights(os.path.join(cfg.model_path, '0.4092.h5'))
    num = 0
    img_name = sorted(os.listdir(cfg.msra_img_path))
    # img_name = sorted(os.listdir(cfg.test_img_path))
    for i in img_name:
        i_path = cfg.msra_img_path + '/' + i
        # i_path = cfg.test_img_path + '/' + i
        original_img = cv2.imread(i_path)
        img = cv2.resize(original_img, (224, 224))

        im = np.zeros((1, 224, 224, 13), dtype=np.float32)
        im[:, :, :, :3] = img
        im[:, :, :, 0] -= cfg.img_channel_mean[0]
        im[:, :, :, 1] -= cfg.img_channel_mean[1]
        im[:, :, :, 2] -= cfg.img_channel_mean[2]

        # pro_name = sorted(os.listdir(cfg.test_pro + '/' + i[:-4]))
        pro_name = sorted(os.listdir(cfg.msra_p_path + '/' + i[:-4]))
        num1 = 3
        for j in pro_name:
            # original_pro = cv2.imread(cfg.test_pro + '/' + i[:-4] + '/' + j)
            original_pro = cv2.imread(cfg.msra_p_path + '/' + i[:-4] + '/' + j)
            original_pro = cv2.cvtColor(original_pro, cv2.COLOR_RGB2GRAY)
            original_pro = cv2.resize(original_pro, (224, 224))
            im[:, :, :, num1] = original_pro
            num1 += 1

        print 'testing:', num + 1
        num += 1
        pre = m.predict(im)
        prediction = pre[0, :]
        re = postprocess_prediction(prediction, original_img.shape[0], original_img.shape[1])
        # cv2.imwrite(cfg.result_save_path + '/%s.png' % i[:-4], re)
        cv2.imwrite(cfg.msra_test + '/%s.png' % i[:-4], re)
