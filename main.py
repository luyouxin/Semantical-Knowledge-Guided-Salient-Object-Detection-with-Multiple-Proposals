#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

import os
import numpy as np
from train import *
from test import *
from getFirstRes import *
import cfg
import cv2
import matplotlib.pyplot as plt
from model import CGS
from keras import Input, Model
from model_compare import CGScom


# 读取训练图像
def get_train_imgs(img_path, pro_path, start, end, stage):
    img_name = sorted(os.listdir(img_path))
    img = np.zeros((1500, 224, 224, 13), dtype=np.float32)
    original_imgs = []
    num = 0  # 计数器
    for i in img_name[start:end]:
        i_path = img_path + '/' + i
        im = cv2.imread(i_path)
        result_img = cv2.resize(im, (224, 224))
        original_imgs.append(im)
        img[num, :, :, :3] = result_img
        num += 1
        print 'Loading ', stage, ' img:', num
    img[:, :, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, :, 2] -= cfg.img_channel_mean[2]

    pro_name = sorted(os.listdir(pro_path))
    num = 0
    for p in pro_name[start:end]:
        num1 = 3
        p_name = sorted(os.listdir(pro_path + '/' + p))
        for p1 in p_name:
            ff = cv2.imread(pro_path + '/' + p + '/' + p1)
            ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
            flag = cv2.resize(ff, (224, 224))
            img[num, :, :, num1] = flag
            print 'Loading ', stage, ' pro', num1, '/', num
            num1 += 1
        num += 1
    return img, original_imgs

# 读取验证图像
def get_val_imgs(img_path, pro_path, start, end, stage):
    img_name = sorted(os.listdir(img_path))
    img = np.zeros((750, 224, 224, 13), dtype=np.float32)
    original_imgs = []
    num = 0  # 计数器
    for i in img_name[start:end]:
        i_path = img_path + '/' + i
        im = cv2.imread(i_path)
        result_img = cv2.resize(im, (224, 224))
        original_imgs.append(im)
        img[num, :, :, :3] = result_img
        num += 1
        print 'Loading ', stage, ' img:', num
    img[:, :, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, :, 2] -= cfg.img_channel_mean[2]

    pro_name = sorted(os.listdir(pro_path))
    num = 0
    for p in pro_name[start:end]:
        num1 = 3
        p_name = sorted(os.listdir(pro_path + '/' + p))
        for p1 in p_name:
            f = cv2.imread(pro_path + '/' + p + '/' + p1)
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
            flag = cv2.resize(f, (224, 224))
            img[num, :, :, num1] = flag
            print 'Loading ', stage, ' pro', num1, '/', num
            num1 += 1
        num += 1
    return img, original_imgs

# 读取trainGT
def get_train_gts(gt_path, start, end, stage):
    gts_name = sorted(os.listdir(gt_path))
    gt = np.zeros((1500, 224, 224, 1), dtype=np.float32)
    num = 0
    for g in gts_name[start:end]:
        g_path = gt_path + '/' + g
        f = cv2.imread(g_path)
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
        f = (f - np.min(f)) / (np.max(f) - np.min(f))
        gt[num, :, :, 0] = cv2.resize(f, (224, 224))
        num += 1
        print 'Loading ', stage, ' gt:', num
    return gt


# 读取valGT
def get_val_gts(gt_path, start, end, stage):
    gts_name = sorted(os.listdir(gt_path))
    gt = np.zeros((750, 224, 224, 1), dtype=np.float32)
    num = 0
    for g in gts_name[start:end]:
        g_path = gt_path + '/' + g
        f = cv2.imread(g_path)
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
        f = (f - np.min(f)) / (np.max(f) - np.min(f))
        gt[num, :, :, 0] = cv2.resize(f, (224, 224))
        num += 1
        print 'Loading ', stage, ' gt:', num
    return gt

#
# # 读取训练pro
# def get_train_pros(pro_path, start, end, stage):
#     pro_name = sorted(os.listdir(pro_path))
#     pro = np.zeros((1500, 224, 224, 13), dtype=np.float32)
#     num = 0
#     for p in pro_name[start:end]:
#         num1 = 0
#         p_name = sorted(os.listdir(pro_path + '/' + p))
#         for p1 in p_name:
#             ff = cv2.imread(pro_path + '/' + p + '/' + p1)
#             ff = cv2.cvtColor(ff, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
#             flag = cv2.resize(ff, (224, 224))
#             pro[num, :, :, num1] = flag
#             print 'Loading ', stage, ' pro', num1, '/', num
#             num1 += 1
#         num += 1
#     return pro


# 读取验证pro
# def get_val_pros(pro_path, start, end, stage):
#     pro_name = sorted(os.listdir(pro_path))
#     pro = np.zeros((750, 224, 224, 10), dtype=np.float32)
#     num = 0
#     for p in pro_name[start:end]:
#         num1 = 0
#         p_name = sorted(os.listdir(pro_path + '/' + p))
#         for p1 in p_name:
#             f = cv2.imread(pro_path + '/' + p + '/' + p1)
#             f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)  # 将彩色图变成灰度图
#             flag = cv2.resize(f, (224, 224))
#             pro[num, :, :, num1] = flag
#             print 'Loading ', stage, ' pro', num1, '/', num
#             num1 += 1
#         num += 1
#     return pro


if __name__ == '__main__':
    # 获取训练图像数据
    # # 1
    # train_img, original_train_img = get_train_imgs(cfg.train_img_path, cfg.train_pro, 0, 1500, 'train')
    # train_gt = get_train_gts(cfg.train_gt_path, 0, 1500, 'train')
    #
    # validation_img, original_validation_img = get_val_imgs(cfg.validation_img_path, cfg.val_pro, 0, 750, 'validation')
    # validation_gt = get_val_gts(cfg.validation_gt_path, 0, 750, 'validation')

    # # 2
    # train_img, original_train_img = get_train_imgs(cfg.train_img_path, cfg.train_pro, 1500, 3000, 'train')
    # train_gt = get_train_gts(cfg.train_gt_path, 1500, 3000, 'train')
    #
    # validation_img, original_validation_img = get_val_imgs(cfg.validation_img_path, cfg.val_pro, 750, 1500, 'validation')
    # validation_gt = get_val_gts(cfg.validation_gt_path, 750, 1500, 'validation')

    # # 3
    # train_img, original_train_img = get_train_imgs(cfg.train_img_path, cfg.train_pro, 3000, 4500, 'train')
    # train_gt = get_train_gts(cfg.train_gt_path, 3000, 4500, 'train')
    #
    # validation_img, original_validation_img = get_val_imgs(cfg.validation_img_path, cfg.val_pro, 1500, 2250, 'validation')
    # validation_gt = get_val_gts(cfg.validation_gt_path, 1500, 2250, 'validation')

    # # 4
    # train_img, original_train_img = get_train_imgs(cfg.train_img_path, cfg.train_pro, 4500, 6000, 'train')
    # train_gt = get_train_gts(cfg.train_gt_path, 4500, 6000, 'train')
    #
    # validation_img, original_validation_img = get_val_imgs(cfg.validation_img_path, cfg.val_pro, 2250, 3000, 'validation')
    # validation_gt = get_val_gts(cfg.validation_gt_path, 2250, 3000, 'validation')

    # # final
    # x1 = Input(shape=(224, 224, 13))
    # m = Model(inputs=x1, outputs=CGS(x1))
    # print(m.summary())
    # history = train(m, train_img, [train_gt, train_gt], validation_img, [validation_gt, validation_gt])

    # # 测试模型
    # x1 = Input(shape=(224, 224, 13))
    # # x2 = Input(shape=(224, 224, 10))
    # m = Model(inputs=x1, outputs=CGS(x1).output)
    # print(m.summary())
    # test(m)

    # #compare
    # x1 = Input(shape=(224, 224, 13))
    # m = Model(inputs=x1, outputs=CGScom(x1))
    # print(m.summary())
    # history = train(m, train_img, [train_gt, train_gt], validation_img, [validation_gt, validation_gt])
    #compare
    x1 = Input(shape=(224, 224, 13))
    m = Model(inputs=x1, outputs=CGScom(x1).output)
    print(m.summary())
    test(m)
