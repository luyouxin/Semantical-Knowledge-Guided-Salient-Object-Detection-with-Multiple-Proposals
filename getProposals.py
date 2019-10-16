#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

import cv2
import cfg
import os
import keras.utils
import numpy as np
from keras import layers
import tensorflow as tf
from keras import backend as K
import shutil


def getProposals(gt_path, pro_path):
    gt = sorted(os.listdir(gt_path))
    num = 0
    for g in gt:
        print 'processing:', num, '/', len(gt)
        save_path = '/home/dorothy/projectData/pro/ECSSD_p/' + g[:-4]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            num += 1
            continue

        img = cv2.imread(gt_path + '/' + g)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        train_pro_name = sorted(os.listdir(pro_path + '/' + g[:-4]))
        pro_score = {}
        if len(train_pro_name) == 0:
            continue
        for pro_name in train_pro_name:
            if pro_name == 'pro':
                continue
            pro_img = cv2.imread(pro_path + '/' + g[:-4] + '/' + pro_name)
            pro_img = cv2.resize(pro_img, (img.shape[1], img.shape[0]))
            pro_img = cv2.cvtColor(pro_img, cv2.COLOR_RGB2GRAY)
            I = cv2.bitwise_and(img, pro_img)  # 交集, 共有的部分
            numOfI = cv2.countNonZero(I)  # 统计U中非零像素个数
            U = cv2.bitwise_and(cv2.bitwise_or(img, pro_img), cv2.bitwise_not(I))  # 并集-交集的部分，不准的部分
            # U = cv2.bitwise_or(img, pro_img)
            numOfU = cv2.countNonZero(U)  # 统计N中非零像素个数
            IoU = float(numOfI) / float(numOfU)
            pro_score.update({pro_name: IoU})
        pro_score = sorted(pro_score.items(), key=lambda item: item[1], reverse=True)  # 按分值排序



        if len(train_pro_name) >= 10:
            i = 0
            for select in pro_score[:10]:
                if select[1] <= 3:
                    src_path = pro_path + '/' + g[:-4] + '/' + pro_score[0][0]
                    shutil.copy(src_path, save_path)
                    src = save_path + '/' + pro_score[0][0]
                    dst = save_path + '/{}.jpg'
                    dst = dst.format(i)
                    os.rename(src, dst)
                    i += 1
                    # break
                else:
                    src_path = pro_path + '/' + g[:-4] + '/' + select[0]
                    shutil.copy(src_path, save_path)
                    src = save_path + '/' + select[0]
                    dst = save_path + '/{}.jpg'
                    dst = dst.format(i)
                    os.rename(src, dst)
                    i += 1
        else:
            count = int(len(train_pro_name))
            j = 0
            for select in pro_score[:count]:
                if select[1] <= 3:
                    src_path = pro_path + '/' + g[:-4] + '/' + pro_score[0][0]
                    shutil.copy(src_path, save_path)
                    src = save_path + '/' + pro_score[0][0]
                    dst = save_path + '/{}.jpg'
                    dst = dst.format(j)
                    os.rename(src, dst)
                    j += 1
                    # break
                else:
                    src_path = pro_path + '/' + g[:-4] + '/' + select[0]
                    shutil.copy(src_path, save_path)
                    src = save_path + '/' + select[0]
                    dst = save_path + '/{}.jpg'
                    dst = dst.format(j)
                    os.rename(src, dst)
                    j += 1

            while j < 10:
                src_path = pro_path + '/' + g[:-4] + '/' + pro_score[0][0]
                shutil.copy(src_path, save_path)
                src = save_path + '/' + pro_score[0][0]
                dst = save_path + '/{}.jpg'
                dst = dst.format(j)
                os.rename(src, dst)
                j += 1

        num += 1



if __name__ == '__main__':
    getProposals(cfg.ecssd_gt_path, cfg.eccsd_pro_path)
