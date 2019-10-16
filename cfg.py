#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@gmail.com

import numpy as np

# 各种路径
train_img_path = '/home/dorothy/project/CGS/dataset/train'
validation_img_path = '/home/dorothy/project/CGS/dataset/validation'
train_gt_path = '/home/dorothy/project/CGS/dataset/train_gt'
validation_gt_path = '/home/dorothy/project/CGS/dataset/validation_gt'
test_img_path = '/home/dorothy/project/CGS/dataset/test'
test_gt_path = '/home/dorothy/project/CGS/dataset/test_gt'

model_path = '/home/dorothy/project/CGS/model'
result_save_path = '/home/dorothy/project/CGS/result'

first_train_res = '/home/dorothy/project/CGS/dataset/firstTrain'
first_train_model = '/home/dorothy/project/CGS/model/second'

train_pro = '/home/dorothy/projectData/pro/train_p'
# train_p = '/home/dorothy/projectData/train_p'
val_pro = '/home/dorothy/projectData/pro/val_p'
# val_p = '/home/dorothy/projectData/val_p'
test_pro = '/home/dorothy/projectData/pro/test_p'
# test_p = '/home/dorothy/projectData/test_p'


# #########MSRA########### #
msra_img_path = '/home/dorothy/dataset/MSRA-B/img'
msra_gt_path = '/home/dorothy/dataset/MSRA-B/gt'
msra_pro_path = '/home/dorothy/projectData/MSRA-B_pro'
msra_p_path = '/home/dorothy/projectData/pro/MSRA-B_p'
msra_test = '/home/dorothy/projectResult/msra'


# #########PASCAL-S########### #
pascal_img_path = '/home/dorothy/dataset/PASCAL-S/img'
pascal_gt_path = '/home/dorothy/dataset/PASCAL-S/gt'
pascal_pro_path = '/home/dorothy/projectData/PASCAL-S_pro'
pascal_p_path = '/home/dorothy/projectData/pro/PASCAL-S_p'
pascal_test = '/home/dorothy/projectResult/pascal'

# ########################HKU-IS##################### #
hku_img_path = '/home/dorothy/dataset/HKU-IS/img'
hku_gt_path = '/home/dorothy/dataset/HKU-IS/gt'
hku_pro_path = '/home/dorothy/projectData/HKU-IS_pro'
hku_p_path = '/home/dorothy/projectData/pro/HKU-IS_p'
hku_test = '/home/dorothy/projectResult/hku'

# ########################ECSSD##################### #
ecssd_img_path = '/home/dorothy/dataset/ECCSD/img'
ecssd_gt_path = '/home/dorothy/dataset/ECCSD/gt'
eccsd_pro_path = '/home/dorothy/projectData/ECCSD_pro'
ecssd_p_path = '/home/dorothy/projectData/pro/ECSSD_p'
ecssd_test = '/home/dorothy/projectResult/ecssd'


# ImageNet所有图的均值
img_channel_mean = np.array([104.00698793, 116.66876762, 122.67891434])