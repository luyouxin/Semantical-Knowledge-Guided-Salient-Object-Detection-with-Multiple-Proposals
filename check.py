#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com
import os
import cv2
import cfg
import os
import shutil
path = '/home/dorothy/projectData/pro/ECSSD_p'
name = sorted(os.listdir(path))
o=[]
for i in name:
    p = path+'/'+i
    m = sorted(os.listdir(p))
    print i
    if len(m)<10:
        # for num in range(10):
        #     if not os.path.exists(path+'/'+i+'/'+str(num)+'.jpg'):
        #         src_path = path+'/'+i+'/0.jpg'
        #         save_path = path+'/'+i+'/'+str(num)+'.jpg'
        #         shutil.copy(src_path, save_path)
        o.append(i)
print(o)