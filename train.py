#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

from keras import optimizers
from keras.callbacks import ModelCheckpoint
import os
from keras.losses import binary_crossentropy, MAE
import cfg


def train(m, train_img, train_gt, validation_img, validation_gt):
    # 编译
    m.compile(loss=[binary_crossentropy, MAE],
              optimizer=optimizers.Adam(lr=1e-6),
              metrics=['acc'])

    # 画出网络模型
    print m.summary()

    # 每个epoch保存一次model，以验证损失命名
    checkpoint = ModelCheckpoint('{}'.format(cfg.model_path) + '/{val_loss:.4f}.h5', verbose=1)

    # m.load_weights(os.path.join(cfg.model_path, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop(1).h5'), by_name=True)
    m.load_weights(os.path.join(cfg.model_path, '0.4609.h5'))
    # 训练模型
    history = m.fit(train_img, train_gt, validation_data=(validation_img, validation_gt),
                    epochs=15, batch_size=1, verbose=1, callbacks=[checkpoint])

    # 保存模型
    return history
