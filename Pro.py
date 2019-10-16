#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ----ZhangXue----
# Dorothyzhx@foxmail.com

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Pro(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Pro, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[200], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Pro, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
