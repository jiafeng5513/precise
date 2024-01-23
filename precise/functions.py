# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mathematical functions used to customize
computation in various places
"""
from math import exp, log, sqrt, pi
import numpy as np
from typing import *
import tensorflow as tf
LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias


def set_loss_bias(bias: float):
    """
    Changes the loss bias

    This allows customizing the acceptable tolerance between
    false negatives and false positives

    Near 1.0 reduces false positives
    Near 0.0 reduces false negatives
    """
    global LOSS_BIAS
    LOSS_BIAS = bias


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    pos_loss = -(0 + yt) * tf.math.log(0 + yp + tf.keras.backend.epsilon())
    neg_loss = -(1 - yt) * tf.math.log(1 - yp + tf.keras.backend.epsilon())

    return LOSS_BIAS * tf.math.reduce_mean(neg_loss) + (1. - LOSS_BIAS) * tf.math.reduce_mean(pos_loss)


class WeightedLogLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        pos_loss = -(0 + y_true) * tf.math.log(0 + y_pred + tf.keras.backend.epsilon())
        neg_loss = -(1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())

        return LOSS_BIAS * tf.math.reduce_mean(neg_loss) + (1. - LOSS_BIAS) * tf.math.reduce_mean(pos_loss)

def weighted_mse_loss(yt, yp) -> Any:
    """Standard mse loss with a weighting between false negatives and positives"""
    total = tf.math.reduce_sum(tf.ones_like(yt))
    neg_loss = total * tf.math.reduce_sum(tf.math.reduce_square(yp * (1 - yt))) / tf.math.reduce_sum(1 - yt)
    pos_loss = total * tf.math.reduce_sum(tf.math.square(1. - (yp * yt))) / tf.math.reduce_sum(yt)

    return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss


def false_pos(yt, yp) -> Any:
    """
    Metric for Keras that *estimates* false positives while training
    This will not be completely accurate because it weights batches
    equally
    """
    return tf.math.reduce_sum(tf.cast(yp * (1 - yt) > 0.5, 'float')) / tf.math.maximum(1.0, tf.math.reduce_sum(1 - yt))


def false_neg(yt, yp) -> Any:
    """
    Metric for Keras that *estimates* false negatives while training
    This will not be completely accurate because it weights batches
    equally
    """
    return tf.math.reduce_sum(tf.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / tf.math.maximum(1.0, tf.math.reduce_sum(0 + yt))


# def load_keras() -> Any:
#     """Imports Keras injecting custom functions to prevent exceptions"""
#     import keras
#     keras.losses.weighted_log_loss = weighted_log_loss
#     # keras.losses.Loss = WeightedLogLoss
#     keras.metrics.false_pos = false_pos
#     keras.metrics.false_positives = false_pos
#     keras.metrics.false_neg = false_neg
#     return keras


def sigmoid(x):
    """Sigmoid squashing function for scalars"""
    return 1 / (1 + exp(-x))


def asigmoid(x):
    """Inverse sigmoid (logit) for scalars"""
    return -log(1 / x - 1)


def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    if std == 0:
        return 0
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
