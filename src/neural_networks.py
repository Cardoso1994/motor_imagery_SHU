#!/usr/bin/env python3
"""
Keras and Tensorflow related
"""

import numpy as np
import tensorflow as tf
from keras import activations, layers, losses, optimizers
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def base_cnn(input_shape, activ='tanh'):
    """
    Base ConvNet for SHU Dataset experimentation

    Parameters
    ----------
    input_shape : tuple
        shape ConvNet's input
    activ : str
        activation function as a string. Values: 'tanh', 'sigmoid', 'relu'
    """

    _activations = {'tanh': activations.tanh,
                   'relu': activations.relu,
                   'sigmoid': activations.sigmoid}

    input = layers.Input(shape=input_shape)
    activ_layer = _activations[activ]

    # first layer
    x = layers.Conv1D(16, kernel_size=3, strides=1, padding="same")(input)
    x = activ_layer(x)

    # second layer
    x = layers.Conv1D(32, kernel_size=3, strides=1, padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = activ_layer(x)

    # third layer
    x = layers.Conv1D(32, kernel_size=3, strides=1, padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = activ_layer(x)

    # fourth layer
    x = layers.Conv1D(64, kernel_size=3, strides=1, padding="same")(x)
    x = activ_layer(x)

    # classifier
    x = layers.Flatten()(x)
    # x = layers.Dense(64)(x)
    # x = activ_layer(x)
    output = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model


def scheduler(epoch, lr):
    """
    scheduler to change the value of the learning rate depending on whic epoch
    the training is
    """
    if epoch < 5:
        return lr
    # elif epoch >= 5 and epoch < 20:
    #     return lr * np.exp(-0.009)
    else:
        # return lr * np.exp(-0.1)
        return lr * np.exp(-0.0009)
