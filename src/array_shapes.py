#!/usr/bin/env python3

"""
Module to change the shape of numpy arrays depending on normalization direction

Developer: Marco Cardoso
"""

import numpy as np

def time_epochs_to_2d(epochs_arr, shape):
    """
    Converts 3D arrary of shape (epochs, channels, time) to 2D array of shape
    (epochs * channels, time).
    First `channels` rows correspond to the first epochs,
    rows [channels:channels * 2] correspond to the channels of second epoch and
    so forth.

    Parameters
    ----------
    epochs_arr : np.array
        3D array with EEG data
    shape : tuple
        shape of `epochs_arr`

    Returns
    -------
    new : np.array
        2D array with EEG data
    """
    epochs, channels, time = shape
    new = epochs_arr.copy().reshape(-1, time)
    # new = np.empty((epochs * channels, time))
    # for epoch in range(epochs):
    #     start = epoch * channels
    #     end = start + channels
    #     new[start:end, :] = epochs_arr[epoch, ...]

    return new


def time_2d_to_epochs(epochs_arr, shape):
    """
    Converts 2D array of shape (epochs * channels, time) to 3D array of shape
    (epochs, channels, time)
    First `channels` rows correspond to the first epoch,
    rows [channels:channels * 2] correspond to the channels of second epoch and
    so forth.

    Parameters
    ----------
    epochs_arr : np.array
        2D array with EEG data
    shape : tuple
        shape of `epochs_arr`

    Returns
    -------
    new : np.array
        3D array with EEG data
    """
    epochs, channels, time = shape
    new = epochs_arr.reshape(epochs, channels, time)
    # new = np.empty((epochs, channels, time))
    # for epoch in range(epochs):
    #     start = epoch * channels
    #     end = start + channels
    #     new[epoch, ...] = epochs_arr[start:end, :]

    return new


def channels_epochs_to_2d(epochs_arr, shape):
    """
    Converts 3D arrary of shape (epochs, channels, time) to 2D array of shape
    (channels, epochs * time).
    First `time` points correspond to the first epochs,
    points [time:time * 2] correspond to the time samples of second epoch and
    so forth.

    Parameters
    ----------
    epochs_arr : np.array
        3D array with EEG data
    shape : tuple
        shape of `epochs_arr`

    Returns
    -------
    new : np.array
        2D array with EEG data of shape (time * epochs, channels)
    """
    epochs, channels, time = shape
    new = epochs_arr.transpose(1, 0, 2).reshape(channels, epochs * time)
    # new = np.empty((channels, epochs * time))
    # for epoch in range(epochs):
    #     start = epoch * time
    #     end = start + time
    #     new[:, start:end] = epochs_arr[epoch, ...]

    return new.T


def channels_2d_to_epochs(epochs_arr, shape):
    """
    Converts 2D array of shape (channels, epochs * time) to 3D array of shape
    (epochs, channels, time)
    First `time` points correspond to the first epochs,
    points [time:time * 2] correspond to the time samples of second epoch and
    so forth.

    Parameters
    ----------
    epochs_arr : np.array
        2D array with EEG data of shape (time * epochs, channels)
    shape : tuple
        shape of `epochs_arr`

    Returns
    -------
    new : np.array
        3D array with EEG data
    """
    epochs, channels, time = shape
    epochs_arr = epochs_arr.T
    new = epochs_arr.reshape(channels, epochs, time)\
                                .transpose(1, 0, 2)
    # new = np.empty((epochs, channels, time))
    # for epoch in range(epochs):
    #     start = epoch * time
    #     end = start + time
    #     new[epoch, ...] = epochs_arr[:, start:end]


    return new
