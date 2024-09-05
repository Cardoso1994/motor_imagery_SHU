#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from array_shapes import time_epochs_to_2d, time_2d_to_epochs, \
    channels_epochs_to_2d, channels_2d_to_epochs

def norm_0(train, test, dir='channels'):
    """
    Training data is transformed computing mean and std on train data; test is
    transformed using mean and std from train

    Parameters
    ----------
    train : list of mne.Epochs
    test : list of mne.Epochs
    dir : str
        Direction of normalization.
        - 'channels': normalize each channel on its own
        - 'time': normalize each point in time on its own

    Returns
    -------
    (normalized_train, normalized_test) : tuple
        - normalized_train: list of normalized train mne epochs
        - normalized_test: list of normalized test mne epochs
    """

    """ extracting data and info from epochs"""
    train_epochs = mne.concatenate_epochs(train, add_offset=True)
    train_epochs_info = train_epochs.info
    _train = train_epochs.get_data(copy=True)
    train_epochs_shape = _train.shape  # (epochs, channels, time)

    """ cast arrays to correpsonding dimensions depending on dir """
    if dir == 'channels':
        _train = channels_epochs_to_2d(_train, train_epochs_shape)
        # _test = channels_epochs_to_2d(_test, test_epochs_shape).T
    elif dir == 'time':
        _train = time_epochs_to_2d(_train, train_epochs_shape)
        # _test = time_epochs_to_2d(_test, test_epochs_shape)
    else:
        print("Wrong direction for normalization")
        sys.exit(-1)

    """ normalization """
    # fit to train
    scaler = StandardScaler()
    scaler.fit(_train)
    del _train, train_epochs, train_epochs_shape

    # normalize train
    normalized_train = []
    for train_epoch in train:
        train_epoch_info = train_epoch.info
        train_epoch_event_id = train_epoch.event_id
        _train = train_epoch.get_data(copy=True)
        train_epoch_shape = _train.shape  # (epochs, channels, time)
        if dir == 'channels':
            _train = channels_epochs_to_2d(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_epochs_to_2d(_train, train_epoch_shape)

        _train = scaler.transform(_train)

        if dir == 'channels':
            _train = channels_2d_to_epochs(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_2d_to_epochs(_train, train_epoch_shape)

        normalized_train.append(mne.EpochsArray(_train, train_epoch_info,
                                                events=train_epoch.events,
                                                event_id=train_epoch_event_id))

    # normalize test
    normalized_test = []
    for test_epoch in test:
        test_epoch_info = test_epoch.info
        test_epoch_event_id = test_epoch.event_id
        _test = test_epoch.get_data(copy=True)
        test_epoch_shape = _test.shape  # (epochs, channels, time)
        if dir == 'channels':
            _test = channels_epochs_to_2d(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_epochs_to_2d(_test, test_epoch_shape)

        _test = scaler.transform(_test)

        if dir == 'channels':
            _test = channels_2d_to_epochs(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_2d_to_epochs(_test, test_epoch_shape)
        normalized_test.append(mne.EpochsArray(_test, test_epoch_info,
                                               events=test_epoch.events,
                                               event_id=test_epoch_event_id))

    return (normalized_train, normalized_test)


def norm_1(train, test, dir='channels'):
    """
    Each session in the training set is normalized with its own mean and std.
    Test data is transformed using mean and std of the whole training set

    Parameters
    ----------
    train : list of mne.Epochs
    test : list of mne.Epochs
    dir : str
        Direction of normalization.
        - 'channels': normalize each channel on its own
        - 'time': normalize each point in time on its own

    Returns
    -------
    (normalized_train, normalized_test) : tuple
        - normalized_train: list of normalized train mne epochs
        - normalized_test: list of normalized test mne epochs
    """

    """
    Normalize train data
    """
    normalized_train = []
    """ normalize train epochs individually """
    for train_epoch in train:
        train_epoch_info = train_epoch.info
        train_epoch_event_id = train_epoch.event_id
        _train = train_epoch.get_data(copy=True)
        train_epoch_shape = _train.shape  # (epochs, channels, time)
        if dir == 'channels':
            _train = channels_epochs_to_2d(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_epochs_to_2d(_train, train_epoch_shape)

        #  normalization
        scaler = StandardScaler()
        scaler.fit(_train)
        _train = scaler.transform(_train)

        # restore to 3D arrays
        if dir == 'channels':
            _train = channels_2d_to_epochs(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_2d_to_epochs(_train, train_epoch_shape)

        _train_epoch = mne.EpochsArray(_train, train_epoch_info,
                                       events=train_epoch.events,
                                       event_id=train_epoch_event_id)

        normalized_train.append(_train_epoch)

    """
    Normalize test data
    """
    """ extracting data and info from epochs"""
    train_epochs = mne.concatenate_epochs(train, add_offset=True)
    train_epochs_info = train_epochs.info
    _train = train_epochs.get_data(copy=True)
    train_epochs_shape = _train.shape  # (epochs, channels, time)

    test_epochs = mne.concatenate_epochs(test, add_offset=False)
    test_epochs_info = test_epochs.info
    test_epochs_event_id = test_epochs.event_id
    _test = test_epochs.get_data(copy=True)
    test_epochs_shape = _test.shape  # (epochs, channels, time)

    """ cast arrays to correpsonding dimensions depending on dir """
    if dir == 'channels':
        _train = channels_epochs_to_2d(_train, train_epochs_shape)
        _test = channels_epochs_to_2d(_test, test_epochs_shape)
    elif dir == 'time':
        _train = time_epochs_to_2d(_train, train_epochs_shape)
        _test = time_epochs_to_2d(_test, test_epochs_shape)
    else:
        print("Wrong direction for normalization")
        sys.exit(-1)

    """ normalization """
    scaler = StandardScaler()
    scaler.fit(_train)
    _test = scaler.transform(_test)
    del scaler, train_epochs, train_epochs_info, _train, train_epochs_shape

    """ restore to 3D arrays """
    if dir == 'channels':
        _test = channels_2d_to_epochs(_test, test_epochs_shape)
    elif dir == 'time':
        _test = time_2d_to_epochs(_test, test_epochs_shape)

    """ convert to mne epochs array """
    test_dat = _test
    normalized_test = [mne.EpochsArray(test_dat, test_epochs_info,
                                       events=test_epochs.events,
                                       event_id=test_epochs_event_id), ]

    return (normalized_train, normalized_test)


def norm_2(train, test, dir='channels', norm='z-score'):
    """
    Each session, regardless if in training or testing is transformed using its
    own mean and std

    Parameters
    ----------
    train : list of mne.Epochs
    test : list of mne.Epochs
    dir : str
        Direction of normalization.
        - 'channels': normalize each channel on its own
        - 'time': normalize each point in time on its own
    Returns
    -------
    (normalized_train, normalized_test) : tuple
        - normalized_train: list of normalized train mne epochs
        - normalized_test: list of normalized test mne epochs
    """

    _scalers = {'minmax': MinMaxScaler,
                'z-score': StandardScaler}
    _scaler = _scalers[norm]
    """
    Normalize train data
    """
    normalized_train = []
    """ normalize train epochs individually """
    for train_epoch in train:
        train_epoch_info = train_epoch.info
        train_epoch_event_id = train_epoch.event_id
        _train = train_epoch.get_data(copy=True)
        train_epoch_shape = _train.shape  # (epochs, channels, time)

        # convert to 2D for corresponding normalization
        if dir == 'channels':
            _train = channels_epochs_to_2d(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_epochs_to_2d(_train, train_epoch_shape)

        #  normalization
        scaler = _scaler()
        scaler.fit(_train)
        _train = scaler.transform(_train)

        # restore to 3D arrays
        if dir == 'channels':
            _train = channels_2d_to_epochs(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_2d_to_epochs(_train, train_epoch_shape)

        _train_epoch = mne.EpochsArray(_train, train_epoch_info,
                                       events=train_epoch.events,
                                       event_id=train_epoch_event_id)
        normalized_train.append(_train_epoch)

    """
    Normalize test data
    """
    normalized_test = []
    """ normalize test epochs individually """
    for test_epoch in test:
        test_epoch_info = test_epoch.info
        test_epoch_event_id = test_epoch.event_id
        _test = test_epoch.get_data(copy=True)
        test_epoch_shape = _test.shape  # (epochs, channels, time)

        # convert to 2D for corresponding normalization
        if dir == 'channels':
            _test = channels_epochs_to_2d(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_epochs_to_2d(_test, test_epoch_shape)

        #  normalization
        scaler = _scaler()
        scaler.fit(_test)
        _test = scaler.transform(_test)

        # restore to 3D arrays
        if dir == 'channels':
            _test = channels_2d_to_epochs(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_2d_to_epochs(_test, test_epoch_shape)

        _test_epoch = mne.EpochsArray(_test, test_epoch_info,
                                      events=test_epoch.events,
                                      event_id=test_epoch_event_id)
        normalized_test.append(_test_epoch)

    return (normalized_train, normalized_test)


def norm_3(train, test, dir='channels'):
    """
    Training data is transformed computing mean and std on train data; test is
    transformed using its own mean and std

    Parameters
    ----------
    train : list of mne.Epochs
    test : list of mne.Epochs
    dir : str
        Direction of normalization.
        - 'channels': normalize each channel on its own
        - 'time': normalize each point in time on its own
    Returns
    -------
    (normalized_train, normalized_test) : tuple
        - normalized_train: list of normalized train mne epochs
        - normalized_test: list of normalized test mne epochs
    """

    """ extracting data and info from epochs"""
    train_epochs = mne.concatenate_epochs(train, add_offset=True)
    train_epochs_info = train_epochs.info
    _train = train_epochs.get_data(copy=True)
    train_epochs_shape = _train.shape  # (epochs, channels, time)

    """ cast arrays to correpsonding dimensions depending on dir """
    if dir == 'channels':
        _train = channels_epochs_to_2d(_train, train_epochs_shape)
    elif dir == 'time':
        _train = time_epochs_to_2d(_train, train_epochs_shape)
    else:
        print("Wrong direction for normalization")
        sys.exit(-1)

    """ normalization """
    # fit to train
    scaler_train = StandardScaler()
    scaler_train.fit(_train)
    del _train, train_epochs, train_epochs_info, train_epochs_shape

    # normalize train
    normalized_train = []
    for train_epoch in train:
        train_epoch_info = train_epoch.info
        train_epoch_event_id = train_epoch.event_id
        _train = train_epoch.get_data(copy=True)
        train_epoch_shape = _train.shape  # (epochs, channels, time)
        if dir == 'channels':
            _train = channels_epochs_to_2d(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_epochs_to_2d(_train, train_epoch_shape)

        _train = scaler_train.transform(_train)

        if dir == 'channels':
            _train = channels_2d_to_epochs(_train, train_epoch_shape)
        elif dir == 'time':
            _train = time_2d_to_epochs(_train, train_epoch_shape)

        _train_epoch = mne.EpochsArray(_train, train_epoch_info,
                                       events=train_epoch.events,
                                       event_id=train_epoch_event_id)
        normalized_train.append(_train_epoch)

    # normalize test
    normalized_test = []
    for test_epoch in test:
        test_epoch_info = test_epoch.info
        test_epoch_event_id = test_epoch.event_id
        _test = test_epoch.get_data(copy=True)
        test_epoch_shape = _test.shape  # (epochs, channels, time)
        if dir == 'channels':
            _test = channels_epochs_to_2d(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_epochs_to_2d(_test, test_epoch_shape)

        scaler_test = StandardScaler()
        scaler_test.fit(_test)
        _test = scaler_test.transform(_test)

        if dir == 'channels':
            _test = channels_2d_to_epochs(_test, test_epoch_shape)
        elif dir == 'time':
            _test = time_2d_to_epochs(_test, test_epoch_shape)

        _test_epoch = mne.EpochsArray(_test, test_epoch_info,
                                      events=test_epoch.events,
                                      event_id=test_epoch_event_id)
        normalized_test.append(_test_epoch)

    return (normalized_train, normalized_test)
