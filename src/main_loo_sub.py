#!/usr/bin/env python3
"""
1. import all sessions from each subject
"""

import os
import sys

from keras import activations, layers, losses, optimizers
from keras.models import Model, load_model
from keras.metrics import Accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import to_categorical, set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import normalization
from neural_networks import base_cnn


SEED = 73
SUB = int(sys.argv[1])
NUM_SES = 5


""" set random seeds """
tf.random.set_seed(SEED) # Set the seed for TensorFlow
np.random.seed(SEED) # Set the seed for NumPy
set_random_seed(SEED) # Set the seed for Keras

""" import session files """
sessions = []
dir_path = os.path.join("..", "preprocess_first", f"sub_{SUB:02d}")
for session in range(1, NUM_SES + 1):
    fname = os.path.join(dir_path, f"ses_{session:02d}",
                         "epochs_preprocessed-epo.fif")
    epochs_ses = mne.read_epochs(fname, preload=True)
    sessions.append(epochs_ses)

""" leave one session out """
train_accs = []
val_accs = []
accs = []
for ses in range(NUM_SES):
    train = [*sessions[:ses], *sessions[ses + 1:]]
    test = [sessions[ses]]

    """ normalization """
    train_epochs, test_epochs = normalization.norm_2(train, test,
                                                     dir='channels')

    # mne.concatenate_epochs(train).average().plot(show=False)
    # mne.concatenate_epochs(test).average().plot(show=False)
    """ concatenate epochs """
    X_train = mne.concatenate_epochs(train_epochs)
    X_test = mne.concatenate_epochs(test_epochs)
    y_train = X_train.events[:, -1] - 1  # convert to [0, 1]
    y_test = X_test.events[:, -1] - 1

    # plt.show()
    # continue

    """ retrieve data from epochs objects and change to keras data_format """
    tmin = 0.1
    tmin = 1
    tmax = 2
    tmax = 3
    X_train = X_train.get_data(picks='eeg', tmin=tmin, tmax=tmax, copy=False)
    X_test = X_test.get_data(picks='eeg', tmin=tmin, tmax=tmax, copy=False)

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    """ convert events to one-hot """
    y_train_oh = to_categorical(y_train, num_classes=2)

    """ model """
    _, time_samples, eeg_channs = X_train.shape
    model = base_cnn(input_shape=(time_samples, eeg_channs), activ='tanh')

    """ hyperparameters and compilation """
    lr = 4e-5
    lr = 2e-5
    n_epochs = 100
    n_epochs = 1000
    patience = n_epochs / 10
    bs = 8

    _optim = optimizers.SGD(learning_rate=lr)
    _loss = losses.CategoricalCrossentropy(from_logits=False)
    _fpath = f"../best_models/sub_{SUB:02d}_sestest_{ses:02d}.keras"
    _best_model = ModelCheckpoint(_fpath, monitor='val_loss', mode='min',
                                  save_best_only=True)
    _early_stopping = EarlyStopping(monitor='val_loss',
                                    patience=patience,
                                    mode='min',
                                    verbose='1',
                                    restore_best_weights=True)

    """ model compilation and training """
    model.compile(optimizer=_optim, loss=_loss, metrics=['accuracy'])
    history = model.fit(X_train, y_train_oh, epochs=n_epochs, batch_size=bs,
                        validation_split=0.1, shuffle=True,
                        callbacks=[_best_model,])
                                   # _early_stopping])

    # summarize history for accuracy
    # plt.figure()
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # summarize history for loss
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    """ testing """
    model = load_model(_fpath)
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    best_epoch = np.argwhere(
        history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

    train_acc = history.history['accuracy'][best_epoch]
    val_acc = history.history['val_accuracy'][best_epoch]
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(y_test, predictions)
    accuracy_score = accuracy.result().numpy()


    accs.append(accuracy_score)


print("---- Fold results ----")
for ses, (train_acc, val_acc, acc) in enumerate(zip(train_accs, val_accs,
                                                    accs)):
    print(f"Session {ses + 1}, train: {train_acc:.4f} "
          + f"val: {val_acc:.4f} test: {acc:.4f}")

print()
print("---- General result ----")
print(f"Acc: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
