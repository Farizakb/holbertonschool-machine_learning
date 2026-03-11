#!/usr/bin/env python3
"""
    Train with Learning Rate Decay
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
        Function that trains a model using mini-batch gradient descent

        :param network: model to train
        :param data: ndarray, shape(m, nx), input data
        :param labels: ndarray, shape(m,classes), labels
        :param batch_size: size of the batch
        :param epochs: number of passes through data for mini-bath
        :param validation_data: data to validate the model
        :param early_stopping: boolean, use or not early stopping
        :param patience: patience for early stopping
        :param learning_rate_decay: boolean, use or not learning rate decay
        :param alpha: initial learning rate
        :param decay_rate: decay rate
        :param save_best: boolean, save best model or not
        :param filepath: path to save the model
        :param verbose: boolean, print or not during training
        :param shuffle: boolean, shuffle or not every epoch

        :return: History
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data:
        def scheduler(epochs):
            lr = alpha / (1 + decay_rate * epochs)
            return lr

        inv_time_decay = K.callbacks.LearningRateScheduler(scheduler,
                                                           verbose=1)

        callbacks.append(inv_time_decay)

    if save_best:
        save_best_model = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(save_best_model)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,    
                          batch_size=batch_size,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)

    return history
