import tensorflow as tf
import os
import pickle as pk
import numpy as np
from os.path import join
from Layers import QR_sparse, Sparse_Dense


def QR_conf(units=1000,
            activation='tanh',
            is_eig_in_trainable=True,
            is_eig_out_trainable=True,
            use_bias=True
            ):
    return {'units': units,
            'activation': activation,
            'is_eig_in_trainable': is_eig_in_trainable,
            'is_eig_out_trainable': is_eig_out_trainable,
            'use_bias': use_bias
            }


def load_dataset(config, grup_channels=True):
    if config['dataset'] == 'CIFAR10':
        dataset = tf.keras.datasets.cifar10
        in_shape = 32 * 32 * 3
    elif config['dataset'] == 'F-MNIST':
        dataset = tf.keras.datasets.fashion_mnist
        in_shape = 28 * 28
    elif config['dataset'] == 'MNIST':
        dataset = tf.keras.datasets.mnist
        in_shape = 28 * 28

    if grup_channels:
        o = 'F'
    else:
        o = 'C'

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    flat_train = np.reshape(x_train, [x_train.shape[0], in_shape], order=o)  # Tutto dataset
    flat_test = np.reshape(x_test, [x_test.shape[0], in_shape], order=o)  # Tutto testset
    return (flat_train, y_train), (flat_test, y_test)


def sparse_model_build(config):
    model = tf.keras.Sequential()
    if config['dataset'] == 'CIFAR10':
        model.add(tf.keras.Input(shape=(32 * 32 * 3,)))
    else:
        model.add(tf.keras.Input(shape=(28 * 28,)))

    if config['type'] == 'QR_sparse':

        hid_parameters = QR_conf(units=config['n2'])
        last_parameters = QR_conf(units=10, activation='softmax')

        for i in range(config['hidden_layers']):
            model.add(QR_sparse(**hid_parameters,
                                dynamic_sparse=config['dyn_sparse'],
                                percentile=config['percentile']))
        model.add(QR_sparse(**last_parameters,
                            dynamic_sparse=config['dyn_sparse'],
                            percentile=config['percentile']))

        return model


    elif config['type'] == 'Dense_sparse':

        for i in range(config['hidden_layers']):
            model.add(Sparse_Dense(units=config['n2'],
                                   activation='tanh',
                                   use_bias=False,
                                   dynamic_sparse=config['dyn_sparse'],
                                   percentile=config['percentile']))

        model.add(Sparse_Dense(units=10,
                               activation="softmax",
                               use_bias=False,
                               dynamic_sparse=config['dyn_sparse'],
                               percentile=config['percentile']))
        return model



def sparse_training(configur):
    if configur['dataset'].find('CIFAR') != -1:
        lr = {'Dense': 0.0001, 'Dense_sparse': 0.0001, 'QR': 0.003, 'QR_sparse': 0.003}
    elif configur['dataset'] == 'F-MNIST':
        lr = {'Dense': 0.001, 'Dense_sparse': 0.001, 'QR': 0.001, 'QR_sparse': 0.001}
    elif configur['dataset'] == 'MNIST':
        lr = {'Dense': 0.005, 'Dense_sparse': 0.005, 'QR': 0.005, 'QR_sparse': 0.005}

    (x, y), (xtest, ytest) = load_dataset(configur)

    model = sparse_model_build(configur)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  min_delta=0.005,
                                                  mode='max',
                                                  patience=5)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr[configur['type']]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=False)
    print('\nStart model.fit, layer type = {}'.format(configur['type'], configur['type']))

    model.fit(x, y,
              validation_split=0.3,
              epochs=configur['epochs'],
              batch_size=configur['batch_size'],
              callbacks=stop_early,
              verbose=0)

    R = model.evaluate(xtest, ytest, batch_size=1000, verbose=1)
    folder = join(configur['path'], configur['dataset'])
    os.makedirs(folder, exist_ok=True)

    # Dynamic sparse
    n = configur['type'] + '_' + str(configur[
                                         'percentile']) + '.p'
    name = join(folder, n)

    with open(name, "ab") as f:
        pk.dump(R[1], f)
        print('\nResults saved in:\n {}'.format(name))

