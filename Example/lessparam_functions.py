import tensorflow as tf
import numpy as np
import pickle as pk
import os
from Layers import SSVD, QR


def Spec_full_conf(units=1000,
                   activation='relu',
                   is_eig_in_trainable=True,
                   is_eig_out_trainable=True,
                   is_svd_trainable=False,
                   use_bias=True):
    """
    :return: configuration for the Spectral layer
    :rtype: dict
    """
    return {'units': units,
            'activation': activation,
            'is_eig_in_trainable': is_eig_in_trainable,
            'is_eig_out_trainable': is_eig_out_trainable,
            'is_svd_trainable': is_svd_trainable,
            'use_bias': use_bias}


def SSVD_full_conf(units=1000,
                   activation='relu',
                   is_eig_in_trainable=True,
                   is_eig_out_trainable=True,
                   is_svd_trainable=True,
                   use_bias=True):
    """
    :return: configuration for the S-SVD layer
    :rtype: dict
    """
    return {'units': units,
            'activation': activation,
            'is_eig_in_trainable': is_eig_in_trainable,
            'is_eig_out_trainable': is_eig_out_trainable,
            'is_svd_trainable': is_svd_trainable,
            'use_bias': use_bias,
            'eig_in_initializer': 'zeros',
            'eig_out_initializer': 'ones'
            }


def QR_conf(units=1000,
            activation='relu',
            is_eig_in_trainable=True,
            is_eig_out_trainable=True,
            use_bias=True
            ):
    """
    :return: configuration for the QR layer
    :rtype: dict
    """
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

    flat_train = np.reshape(x_train, [x_train.shape[0], in_shape], order=o)
    flat_test = np.reshape(x_test, [x_test.shape[0], in_shape], order=o)

    return (flat_train, y_train), (flat_test, y_test)


def build_model(config):
    model = tf.keras.Sequential()

    # Input Layer
    if config['dataset'] == 'CIFAR10':
        model.add(tf.keras.Input(shape=(32 * 32 * 3,)))
    else:
        model.add(tf.keras.Input(shape=(28 * 28,)))

    # Hidden Layers
    if config['type'] == 'Dense':
        for i in range(config['hidden_layers']):
            model.add(tf.keras.layers.Dense(config['n2'],
                                            activation='relu',
                                            use_bias=True))
        model.add(tf.keras.layers.Dense(10,
                                        activation="softmax",
                                        use_bias=True))
        return model


    elif config['type'] == 'Spectral_full':
        hid_parameters = Spec_full_conf(units=config['n2'])
        last_parameters = Spec_full_conf(units=10, activation='softmax')


    elif config['type'] == 'SSVD_full':
        hid_parameters = SSVD_full_conf(units=config['n2'])
        last_parameters = SSVD_full_conf(units=10, activation='softmax')

    elif config['type'] == 'QR':
        hid_parameters = QR_conf(units=config['n2'])
        last_parameters = QR_conf(units=10, activation='softmax')
        for i in range(config['hidden_layers']):
            model.add(QR(**hid_parameters, density=config['density']))
        model.add(QR(**last_parameters, density=config['density']))
        return model

    for i in range(config['hidden_layers']):
        model.add(SSVD(**hid_parameters))
    model.add(SSVD(**last_parameters))

    return model


def train_model(config):
    (flat_train, y_train), (flat_test, y_test) = load_dataset(config)
    model = build_model(config)

    if config['type'] == 'Dense':
        if config['dataset'] == 'F-MNIST':
            lr = 0.001
        elif config['dataset'].find('CIFAR') != -1:
            lr = 0.002
        else:
            lr = config['learning_rate']
    elif config['type'] == 'QR':
        if config['dataset'].find('CIFAR') != -1:
            lr = 0.003
        elif config['dataset'] == 'F-MNIST':
            lr = 0.001
        elif config['dataset'] == 'MNIST':
            lr = 0.005
    else:
        lr = config['learning_rate']

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=False)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  min_delta=0.005,
                                                  mode='max',
                                                  patience=8)
    print('\nStart model.fit, n = {}, layer type = {}'.format(config['n2'], config['type']))
    model.fit(flat_train,
              y_train,
              epochs=config['epochs'],
              validation_split=0.2,
              batch_size=config['batch_size'],
              callbacks=[stop_early],
              verbose=0)

    outcome = model.evaluate(flat_test, y_test, batch_size=1000, verbose=1)
    folder = os.path.join(config['save_path'], config['dataset'])
    os.makedirs(folder, exist_ok=True)
    name = os.path.join(folder, config['type'] + str(config['n2']) + '.p')

    with open(name, "ab") as f:
        pk.dump(outcome[1], f)
    print('\nResults saved in:\n {}'.format(name))
    return model
