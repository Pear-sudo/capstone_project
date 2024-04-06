import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model.loader import head
from model.networks.dense import get_dense
from model.preprocessing import Preprocessor, StockLoadingStrategy, test_stock_number
from model.window import WindowGenerator

"""
This module groups training related functions.
"""

MAX_EPOCHS = 1000
PATIENCE = 10

out_dir: Path = Path('../../out/training')
if not out_dir.exists():
    out_dir.mkdir()


def compile_and_fit(model: tf.keras.Model,
                    window: WindowGenerator,
                    patience: int = PATIENCE,
                    max_epochs: int = MAX_EPOCHS,
                    seed: int | None = None
                    ):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')  # for fit/training process only

    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam()
    else:
        opt = tf.keras.optimizers.Adam()

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def train_test_data():
    data_path = Path('../../out/test/testing_data.csv')
    if not data_path.exists():
        pass
    data_df = pd.read_csv(data_path)
    preprocessor = Preprocessor(StockLoadingStrategy())
    train, val, test = preprocessor.split_to_dataframes(data_df)

    window = WindowGenerator(1, 1, 1, ['z'],
                             train_df=train, val_df=val, test_df=test)
    dense = get_dense()

    compile_and_fit(dense, window, seed=0)


def get_stock_level_dict() -> dict:
    stock_level_vars = ['Opnprc',
                        'Hiprc',
                        'Loprc',
                        'Clsprc',
                        'Dnshrtrd',
                        'Dnvaltrd',
                        'Dsmvosd',
                        'Dsmvtll',
                        'Dretwd',
                        'Dretnd',
                        'Adjprcwd',
                        'Adjprcnd',
                        'PreClosePrice',
                        'ChangeRatio'
                        ]

    ids = test_stock_number()
    stock_level_dict: dict[str, list[str]] = {}
    for i in ids:
        stock_level_labels = [f'{v}_{i}' for v in stock_level_vars]
        stock_level_dict[i] = stock_level_labels

    return stock_level_dict


def get_labels() -> list[str]:
    ids = test_stock_number()
    closing_price_label = 'Clsprc'
    labels = [f'{closing_price_label}_{i}' for i in ids]
    return labels


def altogether():
    closing_price_label = 'Clsprc'

    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test.csv')

    labels = get_labels()

    window = WindowGenerator(1, 1, 1, get_labels(),
                             train_df=train, val_df=val, test_df=test)

    dense = get_dense()

    compile_and_fit(dense, window, seed=0, patience=10)


def individually():
    closing_price_label = 'Clsprc'

    macros = (head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/merge/macro_test.csv'), 1)[0]
              .split(','))
    macros.remove('Date')

    stock_level_dict = get_stock_level_dict()
    labels = get_labels()

    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test.csv')

    i = '2'
    single = [f'{closing_price_label}_{i}']
    train_i = train[macros + stock_level_dict[i]]
    val_i = val[macros + stock_level_dict[i]]
    test_i = test[macros + stock_level_dict[i]]

    window = WindowGenerator(1, 1, 1, single,
                             train_df=train_i, val_df=val_i, test_df=test_i)

    dense = get_dense()

    compile_and_fit(dense, window, seed=0, patience=10)


if __name__ == '__main__':
    train_test_data()
