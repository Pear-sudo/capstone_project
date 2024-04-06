import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

from model.loader import head
from model.networks.cnn import get_cnn
from model.networks.dense import get_dense
from model.preprocessing import Preprocessor, StockLoadingStrategy, test_stock_number
from model.window import WindowGenerator

"""
This module groups training related functions.
"""

MAX_EPOCHS = 3
PATIENCE = 10

out_dir: Path = Path('../../out/training')
if not out_dir.exists():
    out_dir.mkdir()


def get_all_models() -> dict[str, tf.keras.models.Sequential]:
    # d dropout; c cnn; n neural network
    d = {
        'n': get_dense(),
        'c': get_cnn(),
        # 'linear': get_liner(),
    }
    return d


def extract_labels_predictions(model, window: WindowGenerator) -> tuple[list, list]:
    """

    :param model:
    :param window:
    :return: true_values, predicted_values
    """
    predictions = list(model.predict(window.test).flatten())
    true_values = []
    # input_values = []
    for inputs, labels in window.test:
        true_values.extend(labels.numpy().flatten())
        # input_values.append(list(inputs.numpy().reshape(inputs.shape[0], -1)))
    return true_values, predictions


def calculate_r2_score(true_values: list, predictions: list) -> float:
    r2 = r2_score(true_values, predictions)
    return r2


def load_model(model_path: Path):
    model = tf.keras.models.load_model(str(model_path.absolute()))
    return model


def compile_and_fit(model: tf.keras.Model,
                    window: WindowGenerator,
                    patience: int = PATIENCE,
                    max_epochs: int = MAX_EPOCHS,
                    seed: int | None = None,
                    model_save_path: Path | None = None,
                    ):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    if model_save_path and not model_save_path.exists():
        model_save_path.mkdir(exist_ok=False, parents=True)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience,
                                                  mode='min')]

    if model_save_path:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(model_save_path.absolute()),
                                                            save_best_only=True,
                                                            monitor='val_loss',
                                                            mode='min'))

    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam()
    else:
        opt = tf.keras.optimizers.Adam()

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=callbacks)

    return model


def get_input_label_width_cnn(input_width: int, conv_width: int) -> tuple[int, int]:
    """

    :param input_width:
    :param conv_width:
    :return: input width, label width
    """
    offset = conv_width - 1
    return input_width, input_width - offset


def train_test_data():
    input_width = 3
    conv_width = 3
    check_dir = Path('../checkpoints/testing')
    data_path = Path('../../out/test/testing_data.csv')
    if not data_path.exists():
        pass

    data_df = pd.read_csv(data_path)
    preprocessor = Preprocessor(StockLoadingStrategy())
    train, val, test = preprocessor.split_to_dataframes(data_df)

    ordinary_window: WindowGenerator | None = None
    cnn_window: WindowGenerator | None = None
    window: WindowGenerator | None = None

    for model_name, model in get_all_models().items():
        if 'c' in model_name:
            if cnn_window is None:
                cnn_window = WindowGenerator(input_width, get_input_label_width_cnn(input_width, conv_width)[1], 1,
                                             ['x'],
                                             train_df=train, val_df=val, test_df=test)
            window = cnn_window
        else:
            if ordinary_window is None:
                ordinary_window = WindowGenerator(input_width, 1, 1, ['x'],
                                                  train_df=train, val_df=val, test_df=test)
            window = ordinary_window

        print(f'Training model {model_name}')
        check_path = check_dir.joinpath(model_name)

        model_trained = compile_and_fit(model, window, seed=0, model_save_path=check_path)

        model_path = Path(f'/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints/testing/{model_name}')
        model_trained: tf.keras.models.Sequential = load_model(model_path)
        r = calculate_r2_score(*extract_labels_predictions(model_trained, window))
        print(f'R score: {r}')


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

    window = WindowGenerator(21, 1, 1, get_labels(),
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
