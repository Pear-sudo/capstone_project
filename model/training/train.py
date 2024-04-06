import inspect
import os
import platform
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

import model.networks.cnn as cnn
import model.networks.dense as dense
import model.networks.linear as linear
from model.loader import head
from model.networks.dense import n3
from model.preprocessing import Preprocessor, StockLoadingStrategy, test_stock_number
from model.window import WindowGenerator

"""
This module groups training related functions.
"""

MAX_EPOCHS = 100
PATIENCE = 10

out_dir: Path = Path('../../out/training')
if not out_dir.exists():
    out_dir.mkdir()


def extract_functions_to_dict(module) -> dict:
    functions_dict = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        functions_dict[name] = func
    return functions_dict


def get_all_models() -> dict[str, tf.keras.models.Sequential]:
    # d dropout; c cnn; n neural network
    d = {
        'n': n3(),
        # 'c': get_cnn(),
        # 'linear': get_liner(),
    }
    return d


def get_all_models_f() -> dict[str, Callable[[], tf.keras.models.Sequential]]:
    d_dense = extract_functions_to_dict(dense)
    d_cnn = extract_functions_to_dict(cnn)
    d_linear = extract_functions_to_dict(linear)
    return d_dense | d_cnn | d_linear


def extract_labels_predictions(model, window: WindowGenerator) -> tuple[list, list]:
    """

    :param model:
    :param window:
    :return: true_values, predicted_values
    """
    predictions = list(model.predict(window.test).flatten())
    true_values = []
    input_values = []
    for inputs, labels in window.test:
        true_values.extend(labels.numpy().flatten())
        input_values.append(inputs.numpy())
    return true_values, predictions


def calculate_r2_score(true_values: list, predictions: list) -> float:
    # true_values = np.array(true_values)
    # predictions = np.array(predictions)
    # r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum(true_values ** 2)
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
                    verbose: int = 1,
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

    learning_rate = 0.001
    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=callbacks, verbose=verbose)

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

    dense = n3()

    compile_and_fit(dense, window, seed=0, patience=10)


def individually(input_width: int = 7):
    input_width = input_width
    conv_width = 3

    result_dir = Path('../checkpoints/result')
    if not result_dir.exists():
        result_dir.mkdir()

    check_dir = Path('../checkpoints/main')

    macros_all = (head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/merge/macro_test.csv'), 1)[0]
                  .split(','))
    macros_all.remove('Date')

    macros_daily = \
        head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/macro/raw_data_daily_filled_test.csv'), 1)[
            0].split(',')
    macros_daily.remove('Date')

    closing_price_label = 'Clsprc'
    stock_level_dict = get_stock_level_dict()
    stock_all = []
    for v in stock_level_dict.values():
        stock_all.extend(v)

    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test.csv')

    # train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train_random.csv')
    # val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val_random.csv')
    # test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test_random.csv')

    # train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/stock/raw_data_daily_filled_train.csv')
    # val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/stock/raw_data_daily_filled_val.csv')
    # test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/stock/raw_data_daily_filled_test.csv')

    all_models_fs = get_all_models_f()
    total_models = len(all_models_fs)
    model_count = 1
    for model_name, model_f in all_models_fs.items():

        print(f'Training model {model_name}')
        result = {
            'true_values': [],
            'predicted_values': [],
        }
        check_path_model = check_dir.joinpath(model_name)

        stock_count = 1
        for stock_id, stock_vars in stock_level_dict.items():
            total_stock = len(stock_level_dict)

            id_str = '{:06}'.format(stock_id)
            print(
                f'Training stock {id_str} using {model_name} (S: {stock_count}/{total_stock}) M: {model_count}/{total_models}')

            check_path_model_stock = check_path_model.joinpath(id_str)
            target_label = f'{closing_price_label}_{stock_id}'

            selectors = stock_vars + macros_all
            # selectors = stock_vars

            train_i = train[selectors]
            val_i = val[selectors]
            test_i = test[selectors]

            # check_path_model_stock = Path('../checkpoints/testing')
            # data_path = Path('../../out/test/testing_data.csv')
            #
            # data_df = pd.read_csv(data_path)
            # preprocessor = Preprocessor(StockLoadingStrategy())
            # train, val, test = preprocessor.split_to_dataframes(data_df)
            #
            # train_i = train
            # val_i = val
            # test_i = train
            # target_label = 'x'

            if 'c' in model_name:
                window = WindowGenerator(input_width,
                                         get_input_label_width_cnn(input_width, conv_width)[1],
                                         1,
                                         [target_label],
                                         train_df=train_i, val_df=val_i, test_df=test_i)
            else:
                window = WindowGenerator(input_width,
                                         1,
                                         1,
                                         [target_label],
                                         train_df=train_i, val_df=val_i, test_df=test_i)

            compile_and_fit(model_f(), window, seed=0, patience=10, model_save_path=check_path_model_stock, verbose=0)
            model_trained: tf.keras.models.Sequential = load_model(check_path_model_stock)  # load the best

            true_values, predicted_values = extract_labels_predictions(model_trained, window)
            result['true_values'].extend(true_values)
            result['predicted_values'].extend(predicted_values)
            r = calculate_r2_score(true_values, predicted_values)
            print(f'R score: {r}')
            stock_count += 1
            # break

        subdir = result_dir.joinpath(f'{input_width}')
        if not subdir.exists():
            subdir.mkdir()
        csv_path = subdir.joinpath(f'{model_name}.csv')

        r = calculate_r2_score(result['true_values'], result['predicted_values'])
        print(f'R score of model {model_name} of width {input_width}: {r}')
        r_path = result_dir.joinpath(f'{input_width}_{model_name}.txt')
        with open(r_path, 'w') as f:
            f.write(str(r))

        pd.DataFrame(result).to_csv(csv_path, index=False)
        model_count += 1


if __name__ == '__main__':
    individually()
