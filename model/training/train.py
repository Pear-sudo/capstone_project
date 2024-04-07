import concurrent.futures
import inspect
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
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


def train_one_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                    model_name, model_f: Callable[[], tf.keras.models.Sequential],
                    input_width: int, conv_width: int,
                    stock_level_dict: dict[int, list[str]],
                    check_dir: Path, result_saving_dir: Path,
                    macros: list[str],
                    model_count: int, total_model_count: int,
                    is_testing: bool = False,
                    ignore_existing: bool = False,
                    ):
    """
    Train a model on each stock.
    :param ignore_existing: if true, ignore existing result and train the model again
    :param train: training dataframe
    :param val: validation dataframe
    :param test: testing dataframe
    :param model_name: Name of the model
    :param model_f: A FUNCTION that returns a model
    :param input_width:
    :param conv_width:
    :param stock_level_dict: A dict containing all stocks need to be trained and their stock level variables
    :param check_dir: The directory to save the checkpoints (give me the general path and I'll create subdir for you)
    :param result_saving_dir: The directory to save the r score, labels and predictions
    :param macros: A list of macro variables used to train the model.
    :param model_count: The current model count, used for progress reporting
    :param total_model_count: Total number of models to train
    :param is_testing: Train the model on one stock and immediately return
    :return:
    """
    # first work out where to save our results
    # the file to save aggregated r score
    if not result_saving_dir.exists():
        result_saving_dir.mkdir()
    result_path = result_saving_dir.joinpath(f'{input_width}_{model_name}.txt')

    # the file to save the true values, predicted value pairs for later inspection
    subdir = result_saving_dir.joinpath(f'{input_width}')
    if not subdir.exists():
        subdir.mkdir(parents=True)
    csv_path = subdir.joinpath(f'{model_name}.csv')

    # test if the model has already been trained, we do this by only focusing on the final r score file
    if not ignore_existing:
        if result_path.exists():
            print(f'Model {model_name} with input width {input_width} has already been trained, '
                  f'the result r score is at {result_path}')
            return

    total_stock_count = len(stock_level_dict)
    closing_price_label = 'Clsprc'

    check_dir = check_dir.joinpath(str(input_width)).joinpath(model_name)
    if not check_dir.exists():
        check_dir.mkdir(parents=True)

    print(f'Training model {model_name}')
    result = {
        'true_values': [],
        'predicted_values': [],
    }

    stock_count = 1
    for stock_id, stock_vars in stock_level_dict.items():

        id_str = '{:06}'.format(stock_id)
        print(f'Training stock {id_str} using {model_name} '
              f'(S: {stock_count}/{total_stock_count}) '
              f'(M: {model_count}/{total_model_count}) '
              f'(L: {input_width}) '
              f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')

        check_path = check_dir.joinpath(id_str)
        target_label = f'{closing_price_label}_{stock_id}'

        selectors = stock_vars + macros

        train_i = train[selectors]
        val_i = val[selectors]
        test_i = test[selectors]

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

        compile_and_fit(model_f(), window, seed=0, patience=10, model_save_path=check_path, verbose=0)
        model_trained: tf.keras.models.Sequential = load_model(check_path)  # load the best

        true_values, predicted_values = extract_labels_predictions(model_trained, window)
        result['true_values'].extend(true_values)
        result['predicted_values'].extend(predicted_values)

        r = calculate_r2_score(true_values, predicted_values)
        print(f'R score: {r}')

        stock_count += 1

        if is_testing:
            return

        # end for loop for individual stocks

    # calculate the aggregated r score
    true_values, predicted_values = result['true_values'], result['predicted_values']
    r = calculate_r2_score(true_values, predicted_values)
    print(f'R score of model {model_name} of width {input_width}: {r}')

    # save the r score of the model to a txt file
    with open(result_path, 'w') as f:
        f.write(str(r))

    # also backup the true values and predicted values to a csv file
    pd.DataFrame(result).to_csv(csv_path, index=False)

    # end train_one_model()


def train_with_fixed_input_width(input_width: int = 7,
                                 is_testing=False,
                                 model_dict: dict[str, Callable[[], Sequential]] | None = None,
                                 stock_filter: list[str] = None):
    input_width = input_width
    conv_width = 3  # this must match the setting in the conv neural network model

    # no worry about their existence, train_one_model() will handle it
    result_dir = Path('../checkpoints/result')
    check_dir = Path('../checkpoints/main')

    # get the name of all macro variables
    macros_all = (head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/merge/macro_test.csv'), 1)[0]
                  .split(','))
    macros_all.remove('Date')

    # get the name of macro variables at daily level
    macros_daily = \
        head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/macro/raw_data_daily_filled_test.csv'), 1)[
            0].split(',')
    macros_daily.remove('Date')

    # get names of the stock and their corresponding stock level variables
    stock_level_dict = get_stock_level_dict()
    if stock_filter is not None:
        stock_filter = [str(int(s)) for s in stock_filter]
        stock_level_dict = {s: stock_level_dict[s] for s in stock_filter if s in stock_level_dict}
    stock_all = []
    for v in stock_level_dict.values():
        stock_all.extend(v)

    # load the dataset
    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test.csv')

    # get all the models we need to train
    if model_dict is None:
        model_dict = get_all_models_f()
    total_models = len(model_dict)

    model_count = 1
    max_parallel_tasks = 6
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = []
        results = []

        for model_name, model_f in model_dict.items():
            if len(futures) >= max_parallel_tasks:
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    results.append(future.result())
                    futures.remove(future)
                    model_count += 1

            future = executor.submit(train_one_model,
                                     train.copy(), val.copy(), test.copy(),
                                     model_name, model_f,
                                     input_width, conv_width,
                                     stock_level_dict,
                                     check_dir, result_dir,
                                     macros_all,
                                     model_count, total_models,
                                     is_testing)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())


def train_with_multi_sizes(is_testing=False):
    size = [7, 14, 28, 48]
    for size in size:
        train_with_fixed_input_width(size, is_testing=is_testing)


if __name__ == '__main__':
    train_with_fixed_input_width(48, model_dict={'nnn3': dense.nnn3})
