from pathlib import Path

import pandas as pd

from model.loader import head
from model.networks.cnn import *
from model.training.train import get_stock_level_dict, compile_and_fit, extract_labels_predictions, calculate_r2_score, \
    get_input_label_width_cnn
from model.window import WindowGenerator


def get_debug_data():
    macros_all = (head(Path('/Users/a/PycharmProjects/capstone/capstone project/out/merge/macro_test.csv'), 1)[0]
                  .split(','))
    macros_all.remove('Date')
    closing_price_label = 'Clsprc_2'

    stock_level_dict = get_stock_level_dict()
    stock_filter = '2'
    if stock_filter is not None:
        stock_filter = [str(int(s)) for s in stock_filter]
        stock_level_dict = {s: stock_level_dict[s] for s in stock_filter if s in stock_level_dict}
    stock_all = []
    for v in stock_level_dict.values():
        stock_all.extend(v)

    features = stock_all

    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/out/test.csv')

    train_filtered = train[features]
    val_filtered = val[features]
    test_filtered = test[features]

    train_filtered.to_csv(Path('debug_train.csv'), index=False)
    val_filtered.to_csv(Path('debug_val.csv'), index=False)
    test_filtered.to_csv(Path('debug_test.csv'), index=False)


def get_true_labels():
    closing_price_label = 'Clsprc_2'
    df_t = pd.read_csv(Path('debug_test.csv'))
    df_l = df_t[closing_price_label]
    df_l.to_csv(Path('debug_labels.csv'), index=False)


def debug():
    train = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/model/training/debug_train.csv')
    val = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/model/training/debug_val.csv')
    test = pd.read_csv('/Users/a/PycharmProjects/capstone/capstone project/model/training/debug_test.csv')

    m = cn3()

    i_width = 7
    conv_width = 3
    l_width = get_input_label_width_cnn(i_width, conv_width)[1]
    wg = WindowGenerator(
        i_width,
        1,
        1,
        ['Clsprc_2'],
        train_df=train, val_df=val, test_df=test
    )

    total_window_size = wg.total_window_size
    input_slice = wg.input_slice
    labels_slice = wg.labels_slice

    label_columns = wg.label_columns
    column_indices = wg.column_indices

    input_width = wg.input_width
    label_width = wg.label_width

    test_ds = wg.test
    input_tensor = []
    output_tensor = []
    for batch in test_ds:
        input_tensor.append(batch[0])
        output_tensor.append(batch[1])

    m_trained = compile_and_fit(m, wg, max_epochs=500)
    l, p = extract_labels_predictions(m_trained, wg)
    r2 = calculate_r2_score(l, p)
    pass


if __name__ == '__main__':
    debug()
