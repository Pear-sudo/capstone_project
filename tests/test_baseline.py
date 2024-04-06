import tensorflow as tf

from model.networks.baseline import Baseline
from model.stocks import *
from model.window import *


def test_call():
    df = load_normalized_dataset('./data/sample.csv')
    w = WindowGenerator(24, 24, 1, [StockColumn.clsprc.name],
                        *split_to_dataframes(df))

    baseline = Baseline(label_index=w.column_indices[StockColumn.clsprc.name])
    example_input = w.example[0]
    # below is the prediction, its shape matches the shape of example's label
    assert baseline(example_input).shape == (32, 24, 1)
    assert baseline(example_input).shape == w.example[1].shape

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(w.val, verbose=0)
    performance['Baseline'] = baseline.evaluate(w.test, verbose=0)
