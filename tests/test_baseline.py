from model.window import *
from model.loader import *
from model.stocks import *
from training.baseline import Baseline
import tensorflow as tf


def test_call():
    df = load_normalized_dataset('./data/sample.csv')
    w = WindowGenerator(1, 1, 1, [StockColumn.clsprc.name],
                        *split_to_dataframes(df))

    baseline = Baseline(label_index=w.column_indices[StockColumn.clsprc.name])
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(w.val)
    # performance['Baseline'] = baseline.evaluate(w.test, verbose=0)
