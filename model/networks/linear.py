import tensorflow as tf


def get_liner():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
