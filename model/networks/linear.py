import tensorflow as tf


def get_liner():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1),
        tf.keras.layers.Reshape([1, -1]),
    ])
