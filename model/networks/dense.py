import tensorflow as tf


def get_dense():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=3, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='relu'),
        tf.keras.layers.Dense(units=1),
        tf.keras.layers.Reshape([1, -1]),
    ])


def get_multi_step_dense():
    return tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        # I hope from here you will understand why in the notebook,
        # I train the model use single or suited window size (i.e. given one time, predict one time)
        # There is no maic here: you need to comply to model's input size
        tf.keras.layers.Flatten(),
        # Note: After training, the input of the following layer is fixed to time * features
        # That's why in the notebook, the wide window failed to plot the prediction.
        # But the second attempt with eactly the same size succeeded!
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        # -1 in Reshape means "infer this dimension"
        tf.keras.layers.Reshape([1, -1]),
    ])
