import tensorflow as tf

CONV_WIDTH = 3


def cn3(conv_width):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=3,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=3, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])


def cn5(conv_width):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=5,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=5, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])


def cn7(conv_width):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=7,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=7, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])


def cnd7(conv_width):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=7,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=7, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1),
    ])

# def cn32():
#     return tf.keras.Sequential([
#         tf.keras.layers.Conv1D(filters=32,
#                                kernel_size=(CONV_WIDTH,),
#                                activation='relu'),
#         tf.keras.layers.Dense(units=32, activation='relu'),
#         tf.keras.layers.Dense(units=1),
#     ])
#
#
# def cn64():
#     return tf.keras.Sequential([
#         tf.keras.layers.Conv1D(filters=64,
#                                kernel_size=(CONV_WIDTH,),
#                                activation='relu'),
#         tf.keras.layers.Dense(units=64, activation='relu'),
#         tf.keras.layers.Dense(units=1),
#     ])
