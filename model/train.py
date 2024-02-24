import platform

import tensorflow as tf

from model.window import WindowGenerator

MAX_EPOCHS = 1000
PATIENCE = 10


def compile_and_fit(model: tf.keras.Model, window: WindowGenerator, patience: int = PATIENCE,
                    max_epochs: int = MAX_EPOCHS):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')  # for fit/training process only

    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam()
    else:
        opt = tf.keras.optimizers.Adam()

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
