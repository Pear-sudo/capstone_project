import tensorflow as tf

from model.window import WindowGenerator

MAX_EPOCHS = 1000


def compile_and_fit(model: tf.keras.Model, window: WindowGenerator, patience: int = 2, max_epochs: int = MAX_EPOCHS):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')  # for fit/training process only

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
