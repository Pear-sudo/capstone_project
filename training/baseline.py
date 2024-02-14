from typing import override

import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    @override
    def call(self, inputs, training=None, mask=None):
        if self.label_index is None:
            return inputs
        # (batch, time, features); here we simply select and return the current label in 'features' dimension
        result = inputs[:, :, self.label_index]
        # This is to maintain the original shape;
        # since you only take one element from the last dimension, it 'collapsed' into the second dimension,
        # which originally contained a list for dimension 3
        return result[:, :, tf.newaxis]
