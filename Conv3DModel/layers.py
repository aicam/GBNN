import numpy as np
import tensorflow as tf


class TransLayer(tf.keras.layers.Layer):


    def __init__(self, features=None):
        super(TransLayer, self).__init__()
        if features is None:
            features = 12
        self.features = features
        self.w = tf.Variable(initial_value=tf.ones(features), trainable=True)

    def cell_function_map(self, input):
        return tf.reshape(tf.tensordot(input, self.w, 1), [input.shape[0], input.shape[1], input.shape[2], 1])

    def call(self, inputs):
        return tf.map_fn(fn = lambda i: self.cell_function_map(i), elems=inputs)

