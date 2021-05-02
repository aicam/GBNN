import numpy as np
import tensorflow as tf


class TransLayer(tf.keras.layers.Layer):


    def __init__(self, input_dim=None):
        super(TransLayer, self).__init__()
        if input_dim is None:
            input_dim = [4000, 4000, 12]
        self.w = tf.Variable()
