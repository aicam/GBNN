import tensorflow as tf
from tensorflow import keras


class MMLayer(keras.layers.Layer):

    def __init__(self):
        super(MMLayer, self).__init__()
        self.alpha = tf.Variable(initial_value=tf.ones(1), trainable=True)

    def call(self, inputs):
        multplied_ew = tf.multiply(inputs[:, :, :1], inputs[:, :, 1:2])
        return tf.concat([tf.multiply(tf.exp(multplied_ew), self.alpha),
                          tf.reshape(inputs[:, :, 2], [inputs.shape[0], inputs.shape[1], 1])], axis=2)


class GBLayer(keras.layers.Layer):

    def __init__(self, ):
        super(GBLayer, self).__init__()
        self.alpha = tf.Variable(initial_value=tf.ones(1), trainable=True)
        self.beta = tf.Variable(initial_value=tf.zeros(1), trainable=True)
        self.gamma = tf.Variable(initial_value=tf.ones(1), trainable=True)

    def call(self, inputs):
        m_matrix = inputs[:, :, 0]
        q_matrix = inputs[:, :, 1]
        return tf.multiply(tf.multiply(1 / m_matrix, tf.multiply(q_matrix, self.alpha)),
                           tf.exp((tf.math.pow(m_matrix, 2) - tf.math.pow(self.beta, 2)) / tf.math.pow(self.gamma, 2)))


class FilterLayer(keras.layers.Layer):

    def __init__(self, input_dim=None):
        super(FilterLayer, self).__init__()
        self.w = tf.Variable(initial_value=tf.random.normal(shape=input_dim), trainable=True)

    def call(self, inputs):
        return tf.reshape(tf.matmul(inputs, self.w), [1, inputs.shape[0], self.w.shape[1], 1])
