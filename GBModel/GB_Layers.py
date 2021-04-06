import tensorflow as tf
from tensorflow import keras


class MMLayer(keras.layers.Layer):

    def __init__(self, input_dim):
        super(MMLayer, self).__init__()
        self.alpha = tf.Variable(initial_value=tf.ones(1), trainable=True)
        self.multiplied_ew = tf.Variable(initial_value=tf.zeros(input_dim), trainable=False)

    def call(self, inputs):
        self.multiplied_ew.assign(tf.multiply(inputs[0,:, :, :1], inputs[0,:, :, 1:2]))
        return tf.concat([tf.multiply(tf.exp(self.multiplied_ew), self.alpha),
                          tf.reshape(inputs[0, :, :, 2], [tf.shape(inputs)[1], tf.shape(inputs)[2], 1])], axis=2)


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
