import tensorflow as tf
from tensorflow import keras

class MM_Layer(keras.layers.Layer):



    def __init__(self):
        super(MM_Layer, self).__init__()
        self.alpha = tf.Variable(initial_value = tf.ones(1), trainable = True)

    def call(self, inputs):
        multplied_ew = tf.multiply(inputs[:,:,:1], inputs[:,:,1:2])
        return 1 / tf.multiply(tf.exp(multplied_ew), self.alpha)