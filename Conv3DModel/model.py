import tensorflow as tf
from .layers import *

class Conv3DModel(tf.keras.Model):

    def __init__(self, features=None):
        super(Conv3DModel, self).__init__()
        self.transforming_layer = TransLayer(features)
        self.conv3d = tf.keras.layers.Conv3D(20, 3, activation='sigmoid')
        self.maxpooling = tf.keras.layers.MaxPooling3D(4, padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.dense_features = tf.keras.layers.Dense(1)
        self.dense_energy = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x, g_pol, g_nonpol = inputs
        x = self.transforming_layer(x)
        x = self.conv3d(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.dense_features(x)
        x = tf.concat([x, g_pol, g_nonpol], axis=1)
        x = self.dense_energy(x)
        return x