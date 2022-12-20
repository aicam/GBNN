import tensorflow as tf
from deepchem.feat.mol_graphs import ConvMol

from layers import *
class PGGCN(tf.keras.Model):
    def __init__(self, num_atom_features = 80, r_out_channel = 40, c_out_channel = 1024):
        super().__init__()
        self.ruleGraphConvLayer = RuleGraphConvLayer(r_out_channel, num_atom_features)
        self.conv = ConvLayer(c_out_channel)
        self.dense1 = tf.keras.layers.Dense(650, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1,
                 kernel_initializer=tf.keras.initializers.Constant([.5, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]),
                 bias_initializer=tf.keras.initializers.Zeros(), activation=tf.keras.activations.relu)

    def set_physics_info(self, info):
        self.physics_info = info

    def addRule(self, rule, start_index, end_index = None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)

    def call(self, inputs):
        x = self.ruleGraphConvLayer(inputs)
        x = self.conv(x)
        x = self.dense1(x)
        model_var = self.dense2(x)
        print("model var shape ", model_var.shape)
        merged = tf.concat([model_var, self.physics_info], axis=1)
        print("merged shape", merged.shape)
        out = self.dense3(merged)
        return out