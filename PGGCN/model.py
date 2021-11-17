import tensorflow as tf
from deepchem.feat.mol_graphs import ConvMol

class PGGCN(tf.keras.Model):
    def __init__(self,):
        super(PGGCN, self).__init__()

    def call(self, inputs, training=False):
        atom_features = inputs[0]
        degree_slice = tf.cast(inputs[1], dtype=tf.int32)
        membership = tf.cast(inputs[2], dtype=tf.int32)
        n_samples = tf.cast(inputs[3], dtype=tf.int32)
        deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]