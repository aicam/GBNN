import tensorflow as tf
import deepchem as dc


dc.models.GraphConvModel(n_tasks, mode='classification')


dc.feat.ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol

class MultiGraphConvLayer(tf.keras.Model):
    def __init__(self, operations = [[-1], [tf.math.reduce_sum]]):
        super(MultiGraphConvLayer, self).__init__()

    def call(self, inputs):
        features = inputs[0]
        adjacency_list = inputs[1]
        physics_features = inputs[2]

