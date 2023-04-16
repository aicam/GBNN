import deepchem as dc
import numpy as np
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate
from tensorflow.keras import initializers
import sys
import tensorflow as tf
import random


class GBGraphConvModel(tf.keras.Model):

    def modify_graphgather(self, batch_size):
        self.readout.batch_size = batch_size
        self.batch_size = batch_size

    def __init__(self, batch_size):
        super(GBGraphConvModel, self).__init__()
        self.input_shapes = None
        self.batch_size = batch_size
        self.gc1 = GraphConv(64, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(64, activation_fn=tf.nn.tanh)
        self.batch_norm2 = layers.BatchNormalization()
        self.gp2 = GraphPool()

        self.dense1 = layers.Dense(128, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=self.batch_size, activation_fn=tf.nn.tanh)

        self.dense2 = layers.Dense(64, activation=tf.nn.sigmoid)
        self.dense3 = layers.Dense(1)

        ## Dense for overall

        self.dense4 = layers.Dense(1,
         kernel_initializer=initializers.Constant([.5, -1, 1, 1]),
         bias_initializer=initializers.Zeros())

    #     self.dense4 = layers.Dense(1,
    #          kernel_initializer=initializers.Constant([.5, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1]),
    #          bias_initializer=initializers.Zeros(), activation=tf.keras.activations.relu)

    def call(self, inputs):
        #     x_feat, x_add = inputs[0], inputs[1]
        inputs = inputs[0]
        x = []
        #     input_shapes = [[4822, 75], [11, 2], [4822], [1142, 1], [1635, 2], [2042, 3],
        #                    [3, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10]]
        for i in range(len(self.input_shapes)):
            x.append(tf.reshape(inputs[i][inputs[i] != 1.123456], self.input_shapes[i]))
        for i in range(1, len(self.input_shapes)):
            x[i] = tf.cast(x[i], tf.int32)
        x_add = tf.reshape(inputs[13][inputs[13] != 1.123456], [self.batch_size, 3])

        gc1_output = self.gc1(x)
        batch_norm1_output = self.batch_norm1(gc1_output)
        gp1_output = self.gp1([batch_norm1_output] + x[1:])

        gc2_output = self.gc2([gp1_output] + x[1:])
        batch_norm2_output = self.batch_norm1(gc2_output)
        gp2_output = self.gp2([batch_norm2_output] + x[1:])

        dense1_output = self.dense1(gp2_output)
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm3_output] + x[1:])

        model_var = self.dense2(readout_output)
        model_var = self.dense3(model_var)
        # binding_affinity = tf.concat([model_var, x_add], axis=1)
        # ddg = self.dense4(binding_affinity)

        #     tf.print(self.dense4.weights, output_stream="file://weights.txt", summarize=30)
        #     tf.print(binding_affinity[0], output_stream="file://binding_a.txt", summarize=30)
        #     tf.print(ddg[0], output_stream="file://ddg.txt")
        #     tf.print(model_var, output_stream="file://model_var.txt", summarize=30)
        #     tf.print("-------------------------", output_stream=sys.stdout)
        return model_var

def data_generator(PDBs, x_add):
    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    X = [featurizer.featurize(x)[0] for x in PDBs]

    multiConvMol = ConvMol.agglomerate_mols(X)
    x_preprocessed = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        x_preprocessed.append(multiConvMol.get_deg_adjacency_lists()[i])
    x_preprocessed.append(np.array(x_add))

    x_reshaped = np.full([14, np.max([v.shape[0] for v in x_preprocessed]),
                       np.max([v.shape[1] for v in x_preprocessed if len(v.shape) > 1])], 1.123456)
    for i, j in enumerate(x_preprocessed):
        if len(j.shape) > 1:
            x_reshaped[i][:j.shape[0], :j.shape[1]] = np.array(j)
        else:
            x_reshaped[i][:len(j), :1] = np.array(j).reshape(j.shape[0], 1)
    x_reshaped = x_reshaped.reshape([1] + list(x_reshaped.shape))

    return x_reshaped, [i.shape for i in x_preprocessed]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def get_trained_model(X, y, shapes, batch_size, epochs=25):
    model = GBGraphConvModel(batch_size)
    model.compile(optimizer="rmsprop", loss=root_mean_squared_error)
    K.set_value(model.optimizer.learning_rate, 0.001)
    model.input_shapes = shapes
    hist = model.fit(X, y.reshape([1, -1]), epochs=epochs)
    return hist.history['loss'], model

def test_model(X_test, y_test, size, shapes, model):
    model.input_shapes = shapes
    model.modify_graphgather(size)
    return model.evaluate(X_test, y_test.reshape([1, -1]))