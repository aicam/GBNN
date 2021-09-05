from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers
import deepchem as dc
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
batch_size = 100

class MyGraphConvModel(tf.keras.Model):

  def __init__(self):
    super(MyGraphConvModel, self).__init__()
    self.gc1 = GraphConv(128, activation_fn=tf.nn.relu)
    self.batch_norm1 = layers.BatchNormalization()
    self.gp1 = GraphPool()

    self.gc2 = GraphConv(1, activation_fn=tf.nn.relu)
    self.batch_norm2 = layers.BatchNormalization()
    self.gp2 = GraphPool()

    self.dense1 = layers.Dense(3, activation=tf.nn.relu)
    self.batch_norm3 = layers.BatchNormalization()
    self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

    self.dense2 = layers.Dense(1)
    # self.logits = layers.Reshape((n_tasks, 2))
    #add polar none polar here
    self.dense3=layers.Dense(1)
    self.ReLU = layers.ReLU()

  def call(self, inputs):
    x, g_pol, g_nonpol = inputs
    gc1_output = self.gc1(x)
    print(x)
    batch_norm1_output = self.batch_norm1(gc1_output)
    gp1_output = self.gp1([batch_norm1_output] + x[1:])
    gc2_output = self.gc2([gp1_output] + x[1:])
    batch_norm2_output = self.batch_norm1(gc2_output)
    gp2_output = self.gp2([batch_norm2_output] + x[1:])

    dense1_output = self.dense1(gp2_output)
    batch_norm3_output = self.batch_norm3(dense1_output)
    readout_output = self.readout([batch_norm3_output] + x[1:])

    logits_output = self.logits(self.dense2(readout_output))
    x = tf.concat([logits_output, g_pol, g_nonpol], axis=1)
    return self.dense3(x)