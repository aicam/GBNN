import conda_installer

conda_installer.install()

import rdkit
import deepchem as dc
import pandas as pd
import numpy as np
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

### Model ###

from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate
from tensorflow.keras import initializers


class GBGraphConvModel(tf.keras.Model):

    def modify_graphgather(self, batch_size):
        self.readout.batch_size = batch_size
        self.batch_size = batch_size

    def __init__(self, batch_size):
        super(GBGraphConvModel, self).__init__()
        self.counter = 0
        self.input_shapes = None
        self.batch_size = batch_size
        self.gc1 = GraphConv(32, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(32, activation_fn=tf.nn.tanh)
        self.batch_norm2 = layers.BatchNormalization()
        self.gp2 = GraphPool()

        self.dense1 = layers.Dense(64, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=self.batch_size, activation_fn=tf.nn.tanh)

        self.dense2 = layers.Dense(1)
        self.dense3 = layers.Dense(1,
                                   kernel_initializer=initializers.Constant(
                                       [.5, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]),
                                   bias_initializer=initializers.Zeros())

    def call(self, inputs):
        x = inputs
        x_add = X_add
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
        binding_affinity = tf.concat([model_var, x_add], axis=1)
        ans = self.dense3(binding_affinity)
        ans = tf.reshape(ans, [1, -1])
        return ans


### Model ###

featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
TRAIN_SET = .8

df = pd.read_csv('../Datasets/pdbbind.csv')
df = df.dropna()

# truncate for memory optimization
df = df[:3]

complex_names_df = df['complex-name'].to_numpy()
PDBs = {}
from os import listdir
from os.path import isfile, join

pdbs_path = '../Datasets/pdbbind_complex/'
onlyfiles = [f for f in listdir(pdbs_path) if isfile(join(pdbs_path, f))]
for f in onlyfiles:
    if f.split('.')[0] in complex_names_df:
        try:
            PDBs.update({f.split('.')[0]: rdkit.Chem.rdmolfiles.MolFromPDBFile(pdbs_path + f)})
        except:
            continue

X_ids = []
X = []
for k in PDBs.keys():
    X_ids.append(k)
    X.append(featurizer.featurize(PDBs[k]))

X_disk = []
y_disk = []
for i in range(len(X_ids)):
    X_disk.append(X[i])
    y_disk.append(df[df['complex-name'] == X_ids[i]]['ddg'].to_numpy()[0])
w_disk = np.ones([5, 12])
X_add = []
for i in range(len(X_ids)):
    X_add.append(df[df['complex-name'] == X_ids[i]][[i for i in df.columns if ((('gb-' in i))
                                                                               and ('-etot' not in i)) or (
                                                                 '-vdwaals' in i)]].to_numpy()[0])
train_dataset = dc.data.DiskDataset.from_numpy(X=X_disk, y=y_disk, w=w_disk, ids=X_ids)
X_add = np.array(X_add)

batch_size = len(df)


def loss_function(y, y_hat, w):
    return tf.keras.losses.mse(y_hat, y)


model = dc.models.KerasModel(GBGraphConvModel(batch_size), loss=loss_function)


def data_generator(dataset, epochs=30):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, epochs,
                                                                     deterministic=False, pad_batches=True)):
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        #        inputs.append(X_add)
        #         print(inputs[13])
        labels = y_b
        weights = [w_b]
        yield (inputs, labels, weights)


his = model.fit_generator(data_generator(train_dataset, 1))
print(his)
np.save("res.txt", np.array(his))
