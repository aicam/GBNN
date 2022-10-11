from GBModel.GB_Layers import MMLayer, GBLayer, FilterLayer
import tensorflow as tf
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from keras.callbacks import EarlyStopping
import numpy as np
from deepchem.feat.mol_graphs import ConvMol
x = np.ones((1, 8, 4, 3))
import deepchem as dc
dc.data.DiskDataset.from_numpy()
from deepchem.feat import AtomicConvFeaturizer
dc.feat.MolGraphConvFeaturizer

model = tf.keras.models.Sequential()
model.add(MMLayer())
model.add(GBLayer())
model.add(FilterLayer([4, 10]))
model.add(tf.keras.layers.Conv2D(4, (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.compile(loss='MSE', optimizer='adam')

from deepchem.metrics import to_one_hot
from deepchem.models.layers import GraphConv, GraphPool, GraphGather