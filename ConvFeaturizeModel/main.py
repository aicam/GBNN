from GBModel.GB_Layers import MMLayer, GBLayer, FilterLayer
import tensorflow as tf
from deepchem.models.layers import GraphConv, GraphPool, GraphGather

import numpy as np
from deepchem.feat.mol_graphs import ConvMol
x = np.ones((1, 8, 4, 3))
model = tf.keras.models.Sequential()
model.add(MMLayer())
model.add(GBLayer())
model.add(FilterLayer([4, 10]))
model.add(tf.keras.layers.Conv2D(4, (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.compile(loss='MSE', optimizer='adam')

from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

x_preprocessed = []
x_shapes = []
## for X train
for i in range(len(X)):
    multiConvMol = ConvMol.agglomerate_mols([X[i]])
    new_x_preprocessed = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
    for k in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        new_x_preprocessed.append(multiConvMol.get_deg_adjacency_lists()[k])
    new_x_preprocessed.append(np.array(x_add[i]))
    print([f.shape for f in new_x_preprocessed])
    break
    new_x_preprocessed.append(np.array([f.shape[0] for f in new_x_preprocessed]))
    x_preprocessed.append(new_x_preprocessed)