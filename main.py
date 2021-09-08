from GBModel.GB_Layers import MMLayer, GBLayer, FilterLayer
import tensorflow as tf
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

