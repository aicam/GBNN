from GBModel.GB_Layers import MMLayer, GBLayer, FilterLayer
import tensorflow as tf
import numpy as np

x = np.ones((1, 8, 4, 3))
from deepchem import models
# x1 = MMLayer()(x)
# print(x1.shape)
# x2 = GBLayer()(x1)
# print(x2.shape)
# print(FilterLayer([4, 10])(x2).shape)
model = tf.keras.models.Sequential()
model.add(MMLayer())
model.add(GBLayer())
model.add(FilterLayer([4, 10]))
model.add(tf.keras.layers.Conv2D(4, (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.compile(loss='MSE', optimizer='adam')

print(model.fit(x, np.zeros([1, 1]), epochs=2))
