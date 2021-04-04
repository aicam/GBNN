from GBModel.GB_Layers import MMLayer, GBLayer, FilterLayer
import tensorflow as tf

x = tf.ones((8, 4, 3))

model = tf.keras.models.Sequential()
model.add(MMLayer())
model.add(GBLayer())
model.add(FilterLayer([4, 10]))
model.add(tf.keras.layers.Conv2D(4, (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='relu'))
print(model(x))