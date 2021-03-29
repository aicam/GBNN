from GB_Layers import MM_Layer
import tensorflow as tf
import numpy as np

x = tf.ones((3, 3, 3))


print(MM_Layer()(x))

