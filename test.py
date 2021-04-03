import numpy as np
import tensorflow as tf




a = tf.constant([1, 12, 3, 6, 2, 10])
print(tf.math.argmin(a, axis=-1))