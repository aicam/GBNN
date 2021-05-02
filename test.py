import tensorflow as tf
import numpy as np
from Conv3DModel.layers import *
from Conv3DModel.model import *

conv3dModel = Conv3DModel(3)
x = np.ones((10, 50, 50, 50, 3))
g_pol = np.ones((10, 1))
g_nonpol = np.ones((10, 1))
y = np.ones((10, 1))
conv3dModel.compile(loss='mse', optimizer='adam')
conv3dModel.fit((x, g_pol, g_nonpol), y, epoch=2)

