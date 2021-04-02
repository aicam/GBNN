import tensorflow as tf
from biopandas.pdb import PandasPdb
import numpy as np

def get_atoms(pdbfile):
    pdb = PandasPdb().read_pdb(pdbfile)
    coors = pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return tf.Variable(coors)


@tf.function
def AtomDistance(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
    return dist


@tf.function
def get_R_matrix(coors, R, M):
    # R = tf.Variable(tf.zeros([coors.shape[0], M]), dtype=tf.float32)
    R_id = tf.zeros(shape=[coors.shape[0], M])
    i = tf.constant(0)
    c = lambda i, R: tf.less(i, coors.get_shape()[0])
    b = lambda i, R: (i, R[i].assign(tf.cast(tf.sort(AtomDistance(coors[i], coors)[:M]), dtype=tf.float32)))
    def body(i, R):
        with tf.control_dependencies([R[i].assign(tf.cast(tf.sort(AtomDistance(coors[i], coors)[:M]), dtype=tf.float32))]):
            R = tf.identity(R)
        return i, R
        # for i in range(coors.shape[0]):
    #     dist = AtomDistance(coors[i], coors)
    tf.while_loop(c, b, loop_vars=[i, R])
    return R

coors = get_atoms('/home/ali/calstate/amber/ras-raf.pdb')
R = tf.Variable(tf.zeros([coors.shape[0], 12]), dtype=tf.float32)
a = get_R_matrix(coors, R, 12)
print(a)
