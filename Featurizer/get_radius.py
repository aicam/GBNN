import tensorflow as tf
from biopandas.pdb import PandasPdb
import numpy as np

def get_atoms(pdbfile):
    pdb = PandasPdb().read_pdb(pdbfile)
    coors = pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return tf.Variable(coors)


@tf.function
def AtomDistance(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))

R = tf.reshape(tf.Variable([.1 for i in range(12)], dtype=tf.float32), shape=[1, 12])
@tf.function
def get_R_matrix(coors, M):

    R_id = tf.zeros(shape=[coors.shape[0], M])
    i = tf.constant(0)
    c = lambda i, R, coors: tf.less(i, coors.get_shape()[0])
    # print( tf.reshape(tf.cast(tf.sort(AtomDistance(coors[i], coors)[:M]), dtype=tf.float32), [1, M]))
    b = lambda i, R, coors: (i + 1, tf.concat([R, tf.reshape(tf.cast(tf.sort(AtomDistance(coors[i], coors))[1:M + 1], dtype=tf.float32), [1, M])], axis=0), coors)
    def body(i):
        # R[i].assign(tf.cast(tf.sort(AtomDistance(coors[i], coors)[:M]), dtype=tf.float32))
        return i
    # for i in range(coors.shape[0]):
    #      R[i].assign(tf.cast(tf.sort(AtomDistance(coors[i], coors)[:M]), dtype=tf.float32))
    return tf.while_loop(c, b, loop_vars=[i, R, coors], shape_invariants=[i.get_shape(), tf.TensorShape([None, M]), coors.get_shape()])


a = get_R_matrix(get_atoms('/home/ali/calstate/amber/ras-raf.pdb'), 12)
print(a[1][3112])

