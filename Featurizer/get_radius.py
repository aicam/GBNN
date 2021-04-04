import tensorflow as tf
from biopandas.pdb import PandasPdb


def get_atoms(pdbfile):
    pdbfile = '/'.join(pdbfile.split('/')[1:])
    pdb = PandasPdb().read_pdb(pdbfile)
    coors = pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return tf.Variable(coors)


@tf.function
def AtomDistance(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))


def R_wrapper(coors, M):
    R = tf.reshape(tf.Variable([.1 for i in range(M)], dtype=tf.float32), shape=[1, M])
    R_id = tf.reshape(tf.Variable([1 for i in range(M)], dtype=tf.int32), shape=[1, M])

    @tf.function
    def get_R_matrix(coors, M):
        i = tf.constant(0)
        c = lambda i, R, R_id, coors: tf.less(i, coors.get_shape()[0])
        b = lambda i, R, R_id, coors: (i + 1, tf.concat(
            [R, tf.reshape(tf.cast(tf.sort(AtomDistance(coors[i], coors))[1:M + 1], dtype=tf.float32), [1, M])],
            axis=0),
                                       tf.concat(
                                           [R_id, tf.reshape(tf.argsort(AtomDistance(coors[i], coors))[1:M + 1],
                                                                  [1, M])],
                                           axis=0),
                                       coors)
        return tf.while_loop(c, b, loop_vars=[i, R, R_id, coors],
                             shape_invariants=[i.get_shape(), tf.TensorShape([None, M]), tf.TensorShape([None, M]), coors.get_shape()])
    R_matrix = get_R_matrix(coors, M)
    return R_matrix[1], R_matrix[2]


# a = R_wrapper(get_atoms('/home/ali/calstate/amber/ras-raf.pdb'), 12)
# print(a)
