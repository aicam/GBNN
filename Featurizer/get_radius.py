import tensorflow as tf
from biopandas.pdb import PandasPdb

'''
 Get atom coordinates from pdb file
'''


def get_atoms(pdbfile):
    pdbfile = '/'.join(pdbfile.split('/')[1:])
    pdb = PandasPdb().read_pdb(pdbfile)
    coors = pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return tf.Variable(coors)


'''
 Distance function for atoms
'''


@tf.function
def AtomDistance(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))


'''
 This function has been added to cover R and R_id variables 
 out of exact function to be changed in each iteration
'''


def R_wrapper(coors, M):
    R = tf.reshape(tf.Variable([.1 for i in range(M)], dtype=tf.float32), shape=[1, M])
    R_id = tf.reshape(tf.Variable([1 for i in range(M)], dtype=tf.int32), shape=[1, M])

    @tf.function
    def get_R_matrix(coors, M):
        i = tf.constant(0)

        ## Condition lambda function
        c = lambda i, R, R_id, coors: tf.less(i, coors.get_shape()[0])

        ## Body function
        b = lambda i, R, R_id, coors: (i + 1, tf.concat(
            [R, tf.reshape(tf.cast(tf.sort(AtomDistance(coors[i], coors))[1:M + 1], dtype=tf.float32), [1, M])],
            axis=0),
                                       tf.concat(
                                           [R_id, tf.reshape(tf.argsort(AtomDistance(coors[i], coors))[1:M + 1],
                                                             [1, M])],
                                           axis=0),
                                       coors)

        return tf.while_loop(c, b, loop_vars=[i, R, R_id, coors],
                             shape_invariants=[i.get_shape(), tf.TensorShape([None, M]), tf.TensorShape([None, M]),
                                               coors.get_shape()])

    R_matrix = get_R_matrix(coors, M)
    return R_matrix[1], R_matrix[2]

