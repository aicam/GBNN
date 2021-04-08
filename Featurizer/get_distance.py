import tensorflow as tf
from biopandas.pdb import PandasPdb
import os
import subprocess

'''
 Get atom coordinates from pdb file
'''
def get_atoms_coordinates(pdbfile):
    ## Change directory to run bash script
    os.chdir("get_distance")

    ## Run script
    result = subprocess.Popen(["./get_distance.sh " + pdbfile], executable="/bin/bash",
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    s, e = result.communicate()
    while (s.decode("utf-8") == ""):
        if (e.decode("utf-8") != ""):
            exit(e.decode("utf-8"))
        continue

    pandaspdb = PandasPdb()
    ppdb_df = pandaspdb.read_pdb_from_list(pdb_lines=s.decode("utf-8").split("\n"))
    os.chdir('..')
    coors = ppdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return tf.Variable(coors)

## ambpdb
## h++
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
