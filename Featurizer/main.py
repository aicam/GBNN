from Featurizer.get_born import *
from Featurizer.get_charges import *
from Featurizer.get_distance import *
import pandas as pd
import numpy as np

'''
 Experimental function to imitate born radii
'''
def fake_born(fake_file):
    f = open(fake_file)
    B = []
    for l in f:
        if l[:4] != "rinv":
            continue
        B.append(float(list(filter(None, l.split(' ')))[2].replace('\n', '')))
    return B

'''
 Use all feature extractor functions to generate dataframe 
'''
def get_pdb_dataframe(pdbfile):
    C = np.array(get_pdb_charges(pdbfile))
    B = np.array(get_pdb_born(pdbfile))
    # B = np.array(fake_born('get_born/fake_o'))
    R, R_id = R_wrapper(get_atoms(pdbfile), 12)
    df_dict = {'charges': C,
                       'R': R.numpy(),
                       'R_id': R_id.numpy(),
                       'B': B}
    df = pd.DataFrame({k : pd.Series(v.reshape([-1])) for k, v in df_dict.items()})
    return df

# TODO: complete this function
# def transform_matrices(C, B, R, R_id):
#     C_neighbors = np.zeros([len(C), R.shape[1]])
#     B_neighbors = np.zeros([len(B)])

