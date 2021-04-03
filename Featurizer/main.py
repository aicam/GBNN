from Featurizer.get_born import *
from Featurizer.get_charges import *
from Featurizer.get_radius import *
import pandas as pd
import numpy as np

def fake_born(fake_file):
    f = open(fake_file)
    B = []
    for l in f:
        if l[:4] != "rinv":
            continue
        B.append(float(list(filter(None, l.split(' ')))[2].replace('\n', '')))
    return B

def get_pdb_dataframe(pdbfile):
    C = np.array(get_pdb_charges(pdbfile))
    IB = np.array(fake_born('get_born/fake_o'))
    R, R_id = R_wrapper(get_atoms(pdbfile), 12)
    df_dict = {'charges': C,
                       'R': R.numpy(),
                       'R_id': R_id.numpy(),
                       'IB': IB}
    df = pd.DataFrame({k : pd.Series(v.reshape([-1])) for k, v in df_dict.items()})
    df.to_hdf("ras-raf.h5", "rasraf")

get_pdb_dataframe('/home/ali/calstate/amber/ras-raf.pdb')
