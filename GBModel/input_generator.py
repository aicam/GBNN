import numpy as np
import pandas as pd
import os
from os.path import isfile, join

def generate_X(hdf_directory, M):
    inputs = [f for f in os.listdir(hdf_directory) if isfile(join(hdf_directory, f))]
    X = []
    for input in inputs:
        df = pd.read_hdf(hdf_directory + '/' + input)
        R = df['R'].to_numpy().reshape([-1, M])[1:]
        R_id = df['R_id'].to_numpy().reshape([-1, M])[1:]
        B = df['B'].dropna().to_numpy()
        C = df['charges'].dropna().to_numpy()
        B_expa = np.zeros([B.shape[0], R_id.shape[1]])
        C_expa = np.zeros([C.shape[0], R_id.shape[1]])
        for i in range(B.shape[0]):
            B_expa[i] = B[i] * B[R_id[i]]
            C_expa[i] = C[i] * C[R_id[i]]
        new_x = np.zeros([R.shape[0], M, 3])
        new_x[:, :, 0] = R
        new_x[:, :, 1] = B_expa
        new_x[:, :, 2] = C_expa
        X.append(new_x)
    return np.array(X)