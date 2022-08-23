import conda_installer

conda_installer.install()

import rdkit
import deepchem as dc
import pandas as pd
import numpy as np

df = pd.read_csv('../Datasets/pdbbind.csv')
df = df.dropna()

# truncate for memory optimization
complex_names_df = df['complex-name'].to_numpy()
PDBs = {}
from os import listdir
from os.path import isfile, join
mypath = '../Datasets/pdbbind_complex/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in onlyfiles:
    if f.split('.')[0] in complex_names_df:
        PDBs.update({f.split('.')[0] : rdkit.Chem.rdmolfiles.MolFromPDBFile(mypath + f)})