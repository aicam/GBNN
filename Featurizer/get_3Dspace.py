import numpy as np
from biopandas.pdb import PandasPdb
import os
from os.path import isfile, join

PATH = '../PDBs'

def get_3D_space(path):
    PDBs = [f for f in os.listdir(path) if isfile(join(path, f))]
    pandaspdb = PandasPdb()
    pdb_dataframes = [PandasPdb().read_pdb(path + p) for p in PDBs]

