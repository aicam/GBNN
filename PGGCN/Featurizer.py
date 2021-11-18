import numpy as np
import deepchem as dc
import os
import pandas as pd

class PGGCNFeaturizer:



    def __init__(self, mols_dict, physics_csv, atom_csvs_directory):
        self.mols_dict = mols_dict
        self.physics_df = pd.read_csv(physics_csv)
        self.atoms_csvs_directory = atom_csvs_directory if atom_csvs_directory[-1] == '/' else atom_csvs_directory + '/'

    def featurize(self):
        atoms_csvs = [self.atoms_csvs_directory + atoms_csv for atoms_csv in os.listdir(self.atoms_csvs_directory) if atoms_csv.__contains__('.csv')]
        for mol_name in self.mols_dict.keys():

