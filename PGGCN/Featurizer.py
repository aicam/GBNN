import numpy as np
import deepchem as dc
import os

class PGGCNFeaturizer:



    def __init__(self, PDB_directory, physics_csv, atom_csv_directory):
        self.PDB_directory = PDB_directory
        self.physics_csv = physics_csv
        self.atom_csv_directory = atom_csv_directory

    def featurize(self):
