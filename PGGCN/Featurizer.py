import numpy as np
import deepchem as dc
import os
import pandas as pd

class PGGCNFeaturizer:



    def __init__(self, mols_dict, physics_csv, atom_csvs_directory):
        self.mols_dict = mols_dict
        self.physics_df = pd.read_csv(physics_csv)
        self.atoms_csvs_directory = atom_csvs_directory if atom_csvs_directory[-1] == '/' else atom_csvs_directory + '/'

        ## TODO: make compatible with other featurizers
        self.featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)

    def featurize(self):

        # Find all atoms GBNSR6 features csv files
        atoms_csvs = [self.atoms_csvs_directory + atoms_csv for atoms_csv in os.listdir(self.atoms_csvs_directory) if atoms_csv.__contains__('.csv')]

        # Create dictionary with key as PDB id and value as pandas Datafram
        atoms_csvs_dict = {atoms_csv.split('/')[-1].split('.')[0].split('-')[0]: pd.read_csv(atoms_csv).round(3) for atoms_csv in
                           atoms_csvs}


        for mol_name, mol in self.mols_dict.items():
            mol_f = self.featurizer.featurize(mol)[0]
            atoms_gb = atoms_csvs_dict[mol_name]

            for c in mol.GetConformers():
                rdkit_order = c.GetPositions()
            df = pd.DataFrame(columns=atoms_gb.columns)
            for i, atom_pos in enumerate(rdkit_order):
                df = df.append(atoms_gb[(atoms_gb['x'] == atom_pos[0]) & (atoms_gb['y'] == atom_pos[1]) & (
                            atoms_gb['z'] == atom_pos[2])], ignore_index=True)
            atoms_gb = df
            atom_physics_features = atoms_gb[['atomic-charge', 'atomic-radius', 'effective-born-radii']].to_numpy()
            atom_features = np.concatenate([mol_f.atom_features, atom_physics_features], axis=1)


