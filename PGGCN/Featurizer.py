import numpy as np
import deepchem as dc
import os
import pandas as pd
import rdkit
from rdkit import Chem

class PGGCNFeaturizer:
    def __init__(self, mols_dict, physics_csv, atom_csvs_directory):
        self.mols_dict = mols_dict
        self.physics_df = pd.read_csv(physics_csv)
        self.atoms_csvs_directory = atom_csvs_directory if atom_csvs_directory[-1] == '/' else atom_csvs_directory + '/'

        ## TODO: make compatible with other featurizers
        self.featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)

    def featurize(self):
        # input array will returned to use in model as input
        inputs = []
        # Find all atoms GBNSR6 features csv files
        atoms_csvs = [self.atoms_csvs_directory + atoms_csv for atoms_csv in os.listdir(self.atoms_csvs_directory) if atoms_csv.__contains__('.csv')]

        # Create dictionary with key as PDB id and value as pandas Datafram
        atoms_csvs_dict = {atoms_csv.split('/')[-1].split('.')[0].split('-')[0]: pd.read_csv(atoms_csv).round(3) for atoms_csv in
                           atoms_csvs}

        # featurizing each complex by mols_dict
        for mol_name, mol in self.mols_dict.items():
            # molecule featurized
            mol_f = self.featurizer.featurize(mol)[0]
            # atoms features extracted by Amber
            atoms_gb = atoms_csvs_dict[mol_name]
            # hydrogens are used by Amber but RDKit remove them so Amber features should be re ordered
            # atom positions are unique and can be used to equalize orders
            rdkit_order = None
            for c in mol.GetConformers():
                rdkit_order = c.GetPositions()
            df = pd.DataFrame(columns=atoms_gb.columns)
            for i, atom_pos in enumerate(rdkit_order):
                df = df.append(atoms_gb[(atoms_gb['x'] == atom_pos[0]) & (atoms_gb['y'] == atom_pos[1]) & (
                            atoms_gb['z'] == atom_pos[2])], ignore_index=True)
            atoms_gb = df

            # input features used by PGGCN
            atom_physics_features = atoms_gb[['x', 'y', 'z', 'atomic-charge', 'atomic-radius', 'effective-born-radii']].to_numpy()
            atom_features = np.concatenate([mol_f.atom_features, atom_physics_features], axis=1)

            # generate adjacency list with one hot vector (size 22) which indicates bond type
            # this is a nested list ordered by atom id (idx in rdkit)
            adjacency_list = []
            for atom in mol.GetAtoms():
                atom_bonds = []
                for neighbor in atom.GetNeighbors():
                    bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                    neighbor_idx = neighbor.GetIdx()
                    atom_bonds.append([neighbor_idx, self.convertBONDtoOneHotVector(bond_type)])
                adjacency_list.append(atom_bonds)

            inputs.append([atom_features, adjacency_list])

        return inputs



    @staticmethod
    def convertBONDtoOneHotVector(bond_type):
        BOND_DICT = {dir(Chem.rdchem.BondType)[i]: i for i in range(22)}
        one_hot = np.zeros(22)
        one_hot[BOND_DICT[str(bond_type)]] = 1
        return one_hot



