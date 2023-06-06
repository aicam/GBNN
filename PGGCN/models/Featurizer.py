import numpy as np
import deepchem as dc
import os
import pandas as pd
import rdkit
from rdkit import Chem
from deepchem.feat import MolecularFeaturizer
from deepchem.feat.graph_features import one_of_k_encoding_unk


class RuleGraphFeaturizer(MolecularFeaturizer):

    def __init__(self):
        self.dtype = object

    def get_atom_feature(self, atom, conformer):

        ## binary features
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Unknown'
            ])
        results += one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                         SP3D, Chem.rdchem.HybridizationType.SP3D2
        ])
        results += atom.GetIsAromatic()

        ## continuous features
        results += atom.GetDegree()
        results += atom.GetImplicitValence()
        results += atom.GetNumRadicalElectrons()
        results += atom.GetFormalCharge()
        results += atom.GetTotalNumHs()
        results += atom.GetMass()
        results += atom.GetAtomicNum()

        ## position
        position = conformer.GetAtomPosition(atom.GetIdx())
        results += [position.x, position.y, position.z]

        ## bond type feature
        bond_vec = [0, 0, 0, 0]
        for bond in atom.GetBonds():
            new_vec = one_of_k_encoding_unk(bond.GetBondType().name, ['SINGLE', 'DOUBLE', 'AROMATIC', 'Other'])
            bond_vec[0] += new_vec[0]
            bond_vec[1] += new_vec[1]
            bond_vec[2] += new_vec[2]
            bond_vec[3] += new_vec[3]
        results += bond_vec

        ## Overall 26 features
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

            # host dataframe is the key name based on RCSB PDB id and a ligand which is binded
            # these data are prepared by Amber tools by GBNSR6 method
            host_df = self.physics_df[self.physics_df['Host'] == mol_name]
            gbnsr6_features = host_df[['gb_Complex_1-4EEL',
                                       'gb_Complex_EELEC', 'gb_Complex_EGB', 'gb_Complex_ESURF', 'pb_complex_VDWAALS',
                                       'gb_guest_1-4EEL', 'gb_guest_EELEC', 'gb_guest_EGB',
                                       'gb_guest_ESURF', 'pb_guest_VDWAALS', 'gb_host_1-4EEL', 'gb_host_EELEC',
                                       'gb_host_EGB', 'gb_host_ESURF', 'pb_host_VDWAALS']].to_numpy()[0]

            inputs.append([atom_features, adjacency_list, gbnsr6_features])


        return inputs



    @staticmethod
    def convertBONDtoOneHotVector(bond_type):
        BOND_DICT = {dir(Chem.rdchem.BondType)[i]: i for i in range(22)}
        one_hot = np.zeros(22)
        one_hot[BOND_DICT[str(bond_type)]] = 1
        return one_hot



